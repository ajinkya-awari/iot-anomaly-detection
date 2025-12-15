"""
Isolation Forest Baseline for Anomaly Detection
=================================================

Implements sklearn's Isolation Forest as a traditional ML baseline to
compare against the deep learning approaches (LSTM and Transformer
autoencoders). IForest works by randomly partitioning feature space —
anomalies get isolated in fewer splits (Liu et al., 2008).

This is a solid baseline but it doesn't capture temporal dependencies
natively, which is exactly why we need the DL approaches for time-series.
We compensate by engineering rolling window features.

Outputs:
    models/isolation_forest.pkl         — trained model
    models/iforest_scaler.pkl           — fitted scaler
    data/test_with_iforest_preds.csv    — test set with IForest predictions
    outputs/iforest_scores.png          — anomaly score distribution

Usage:
    python src/isolation_forest.py
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Config ─────────────────────────────────────────────────────────────

SEED = 42
# contamination roughly matches our known anomaly ratio
# bumped up slightly because iforest tends to be conservative
CONTAMINATION = 0.07
N_ESTIMATORS = 200          # more trees = more stable, diminishing returns past ~250
MAX_FEATURES = 1.0          # use all features — we only have 4 sensors + engineered

DATA_DIR = 'data'
MODEL_DIR = 'models'
OUTPUT_DIR = 'outputs'

SENSOR_COLS = ['temperature_C', 'pressure_kPa', 'vibration_mm_s', 'current_draw_A']


def add_temporal_features(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Engineer time-based features to give IForest some temporal awareness.
    
    Without these, IForest is purely point-based and misses sequential
    patterns like drifts. Rolling stats bridge that gap somewhat.
    
    TODO: try adding fourier features (sin/cos of hour-of-day) — might help
          catch time-dependent normal ranges without manual thresholding
    """
    df = df.copy()

    for col in sensor_cols:
        # rolling statistics over ~1 hour window (12 samples at 5-min rate)
        df[f'{col}_rolling_mean_12'] = df[col].rolling(12, min_periods=1).mean()
        df[f'{col}_rolling_std_12'] = df[col].rolling(12, min_periods=1).std().fillna(0)

        # rate of change — helps catch sudden spikes
        df[f'{col}_diff'] = df[col].diff().fillna(0)

        # rolling z-score — how unusual is the current reading vs recent history
        rolling_mean = df[col].rolling(24, min_periods=1).mean()
        rolling_std = df[col].rolling(24, min_periods=1).std().fillna(1)
        df[f'{col}_zscore'] = (df[col] - rolling_mean) / rolling_std.replace(0, 1)

        # TODO: add inter-sensor correlation features — anomalies often show up
        #       as broken correlations between sensors that normally move together

    return df


def plot_anomaly_scores(scores: np.ndarray, labels: np.ndarray,
                        save_path: str):
    """Visualize IForest anomaly scores vs ground truth."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # score distribution split by class
    axes[0].hist(scores[labels == 0], bins=60, alpha=0.7, color='#27ae60',
                 label='Normal', density=True)
    axes[0].hist(scores[labels == 1], bins=60, alpha=0.7, color='#e74c3c',
                 label='Anomaly', density=True)
    axes[0].set_title('Isolation Forest Anomaly Score Distribution')
    axes[0].set_xlabel('Anomaly Score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # scores plotted over time
    axes[1].plot(scores, linewidth=0.4, alpha=0.6, color='#2c3e50')
    anomaly_idx = np.where(labels == 1)[0]
    axes[1].scatter(anomaly_idx, scores[anomaly_idx], c='red', s=5, alpha=0.5,
                    label='True anomaly', zorder=5)
    axes[1].set_title('Anomaly Scores Over Time')
    axes[1].set_ylabel('Score')
    axes[1].set_xlabel('Sample Index')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {save_path}')


def main():
    np.random.seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading data...')
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv'))

    # feature engineering
    print('Engineering temporal features...')
    train_eng = add_temporal_features(train_df, SENSOR_COLS)
    test_eng = add_temporal_features(test_df, SENSOR_COLS)

    feature_cols = [c for c in train_eng.columns
                    if c not in ['timestamp', 'is_anomaly', 'anomaly_type']]
    print(f'  Total features: {len(feature_cols)} (4 raw + {len(feature_cols) - 4} engineered)')

    # normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_eng[feature_cols].values)
    X_test = scaler.transform(test_eng[feature_cols].values)

    # fit isolation forest
    print(f'Training Isolation Forest (n_estimators={N_ESTIMATORS})...')
    iforest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_features=MAX_FEATURES,
        random_state=SEED,
        n_jobs=-1,
    )
    iforest.fit(X_train)

    # save
    joblib.dump(iforest, os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'iforest_scaler.pkl'))
    print(f'  Model saved to {MODEL_DIR}/isolation_forest.pkl')

    # predict — IForest returns -1 for anomaly, 1 for normal, we flip to 0/1
    raw_preds = iforest.predict(X_test)
    test_preds = (raw_preds == -1).astype(int)

    # anomaly scores — negate so higher = more anomalous (more intuitive)
    anomaly_scores = -iforest.decision_function(X_test)

    test_df['iforest_pred'] = test_preds
    test_df['iforest_score'] = anomaly_scores
    test_df.to_csv(os.path.join(DATA_DIR, 'test_with_iforest_preds.csv'), index=False)

    print(f'  Anomalies detected: {test_preds.sum()} / {len(test_preds)}')

    plot_anomaly_scores(anomaly_scores, test_df['is_anomaly'].values,
                        os.path.join(OUTPUT_DIR, 'iforest_scores.png'))

    print('Isolation Forest training complete.')


if __name__ == '__main__':
    main()
