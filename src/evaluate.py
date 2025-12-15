"""
Model Comparison & Evaluation (3-Way + Ensemble)
==================================================

Compares LSTM Autoencoder vs Transformer Autoencoder vs Isolation Forest
vs a soft-voting Ensemble on the IoT anomaly detection task. Computes
precision, recall, F1 and generates comparison visualizations.

The three base models represent different points on the complexity /
interpretability spectrum:
  - Isolation Forest: fast, no temporal awareness, strong on point anomalies
  - LSTM Autoencoder: captures sequential patterns via recurrence
  - Transformer Autoencoder: captures long-range dependencies via attention

The ensemble combines their continuous anomaly scores via min-max normalization
and weighted averaging — this often outperforms any single model because each
catches different failure modes (see per_anomaly_type.png after running).

Outputs:
    outputs/comparison_metrics.json     — metrics for all four models
    outputs/comparison_table.png        — side-by-side metrics table
    outputs/timeseries_anomalies.png    — main viz with anomalies in red
    outputs/confusion_matrices.png      — confusion matrices
    outputs/roc_curves.png              — ROC and PR curves
    outputs/per_anomaly_type.png        — breakdown by anomaly type
    outputs/ensemble_analysis.png       — ensemble score distribution

Usage:
    python src/evaluate.py
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec


DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
MODEL_DIR = 'models'

# weights for the soft-vote ensemble — equal weighting by default.
# could be tuned via a held-out validation set if we had one
ENSEMBLE_WEIGHTS = {
    'LSTM Autoencoder': 1.0,
    'Transformer Autoencoder': 1.0,
    'Isolation Forest': 1.0,
}

MODEL_COLORS = {
    'LSTM Autoencoder': '#2980b9',
    'Transformer Autoencoder': '#8e44ad',
    'Isolation Forest': '#e67e22',
    'Ensemble': '#27ae60',
}


def load_predictions() -> pd.DataFrame:
    """
    Merge all three models' predictions into a single dataframe.

    Each model saved its own CSV; we join on index since they all
    come from the same test split.
    """
    lstm_df = pd.read_csv(os.path.join(DATA_DIR, 'test_with_lstm_preds.csv'))
    transformer_df = pd.read_csv(os.path.join(DATA_DIR, 'test_with_transformer_preds.csv'))
    iforest_df = pd.read_csv(os.path.join(DATA_DIR, 'test_with_iforest_preds.csv'))

    merged = lstm_df.copy()
    merged['transformer_pred'] = transformer_df['transformer_pred']
    merged['transformer_error'] = transformer_df['transformer_error']
    merged['iforest_pred'] = iforest_df['iforest_pred']
    merged['iforest_score'] = iforest_df['iforest_score']
    merged['timestamp'] = pd.to_datetime(merged['timestamp'])

    return merged


def compute_ensemble_score(df: pd.DataFrame, valid_mask: np.ndarray) -> tuple:
    """
    Soft-vote ensemble: normalize each model's continuous score to [0, 1],
    then compute a weighted average. Threshold at 0.5 for hard predictions.

    Min-max normalization is done per-model on the valid subset only.
    Equal weights by default — the three models are quite different so
    averaging tends to smooth out each one's idiosyncratic false positives.

    TODO: try learning optimal weights via a small labeled validation set
          using Nelder-Mead or scipy.optimize — probably 5-10% F1 gain
    """
    scaler = MinMaxScaler()

    # gather raw continuous scores for valid samples
    lstm_scores = df.loc[valid_mask, 'lstm_error'].values.reshape(-1, 1)
    trans_scores = df.loc[valid_mask, 'transformer_error'].values.reshape(-1, 1)
    iforest_scores = df.loc[valid_mask, 'iforest_score'].values.reshape(-1, 1)

    # normalize each to [0, 1]
    lstm_norm = scaler.fit_transform(lstm_scores).flatten()
    trans_norm = scaler.fit_transform(trans_scores).flatten()
    iforest_norm = scaler.fit_transform(iforest_scores).flatten()

    w_lstm = ENSEMBLE_WEIGHTS['LSTM Autoencoder']
    w_trans = ENSEMBLE_WEIGHTS['Transformer Autoencoder']
    w_if = ENSEMBLE_WEIGHTS['Isolation Forest']
    total_weight = w_lstm + w_trans + w_if

    ensemble_score = (w_lstm * lstm_norm + w_trans * trans_norm + w_if * iforest_norm) / total_weight

    # threshold at 0.5 for hard binary predictions
    ensemble_pred = (ensemble_score >= 0.5).astype(int)

    return ensemble_score, ensemble_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Compute standard classification metrics."""
    metrics = {
        'model': model_name,
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1_score': round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    return metrics


def compute_per_anomaly_type_metrics(df: pd.DataFrame, pred_col: str,
                                      model_name: str) -> dict:
    """
    Break down recall by anomaly type — this tells us which failure modes
    each model catches best. Super useful for deciding deployment strategy.

    TODO: also compute per-type precision if we had enough data per type
    """
    results = {}
    anomaly_types = df.loc[df['is_anomaly'] == 1, 'anomaly_type'].unique()

    for atype in anomaly_types:
        mask = df['anomaly_type'] == atype
        y_true_sub = df.loc[mask, 'is_anomaly'].values
        y_pred_sub = df.loc[mask, pred_col].values
        recall = recall_score(y_true_sub, y_pred_sub, zero_division=0)
        results[atype] = round(recall, 4)

    return results


# ── Plotting ───────────────────────────────────────────────────────────

def plot_confusion_matrices(y_true: np.ndarray, predictions: dict, save_path: str):
    """Side-by-side confusion matrices for all models."""
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    for ax, (name, pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_true, pred)
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                        fontsize=13, color=color, fontweight='bold')

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(name, fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {save_path}')


def plot_roc_curves(y_true: np.ndarray, scores: dict, save_path: str):
    """ROC and Precision-Recall curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for name, score_values in scores.items():
        color = MODEL_COLORS.get(name, '#95a5a6')

        fpr, tpr, _ = roc_curve(y_true, score_values)
        roc_auc = auc(fpr, tpr)
        linestyle = '--' if name == 'Ensemble' else '-'
        axes[0].plot(fpr, tpr, color=color, linewidth=2, linestyle=linestyle,
                     label=f'{name} (AUC={roc_auc:.3f})')

        prec, rec, _ = precision_recall_curve(y_true, score_values)
        ap = average_precision_score(y_true, score_values)
        axes[1].plot(rec, prec, color=color, linewidth=2, linestyle=linestyle,
                     label=f'{name} (AP={ap:.3f})')

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curves')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {save_path}')


def plot_per_anomaly_type(type_metrics: dict, save_path: str):
    """
    Grouped bar chart showing recall per anomaly type per model.
    This is where the interesting story emerges — each model has
    different strengths on different fault types.
    """
    all_types = sorted(set().union(*[set(v.keys()) for v in type_metrics.values()]))
    n_types = len(all_types)
    n_models = len(type_metrics)

    fig, ax = plt.subplots(figsize=(12, 5))
    bar_width = 0.8 / n_models
    x = np.arange(n_types)

    for i, (model_name, type_recall) in enumerate(type_metrics.items()):
        values = [type_recall.get(t, 0) for t in all_types]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=model_name,
                       color=MODEL_COLORS.get(model_name, '#95a5a6'), alpha=0.85)

        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Anomaly Type')
    ax.set_ylabel('Recall')
    ax.set_title('Detection Recall by Anomaly Type')
    ax.set_xticks(x)
    ax.set_xticklabels(all_types, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {save_path}')


def plot_timeseries_anomalies(df: pd.DataFrame, save_path: str):
    """
    The main visualization — time series with anomalies highlighted.
    Hero image for the README.
    """
    sensor_cols = ['temperature_C', 'pressure_kPa', 'vibration_mm_s', 'current_draw_A']
    sensor_labels = ['Temperature (°C)', 'Pressure (kPa)', 'Vibration (mm/s)', 'Current (A)']

    fig = plt.figure(figsize=(18, 16))
    n_rows = len(sensor_cols) + 2  # sensors + LSTM error + Transformer error
    gs = gridspec.GridSpec(n_rows, 1,
                           height_ratios=[1] * len(sensor_cols) + [0.6, 0.6],
                           hspace=0.35)

    true_anom = df['is_anomaly'] == 1

    for i, (col, label) in enumerate(zip(sensor_cols, sensor_labels)):
        ax = fig.add_subplot(gs[i])
        ax.plot(df['timestamp'], df[col], linewidth=0.5, alpha=0.7, color='#2c3e50')
        ax.scatter(df.loc[true_anom, 'timestamp'], df.loc[true_anom, col],
                   c='#e74c3c', s=10, alpha=0.6, zorder=5, label='True Anomaly')
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    # LSTM reconstruction error panel
    lstm_threshold = np.load(os.path.join(MODEL_DIR, 'lstm_threshold.npy'))
    ax_lstm = fig.add_subplot(gs[-2])
    ax_lstm.fill_between(df['timestamp'], df['lstm_error'], alpha=0.4, color='#2980b9')
    ax_lstm.axhline(y=lstm_threshold, color='red', linestyle='--', linewidth=1.5,
                     label=f'LSTM Threshold ({lstm_threshold:.4f})')
    ax_lstm.set_ylabel('LSTM Error', fontsize=9)
    ax_lstm.legend(fontsize=8)
    ax_lstm.grid(True, alpha=0.2)
    ax_lstm.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # Transformer reconstruction error panel
    trans_threshold = np.load(os.path.join(MODEL_DIR, 'transformer_threshold.npy'))
    ax_trans = fig.add_subplot(gs[-1])
    ax_trans.fill_between(df['timestamp'], df['transformer_error'], alpha=0.4, color='#8e44ad')
    ax_trans.axhline(y=trans_threshold, color='red', linestyle='--', linewidth=1.5,
                      label=f'Transformer Threshold ({trans_threshold:.4f})')
    ax_trans.set_ylabel('Transformer Error', fontsize=9)
    ax_trans.set_xlabel('Date')
    ax_trans.legend(fontsize=8)
    ax_trans.grid(True, alpha=0.2)
    ax_trans.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    fig.suptitle('Industrial IoT Anomaly Detection — Three-Model Comparison',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {save_path}')


def plot_comparison_table(metrics_list: list, save_path: str):
    """Clean comparison table as an image — 4 rows now including ensemble."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Type']
    model_types = {
        'LSTM Autoencoder': 'Deep Learning',
        'Transformer Autoencoder': 'Attention-Based',
        'Isolation Forest': 'Classical ML',
        'Ensemble': 'Ensemble',
    }
    rows = []
    for m in metrics_list:
        rows.append([
            m['model'],
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1_score']:.4f}",
            model_types.get(m['model'], '—'),
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # header styling
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # alternate row colors and highlight best F1
    best_f1_idx = np.argmax([m['f1_score'] for m in metrics_list])
    for i in range(len(rows)):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)
        if i == best_f1_idx:
            table[i + 1, 4].set_facecolor('#27ae60')
            table[i + 1, 4].set_text_props(color='white', fontweight='bold')

    plt.title('Model Comparison Results (including Ensemble)',
              fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {save_path}')


def plot_ensemble_analysis(ensemble_score: np.ndarray, y_true: np.ndarray,
                           save_path: str):
    """
    Visualize the ensemble score distribution — how well separated
    are normal vs anomalous samples?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    normal_scores = ensemble_score[y_true == 0]
    anomaly_scores = ensemble_score[y_true == 1]

    axes[0].hist(normal_scores, bins=60, alpha=0.7, color='#27ae60',
                 label=f'Normal (n={len(normal_scores):,})', density=True)
    axes[0].hist(anomaly_scores, bins=60, alpha=0.7, color='#e74c3c',
                 label=f'Anomaly (n={len(anomaly_scores):,})', density=True)
    axes[0].axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold = 0.5')
    axes[0].set_title('Ensemble Score Distribution')
    axes[0].set_xlabel('Normalized Ensemble Score')
    axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # score over time
    axes[1].plot(ensemble_score, linewidth=0.4, alpha=0.6, color='#2c3e50')
    anomaly_idx = np.where(y_true == 1)[0]
    axes[1].scatter(anomaly_idx, ensemble_score[anomaly_idx], c='red', s=5,
                    alpha=0.5, label='True anomaly', zorder=5)
    axes[1].axhline(0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].set_title('Ensemble Score Over Time')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Score')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Soft-Vote Ensemble Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {save_path}')


# ── Main ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading predictions from all three models...')
    df = load_predictions()

    y_true = df['is_anomaly'].values
    lstm_pred = df['lstm_pred'].values
    transformer_pred = df['transformer_pred'].values
    iforest_pred = df['iforest_pred'].values

    # only evaluate where both DL models have valid predictions
    valid_mask = (df['lstm_error'] > 0) & (df['transformer_error'] > 0)
    y_valid = y_true[valid_mask]
    lstm_valid = lstm_pred[valid_mask]
    trans_valid = transformer_pred[valid_mask]
    iforest_valid = iforest_pred[valid_mask]

    print(f'Evaluation samples: {valid_mask.sum():,} (excluded {(~valid_mask).sum()} '
          f'initial samples without DL predictions)')

    # ── compute ensemble ──
    print('\nComputing soft-vote ensemble...')
    ensemble_score, ensemble_pred = compute_ensemble_score(df, valid_mask)
    print(f'  Ensemble anomalies flagged: {ensemble_pred.sum()} / {len(ensemble_pred)}')

    # ── compute all metrics ──
    lstm_metrics = compute_metrics(y_valid, lstm_valid, 'LSTM Autoencoder')
    trans_metrics = compute_metrics(y_valid, trans_valid, 'Transformer Autoencoder')
    iforest_metrics = compute_metrics(y_valid, iforest_valid, 'Isolation Forest')
    ensemble_metrics = compute_metrics(y_valid, ensemble_pred, 'Ensemble')

    all_metrics = [lstm_metrics, trans_metrics, iforest_metrics, ensemble_metrics]

    print('\n── Results ──────────────────────────────────────────────────')
    for m in all_metrics:
        marker = ' ← best F1' if m['f1_score'] == max(x['f1_score'] for x in all_metrics) else ''
        print(f"  {m['model']:30s}  Prec: {m['precision']:.4f}  "
              f"Rec: {m['recall']:.4f}  F1: {m['f1_score']:.4f}{marker}")
    print('─────────────────────────────────────────────────────────────')

    # ── per-anomaly-type breakdown ──
    print('\n── Per-Anomaly-Type Recall ──────────────────────────────────')
    df_valid = df[valid_mask].copy()
    df_valid['ensemble_pred'] = ensemble_pred

    type_metrics = {}
    model_cols = [
        ('LSTM Autoencoder', 'lstm_pred'),
        ('Transformer Autoencoder', 'transformer_pred'),
        ('Isolation Forest', 'iforest_pred'),
        ('Ensemble', 'ensemble_pred'),
    ]
    for name, pred_col in model_cols:
        type_recall = compute_per_anomaly_type_metrics(df_valid, pred_col, name)
        type_metrics[name] = type_recall
        print(f'  {name}:')
        for atype, recall in sorted(type_recall.items()):
            print(f'    {atype:15s}  {recall:.4f}')

    # ── save metrics ──
    output_data = {
        'overall': all_metrics,
        'per_anomaly_type': type_metrics,
    }
    metrics_path = os.path.join(OUTPUT_DIR, 'comparison_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f'\n  Metrics saved to {metrics_path}')

    # ── generate all visualizations ──
    print('\nGenerating comparison plots...')

    predictions = {
        'LSTM Autoencoder': lstm_valid,
        'Transformer Autoencoder': trans_valid,
        'Isolation Forest': iforest_valid,
        'Ensemble': ensemble_pred,
    }
    plot_confusion_matrices(y_valid, predictions,
                            os.path.join(OUTPUT_DIR, 'confusion_matrices.png'))

    # raw continuous scores — roc_curve handles different scales fine,
    # no need to normalize here (that was only needed for the ensemble vote)
    raw_lstm = df.loc[valid_mask, 'lstm_error'].values
    raw_trans = df.loc[valid_mask, 'transformer_error'].values
    raw_if = df.loc[valid_mask, 'iforest_score'].values

    scores = {
        'LSTM Autoencoder': raw_lstm,
        'Transformer Autoencoder': raw_trans,
        'Isolation Forest': raw_if,
        'Ensemble': ensemble_score,
    }
    plot_roc_curves(y_valid, scores, os.path.join(OUTPUT_DIR, 'roc_curves.png'))

    plot_per_anomaly_type(type_metrics, os.path.join(OUTPUT_DIR, 'per_anomaly_type.png'))
    plot_timeseries_anomalies(df, os.path.join(OUTPUT_DIR, 'timeseries_anomalies.png'))
    plot_comparison_table(all_metrics, os.path.join(OUTPUT_DIR, 'comparison_table.png'))
    plot_ensemble_analysis(ensemble_score, y_valid,
                           os.path.join(OUTPUT_DIR, 'ensemble_analysis.png'))

    # ── detailed classification reports ──
    print('\n── LSTM Autoencoder ──')
    print(classification_report(y_valid, lstm_valid, target_names=['Normal', 'Anomaly']))
    print('── Transformer Autoencoder ──')
    print(classification_report(y_valid, trans_valid, target_names=['Normal', 'Anomaly']))
    print('── Isolation Forest ──')
    print(classification_report(y_valid, iforest_valid, target_names=['Normal', 'Anomaly']))
    print('── Ensemble ──')
    print(classification_report(y_valid, ensemble_pred, target_names=['Normal', 'Anomaly']))

    print('\nEvaluation complete.')


if __name__ == '__main__':
    main()
