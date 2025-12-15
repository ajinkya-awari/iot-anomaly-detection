"""
Synthetic Industrial IoT Data Generator
========================================

Generates multi-variate time-series sensor data mimicking a real power grid
monitoring setup. Each sensor stream (temperature, pressure, vibration,
current draw) follows its own base distribution with layered noise,
diurnal cycles, and injected anomalies of different types.

The anomaly injection is designed to be realistic — we don't just add
Gaussian spikes. Real industrial anomalies include gradual drifts,
sudden step changes, and correlated multi-sensor failures.

Outputs:
    data/sensor_data.csv          — full labeled dataset
    data/train_data.csv           — normal-only training split
    data/test_data.csv            — test split with anomalies
    outputs/data_overview.png     — quick visual sanity check

Usage:
    python src/generate_data.py
"""

import os
import sys
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ── Configuration ──────────────────────────────────────────────────────

SEED = 42
NUM_DAYS = 60                   # two months of data — enough for seasonal patterns
SAMPLE_RATE_MIN = 5             # one reading every 5 minutes
ANOMALY_RATIO = 0.05            # roughly 5% anomalous — realistic for industrial settings
TRAIN_RATIO = 0.7               # first 70% for training (no anomalies injected there)

# sensor baseline params — based on typical industrial ranges
# these were loosely calibrated from publicly available SCADA datasets
SENSOR_CONFIG = {
    'temperature_C': {
        'base': 72.0,
        'noise_std': 1.5,
        'diurnal_amp': 4.0,       # day-night cycle amplitude
        'drift_rate': 0.002,      # slow upward drift per sample
        'weekend_factor': 0.6,    # lower activity on weekends
    },
    'pressure_kPa': {
        'base': 101.3,
        'noise_std': 0.8,
        'diurnal_amp': 0.5,
        'drift_rate': -0.001,
        'weekend_factor': 0.8,
    },
    'vibration_mm_s': {
        'base': 2.5,
        'noise_std': 0.3,
        'diurnal_amp': 0.2,
        'drift_rate': 0.0005,
        'weekend_factor': 0.4,    # machines mostly idle on weekends
    },
    'current_draw_A': {
        'base': 15.0,
        'noise_std': 0.5,
        'diurnal_amp': 3.0,       # current peaks during daytime load
        'drift_rate': 0.001,
        'weekend_factor': 0.5,
    },
}

OUTPUT_DIR = 'data'
PLOT_DIR = 'outputs'


def set_seeds(seed: int = SEED):
    """Lock down all RNGs for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def generate_timestamps(n_days: int, sample_rate_min: int) -> pd.DatetimeIndex:
    """Create a DatetimeIndex at fixed intervals starting Jan 1, 2024."""
    start = datetime(2024, 1, 1)
    n_samples = (n_days * 24 * 60) // sample_rate_min
    timestamps = pd.date_range(start=start, periods=n_samples, freq=f'{sample_rate_min}min')
    return timestamps


def generate_base_signal(n_samples: int, config: dict,
                         timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Produces a single sensor channel with:
      - constant base + slow linear drift
      - diurnal (24h) sinusoidal cycle
      - weekend modulation (lower activity)
      - Gaussian noise
      - random walk perturbations (makes it look less synthetic)
    
    TODO: add seasonal component for multi-month datasets — right now
          60 days isn't long enough for it to matter much
    """
    t = np.arange(n_samples)

    base = config['base'] + config['drift_rate'] * t

    # 24h cycle — period is 288 samples at 5-min intervals
    diurnal = config['diurnal_amp'] * np.sin(2 * np.pi * t / 288)

    # weekend modulation — dampen activity on Sat/Sun
    # this is a simple approach; real plants have more complex schedules
    is_weekend = np.array([ts.weekday() >= 5 for ts in timestamps]).astype(float)
    weekend_mod = 1.0 - (1.0 - config['weekend_factor']) * is_weekend

    noise = np.random.normal(0, config['noise_std'], n_samples)

    # random walk component gives that "wandering baseline" feel
    walk = np.cumsum(np.random.normal(0, config['noise_std'] * 0.02, n_samples))

    # occasional micro-bursts — short 2-3 sample perturbations that are
    # within normal range but add realism (think: valve opening/closing)
    bursts = np.zeros(n_samples)
    burst_indices = np.random.choice(n_samples, size=n_samples // 200, replace=False)
    for idx in burst_indices:
        duration = random.randint(2, 5)
        end = min(idx + duration, n_samples)
        bursts[idx:end] = np.random.normal(0, config['noise_std'] * 1.5)

    signal = (base + diurnal + noise + walk + bursts) * weekend_mod
    return signal


def inject_anomalies(df: pd.DataFrame, anomaly_ratio: float,
                     train_end_idx: int) -> pd.DataFrame:
    """
    Injects anomalies ONLY in the test portion of the data.
    
    Anomaly types (designed to match real industrial failure modes):
      - spike: sudden large deviation (sensor malfunction, power surge)
      - drift: gradual shift over 20-50 samples (calibration error, wear)
      - flatline: sensor stuck at one value (connection dropout)
      - correlated: multiple sensors go haywire simultaneously (system failure)
      - oscillation: rapid periodic fluctuation (mechanical resonance)
    
    TODO: add "contextual anomaly" type where values are normal in isolation
          but abnormal given the time-of-day context — these are the hardest
          to detect and would be a good stress test
    """
    df = df.copy()
    df['is_anomaly'] = 0
    df['anomaly_type'] = 'normal'

    test_indices = np.arange(train_end_idx, len(df))
    n_anomaly_events = int(len(test_indices) * anomaly_ratio) // 5

    sensor_cols = [c for c in df.columns
                   if c not in ['timestamp', 'is_anomaly', 'anomaly_type']]
    anomaly_types = ['spike', 'drift', 'flatline', 'correlated', 'oscillation']

    # pick random start points for anomaly windows, leaving buffer at the end
    if len(test_indices) < 80:
        print('  Warning: test set too small for meaningful anomaly injection')
        return df

    anomaly_starts = np.random.choice(test_indices[:-60], size=n_anomaly_events,
                                       replace=False)

    for start in anomaly_starts:
        atype = random.choice(anomaly_types)
        sensor = random.choice(sensor_cols)

        if atype == 'spike':
            # sharp spike lasting 1-3 samples — like a voltage transient
            duration = random.randint(1, 3)
            end = min(start + duration, len(df))
            magnitude = df[sensor].std() * random.uniform(4, 8)
            sign = random.choice([-1, 1])
            df.loc[start:end-1, sensor] += sign * magnitude
            df.loc[start:end-1, 'is_anomaly'] = 1
            df.loc[start:end-1, 'anomaly_type'] = 'spike'

        elif atype == 'drift':
            # gradual drift over 20-50 samples — sneaky, hard to catch
            # resembles calibration error or slow bearing degradation
            duration = random.randint(20, 50)
            end = min(start + duration, len(df))
            drift_values = np.linspace(0, df[sensor].std() * 3, end - start)
            df.loc[start:end-1, sensor] += drift_values
            df.loc[start:end-1, 'is_anomaly'] = 1
            df.loc[start:end-1, 'anomaly_type'] = 'drift'

        elif atype == 'flatline':
            # sensor freezes — super common in real deployments when connection drops
            duration = random.randint(10, 30)
            end = min(start + duration, len(df))
            frozen_val = df.loc[start, sensor]
            df.loc[start:end-1, sensor] = frozen_val
            df.loc[start:end-1, 'is_anomaly'] = 1
            df.loc[start:end-1, 'anomaly_type'] = 'flatline'

        elif atype == 'correlated':
            # multi-sensor failure — the scary one in production
            # when a transformer overheats you see temp + vibration + current all spike
            duration = random.randint(5, 15)
            end = min(start + duration, len(df))
            affected = random.sample(sensor_cols, k=min(3, len(sensor_cols)))
            for s in affected:
                magnitude = df[s].std() * random.uniform(3, 6)
                df.loc[start:end-1, s] += magnitude
            df.loc[start:end-1, 'is_anomaly'] = 1
            df.loc[start:end-1, 'anomaly_type'] = 'correlated'

        elif atype == 'oscillation':
            # rapid oscillation — think mechanical resonance or feedback loop
            # high frequency relative to normal diurnal cycle
            duration = random.randint(15, 40)
            end = min(start + duration, len(df))
            t_local = np.arange(end - start)
            osc = df[sensor].std() * 2.5 * np.sin(2 * np.pi * t_local / 4)
            df.loc[start:end-1, sensor] += osc
            df.loc[start:end-1, 'is_anomaly'] = 1
            df.loc[start:end-1, 'anomaly_type'] = 'oscillation'

    return df


def plot_data_overview(df: pd.DataFrame, save_path: str):
    """Quick visual check — mostly for debugging during development."""
    sensor_cols = [c for c in df.columns
                   if c not in ['timestamp', 'is_anomaly', 'anomaly_type']]

    fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(16, 3 * len(sensor_cols)),
                              sharex=True)

    anomaly_mask = df['is_anomaly'] == 1
    colors = {'spike': '#e74c3c', 'drift': '#f39c12', 'flatline': '#9b59b6',
              'correlated': '#e67e22', 'oscillation': '#1abc9c'}

    for ax, col in zip(axes, sensor_cols):
        ax.plot(df['timestamp'], df[col], linewidth=0.5, alpha=0.8, color='#2c3e50')

        # color anomalies by type for easier debugging
        for atype, color in colors.items():
            mask = df['anomaly_type'] == atype
            if mask.any():
                ax.scatter(df.loc[mask, 'timestamp'], df.loc[mask, col],
                           c=color, s=8, alpha=0.6, label=atype, zorder=5)

        ax.set_ylabel(col.replace('_', ' '), fontsize=9)
        ax.legend(loc='upper right', fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[-1].set_xlabel('Date')
    fig.suptitle('IoT Sensor Data Overview (anomalies color-coded by type)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {save_path}')


def main():
    set_seeds()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print('Generating synthetic IoT sensor data...')
    timestamps = generate_timestamps(NUM_DAYS, SAMPLE_RATE_MIN)
    n_samples = len(timestamps)
    print(f'  Total samples: {n_samples:,} ({NUM_DAYS} days at {SAMPLE_RATE_MIN}-min intervals)')

    # generate each sensor channel independently
    data = {'timestamp': timestamps}
    for sensor_name, config in SENSOR_CONFIG.items():
        data[sensor_name] = generate_base_signal(n_samples, config, timestamps)

    df = pd.DataFrame(data)

    # inject anomalies only in the test portion
    train_end = int(len(df) * TRAIN_RATIO)
    df = inject_anomalies(df, ANOMALY_RATIO, train_end)

    anomaly_count = df['is_anomaly'].sum()
    print(f'  Anomalies injected: {anomaly_count} ({anomaly_count/len(df)*100:.1f}%)')
    print(f'  Anomaly type breakdown:')
    for atype, count in df.loc[df['is_anomaly'] == 1, 'anomaly_type'].value_counts().items():
        print(f'    {atype}: {count}')

    # save full dataset
    df.to_csv(os.path.join(OUTPUT_DIR, 'sensor_data.csv'), index=False)

    # split into train (normal only) and test
    train_df = df.iloc[:train_end].copy()
    assert train_df['is_anomaly'].sum() == 0, 'Bug: anomalies leaked into training data!'
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_data.csv'), index=False)

    test_df = df.iloc[train_end:].copy()
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_data.csv'), index=False)

    print(f'  Train samples: {len(train_df):,} | Test samples: {len(test_df):,}')

    # quick visual sanity check
    plot_data_overview(df, os.path.join(PLOT_DIR, 'data_overview.png'))
    print('Done.')


if __name__ == '__main__':
    main()
