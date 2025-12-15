"""
Streamlit Real-Time IoT Monitoring Dashboard
==============================================

Interactive web app that simulates real-time sensor data streaming and
runs anomaly detection on the fly. Supports switching between LSTM,
Transformer, and Isolation Forest models for live comparison.

Features:
  - Real-time data simulation with adjustable speed
  - Live anomaly detection with LSTM, Transformer, or Isolation Forest
  - Visual alerts when anomalies are detected
  - Configurable anomaly injection for demo purposes
  - Clear error messages when models haven't been trained yet

Usage:
    streamlit run src/streamlit_app.py
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# make sure we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    LSTMAutoencoder, TransformerAutoencoder, build_model,
    apply_normalization, SENSOR_COLS, SEQUENCE_LENGTH
)


# ── Page Config ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IoT Anomaly Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .alert-box {
        background: linear-gradient(135deg, #e74c3c20, #e74c3c10);
        border-left: 4px solid #e74c3c;
        padding: 12px 16px; border-radius: 6px; margin: 8px 0;
    }
    .normal-box {
        background: linear-gradient(135deg, #27ae6020, #27ae6010);
        border-left: 4px solid #27ae60;
        padding: 12px 16px; border-radius: 6px; margin: 8px 0;
    }
    .warn-box {
        background: linear-gradient(135deg, #f39c1220, #f39c1210);
        border-left: 4px solid #f39c12;
        padding: 12px 16px; border-radius: 6px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Model Loading ──────────────────────────────────────────────────────

@st.cache_resource
def load_dl_model(model_type: str):
    """Load a trained deep learning model and threshold — cached to avoid reloading."""
    if model_type == 'lstm':
        model = LSTMAutoencoder()
        model_path = os.path.join('models', 'lstm_autoencoder.pt')
        threshold_path = os.path.join('models', 'lstm_threshold.npy')
    else:
        model = TransformerAutoencoder()
        model_path = os.path.join('models', 'transformer_autoencoder.pt')
        threshold_path = os.path.join('models', 'transformer_threshold.npy')

    stats_path = os.path.join('models', 'norm_stats.json')

    if not os.path.exists(model_path):
        return None, None, None

    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    threshold = float(np.load(threshold_path))

    with open(stats_path) as f:
        norm_stats = json.load(f)

    return model, threshold, norm_stats


@st.cache_resource
def load_iforest_model():
    """Load the trained Isolation Forest and its fitted scaler."""
    model_path = os.path.join('models', 'isolation_forest.pkl')
    scaler_path = os.path.join('models', 'iforest_scaler.pkl')

    if not os.path.exists(model_path):
        return None, None

    iforest = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return iforest, scaler


@st.cache_data
def load_sensor_data():
    """Load the generated sensor data for simulation."""
    data_path = os.path.join('data', 'sensor_data.csv')
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path, parse_dates=['timestamp'])


# ── Detection ──────────────────────────────────────────────────────────

def detect_anomaly_dl(model, threshold, norm_stats, window: np.ndarray) -> tuple:
    """Run DL model on a single window. Returns (is_anomaly, recon_error)."""
    normalized = apply_normalization(window, norm_stats)
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        recon = model(tensor)
        error = torch.mean((tensor - recon) ** 2).item()

    return error > threshold, error


def add_iforest_features(window: np.ndarray) -> np.ndarray:
    """
    Engineer the same temporal features used during IForest training.
    This has to match exactly what isolation_forest.py computed, or
    the predictions will be garbage.
    """
    df = pd.DataFrame(window, columns=SENSOR_COLS)

    feature_cols = list(SENSOR_COLS)
    for col_name in SENSOR_COLS:
        df[f'{col_name}_rolling_mean_12'] = df[col_name].rolling(12, min_periods=1).mean()
        df[f'{col_name}_rolling_std_12'] = df[col_name].rolling(12, min_periods=1).std().fillna(0)
        df[f'{col_name}_diff'] = df[col_name].diff().fillna(0)
        rolling_mean = df[col_name].rolling(24, min_periods=1).mean()
        rolling_std = df[col_name].rolling(24, min_periods=1).std().fillna(1)
        df[f'{col_name}_zscore'] = (df[col_name] - rolling_mean) / rolling_std.replace(0, 1)

    all_feature_cols = [c for c in df.columns if c in SENSOR_COLS or
                        any(s in c for s in ['rolling', 'diff', 'zscore'])]

    # use the last row — the current time step
    return df[all_feature_cols].values[-1:, :]


def detect_anomaly_iforest(iforest, scaler, window: np.ndarray) -> tuple:
    """Run Isolation Forest on the last sample in the window."""
    features = add_iforest_features(window)
    scaled = scaler.transform(features)
    raw_pred = iforest.predict(scaled)
    score = float(-iforest.decision_function(scaled)[0])
    is_anomaly = raw_pred[0] == -1
    return is_anomaly, score


# ── Preview Chart ──────────────────────────────────────────────────────

def build_preview_chart(df: pd.DataFrame, n_samples: int = 500) -> go.Figure:
    """Build a static preview chart of the first n_samples."""
    sample = df.head(n_samples)
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Temperature (°C)', 'Pressure (kPa)', 'Vibration (mm/s)', 'Current (A)'
    ])
    for sensor_idx, sensor_col in enumerate(SENSOR_COLS):
        row, col_pos = (sensor_idx // 2) + 1, (sensor_idx % 2) + 1
        fig.add_trace(go.Scatter(
            x=sample['timestamp'], y=sample[sensor_col],
            mode='lines', line=dict(color='#3498db', width=1), showlegend=False
        ), row=row, col=col_pos)

    fig.update_layout(height=400, template='plotly_dark',
                      margin=dict(t=40, b=20, l=40, r=20))
    return fig


# ── Main App ───────────────────────────────────────────────────────────

def main():
    st.title("🔧 Industrial IoT Anomaly Detection")
    st.markdown("*Real-time monitoring simulation with deep learning*")

    df = load_sensor_data()
    if df is None:
        st.error("No sensor data found. Run `python src/generate_data.py` first.")
        return

    # ── Sidebar ──
    st.sidebar.header("⚙️ Controls")

    model_choice = st.sidebar.radio(
        "Detection Model",
        ['LSTM Autoencoder', 'Transformer Autoencoder', 'Isolation Forest'],
        help="Switch between trained models"
    )

    # load the appropriate model
    model, threshold, norm_stats, iforest, iforest_scaler = None, None, None, None, None
    use_iforest = (model_choice == 'Isolation Forest')

    if use_iforest:
        iforest, iforest_scaler = load_iforest_model()
        model_loaded = iforest is not None
    else:
        model_type = 'lstm' if 'LSTM' in model_choice else 'transformer'
        model, threshold, norm_stats = load_dl_model(model_type)
        model_loaded = model is not None

    speed = st.sidebar.slider("Simulation Speed", 1, 50, 10, help="Samples per second")
    window_size = st.sidebar.slider("Display Window", 50, 500, 200)
    inject_anomaly = st.sidebar.checkbox("Inject Random Anomalies")
    anomaly_intensity = st.sidebar.slider("Anomaly Intensity", 1.0, 10.0, 5.0) if inject_anomaly else 1.0

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Active Model:** {model_choice}")

    if model_loaded:
        if not use_iforest:
            total_params = sum(p.numel() for p in model.parameters())
            st.sidebar.text(f"Parameters: {total_params:,}")
            st.sidebar.text(f"Threshold: {threshold:.6f}")
        else:
            st.sidebar.text(f"Estimators: {iforest.n_estimators}")
            st.sidebar.text("Contamination: 0.07")
    else:
        st.sidebar.warning("Model not loaded — run `python run_all.py` first")

    # ── Metrics Row ──
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    # ── Session State ──
    if 'sim_idx' not in st.session_state:
        st.session_state.sim_idx = SEQUENCE_LENGTH
        st.session_state.anomaly_count = 0
        st.session_state.total_checked = 0

    # controls
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    start_btn = btn_col1.button("▶️ Start", use_container_width=True)
    stop_btn = btn_col2.button("⏹️ Stop", use_container_width=True)
    reset_btn = btn_col3.button("🔄 Reset", use_container_width=True)

    if reset_btn:
        st.session_state.sim_idx = SEQUENCE_LENGTH
        st.session_state.anomaly_count = 0
        st.session_state.total_checked = 0
        st.rerun()

    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    if start_btn and model_loaded:
        # ── Simulation loop ──
        for step in range(st.session_state.sim_idx, len(df) - 1):
            if stop_btn:
                break

            st.session_state.sim_idx = step
            start_idx = max(0, step - window_size)
            window_df = df.iloc[start_idx:step + 1].copy()

            is_anomaly = False
            recon_error = 0.0

            if step >= SEQUENCE_LENGTH:
                window_data = df.iloc[step - SEQUENCE_LENGTH + 1:step + 1][SENSOR_COLS].values.copy()

                if inject_anomaly and np.random.random() < 0.02:
                    spike_sensor = np.random.randint(0, len(SENSOR_COLS))
                    window_data[-1, spike_sensor] += (
                        anomaly_intensity * df[SENSOR_COLS[spike_sensor]].std()
                    )

                if use_iforest:
                    is_anomaly, recon_error = detect_anomaly_iforest(
                        iforest, iforest_scaler, window_data
                    )
                else:
                    is_anomaly, recon_error = detect_anomaly_dl(
                        model, threshold, norm_stats, window_data
                    )

            st.session_state.total_checked += 1
            if is_anomaly:
                st.session_state.anomaly_count += 1

            # update metric cards
            with metric_col1:
                st.metric("Sample", f"{step:,}")
            with metric_col2:
                st.metric("Anomalies", st.session_state.anomaly_count)
            with metric_col3:
                rate = (st.session_state.anomaly_count / max(1, st.session_state.total_checked)) * 100
                st.metric("Rate", f"{rate:.1f}%")
            with metric_col4:
                label = "Score" if use_iforest else "Error"
                st.metric(label, f"{recon_error:.4f}")

            # update live chart
            fig = make_subplots(rows=2, cols=2, subplot_titles=[
                'Temperature (°C)', 'Pressure (kPa)', 'Vibration (mm/s)', 'Current (A)'
            ])
            for sensor_idx, sensor_col in enumerate(SENSOR_COLS):
                row, col_pos = (sensor_idx // 2) + 1, (sensor_idx % 2) + 1
                fig.add_trace(go.Scatter(
                    x=window_df['timestamp'], y=window_df[sensor_col],
                    mode='lines', line=dict(color='#3498db', width=1), showlegend=False
                ), row=row, col=col_pos)

                anom_mask = window_df['is_anomaly'] == 1
                if anom_mask.any():
                    fig.add_trace(go.Scatter(
                        x=window_df.loc[anom_mask, 'timestamp'],
                        y=window_df.loc[anom_mask, sensor_col],
                        mode='markers', marker=dict(color='red', size=6), showlegend=False
                    ), row=row, col=col_pos)

            fig.update_layout(height=400, template='plotly_dark',
                              margin=dict(t=40, b=20, l=40, r=20))
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            score_label = "Score" if use_iforest else "Threshold"
            score_ref = f"{recon_error:.4f}" if use_iforest else f"{threshold:.6f}"

            if is_anomaly:
                status_placeholder.markdown(
                    f'<div class="alert-box">⚠️ <strong>ANOMALY</strong> at sample {step} '
                    f'— {score_label}: {score_ref}</div>',
                    unsafe_allow_html=True
                )
            else:
                status_placeholder.markdown(
                    f'<div class="normal-box">✅ Normal — Error: {recon_error:.4f}</div>',
                    unsafe_allow_html=True
                )

            time.sleep(1.0 / speed)

    elif start_btn and not model_loaded:
        # this branch was previously missing — would silently show nothing
        st.error(
            f"**{model_choice}** is not loaded. "
            "Run the full training pipeline first: `python run_all.py`"
        )
        chart_placeholder.plotly_chart(build_preview_chart(df), use_container_width=True)

    else:
        # idle state — show data preview before simulation starts
        st.info("Select a model in the sidebar, then click **Start** to begin simulation.")
        chart_placeholder.plotly_chart(build_preview_chart(df), use_container_width=True)


if __name__ == '__main__':
    main()
