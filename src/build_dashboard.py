"""
Interactive HTML Dashboard Generator
======================================

Creates a self-contained HTML dashboard with Plotly.js for interactive
exploration of the 4-model anomaly detection results (LSTM, Transformer,
Isolation Forest, Ensemble). Everything is embedded in a single file —
no server needed.

The dashboard downsamples data to keep the HTML under 3MB — the raw
CSVs have 5000+ rows which would bloat the file otherwise.

Outputs:
    outputs/dashboard.html   — self-contained interactive dashboard

Usage:
    python src/build_dashboard.py
"""

import os
import json

import numpy as np
import pandas as pd


DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
MODEL_DIR = 'models'

# downsample to keep HTML file size reasonable
MAX_DISPLAY_POINTS = 2000


def downsample_df(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """
    Downsample while preserving all anomaly points.
    Normal points get uniformly sampled, anomalies are always kept.
    """
    if len(df) <= max_points:
        return df

    anomaly_df = df[df['is_anomaly'] == 1]
    normal_df = df[df['is_anomaly'] == 0]

    n_normal_needed = max(max_points - len(anomaly_df), 100)
    if len(normal_df) > n_normal_needed:
        normal_sampled = normal_df.iloc[::len(normal_df) // n_normal_needed][:n_normal_needed]
    else:
        normal_sampled = normal_df

    result = pd.concat([normal_sampled, anomaly_df]).sort_index()
    return result


def generate_dashboard_html(test_df: pd.DataFrame, metrics: dict) -> str:
    """Build the complete HTML dashboard with embedded Plotly charts."""
    sensor_cols = ['temperature_C', 'pressure_kPa', 'vibration_mm_s', 'current_draw_A']
    sensor_labels = ['Temperature (°C)', 'Pressure (kPa)', 'Vibration (mm/s)', 'Current Draw (A)']

    # downsample for performance
    display_df = downsample_df(test_df, MAX_DISPLAY_POINTS)

    timestamps = display_df['timestamp'].tolist()
    anomaly_mask = display_df['is_anomaly'] == 1

    # build sensor traces for plotly
    sensor_traces = ""
    for i, (col, label) in enumerate(zip(sensor_cols, sensor_labels)):
        visible = 'true' if i == 0 else 'false'
        values = display_df[col].tolist()
        anom_ts = display_df.loc[anomaly_mask, 'timestamp'].tolist()
        anom_vals = display_df.loc[anomaly_mask, col].tolist()

        sensor_traces += f"""
        sensorTraces.push({{
            x: {json.dumps(timestamps)},
            y: {json.dumps(values)},
            type: 'scatter', mode: 'lines',
            name: '{label}', line: {{color: '#3498db', width: 1}},
            visible: {visible}, sensorGroup: {i}
        }});
        sensorTraces.push({{
            x: {json.dumps(anom_ts)},
            y: {json.dumps(anom_vals)},
            type: 'scatter', mode: 'markers',
            name: 'Anomalies', marker: {{color: '#e74c3c', size: 6}},
            visible: {visible}, sensorGroup: {i}
        }});
"""

    # load thresholds
    lstm_threshold = float(np.load(os.path.join(MODEL_DIR, 'lstm_threshold.npy'))) \
        if os.path.exists(os.path.join(MODEL_DIR, 'lstm_threshold.npy')) else 0
    trans_threshold = float(np.load(os.path.join(MODEL_DIR, 'transformer_threshold.npy'))) \
        if os.path.exists(os.path.join(MODEL_DIR, 'transformer_threshold.npy')) else 0

    recon_lstm = display_df['lstm_error'].tolist() if 'lstm_error' in display_df.columns else []
    recon_trans = display_df['transformer_error'].tolist() if 'transformer_error' in display_df.columns else []

    # anomaly type breakdown
    anom_types = test_df.loc[test_df['is_anomaly'] == 1, 'anomaly_type'].value_counts()
    anom_type_labels = anom_types.index.tolist()
    anom_type_values = anom_types.values.tolist()

    # extract metrics — now 4 models (added ensemble)
    overall = metrics.get('overall', [{}, {}, {}, {}])
    lstm_m = overall[0] if len(overall) > 0 else {}
    trans_m = overall[1] if len(overall) > 1 else {}
    iforest_m = overall[2] if len(overall) > 2 else {}
    ensemble_m = overall[3] if len(overall) > 3 else {}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IoT Anomaly Detection Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0a0e17;
            --bg-card: #1a2332;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent: #3b82f6;
            --green: #10b981;
            --red: #ef4444;
            --orange: #f59e0b;
            --purple: #8b5cf6;
            --border: #1e293b;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-bottom: 1px solid var(--border);
            padding: 1.5rem 2rem;
        }}
        .header h1 {{
            font-size: 1.5rem; font-weight: 600;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .header p {{ color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.25rem; }}
        .dashboard {{ max-width: 1400px; margin: 0 auto; padding: 1.5rem; }}
        .stats-row {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem; margin-bottom: 1.5rem;
        }}
        .stat-card {{
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 12px; padding: 1rem; text-align: center;
        }}
        .stat-card .value {{ font-size: 1.8rem; font-weight: 700; line-height: 1.2; }}
        .stat-card .label {{
            color: var(--text-secondary); font-size: 0.75rem;
            text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.2rem;
        }}
        .chart-grid {{
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 1.5rem; margin-bottom: 1.5rem;
        }}
        .chart-card {{
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 12px; padding: 1.2rem;
        }}
        .chart-card.full-width {{ grid-column: 1 / -1; }}
        .chart-card h3 {{ font-size: 0.95rem; font-weight: 600; margin-bottom: 0.8rem; }}
        .metrics-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        .metrics-table th {{
            background: rgba(59, 130, 246, 0.1); color: var(--accent);
            padding: 0.7rem; text-align: center; font-weight: 600;
            border-bottom: 2px solid var(--border);
        }}
        .metrics-table td {{
            padding: 0.7rem; text-align: center; border-bottom: 1px solid var(--border);
        }}
        .metrics-table tr:hover {{ background: rgba(59, 130, 246, 0.05); }}
        .badge {{
            display: inline-block; padding: 0.15rem 0.5rem;
            border-radius: 4px; font-size: 0.7rem; font-weight: 600;
        }}
        .badge-blue {{ background: rgba(59, 130, 246, 0.15); color: var(--accent); }}
        .badge-purple {{ background: rgba(139, 92, 246, 0.15); color: var(--purple); }}
        .badge-orange {{ background: rgba(245, 158, 11, 0.15); color: var(--orange); }}
        .badge-green {{ background: rgba(16, 185, 129, 0.15); color: var(--green); }}
        .sensor-tabs {{ display: flex; gap: 0.5rem; margin-bottom: 0.8rem; flex-wrap: wrap; }}
        .sensor-tab {{
            padding: 0.4rem 1rem; background: rgba(30,41,59,0.8);
            border: 1px solid var(--border); border-radius: 6px;
            cursor: pointer; font-size: 0.8rem; color: var(--text-secondary);
            transition: all 0.2s;
        }}
        .sensor-tab:hover {{ border-color: var(--accent); color: var(--text-primary); }}
        .sensor-tab.active {{ background: var(--accent); color: white; border-color: var(--accent); }}
        @media (max-width: 768px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
            .stats-row {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Industrial IoT Anomaly Detection Dashboard</h1>
        <p>Four-model comparison: LSTM Autoencoder | Transformer Autoencoder | Isolation Forest | Ensemble</p>
    </div>
    <div class="dashboard">
        <div class="stats-row">
            <div class="stat-card">
                <div class="value" style="color: var(--accent);">{len(test_df):,}</div>
                <div class="label">Total Samples</div>
            </div>
            <div class="stat-card">
                <div class="value" style="color: var(--red);">{anomaly_mask.sum():,}</div>
                <div class="label">True Anomalies</div>
            </div>
            <div class="stat-card">
                <div class="value" style="color: var(--accent);">{lstm_m.get('f1_score', '-')}</div>
                <div class="label">LSTM F1</div>
            </div>
            <div class="stat-card">
                <div class="value" style="color: var(--purple);">{trans_m.get('f1_score', '-')}</div>
                <div class="label">Transformer F1</div>
            </div>
            <div class="stat-card">
                <div class="value" style="color: var(--orange);">{iforest_m.get('f1_score', '-')}</div>
                <div class="label">IForest F1</div>
            </div>
            <div class="stat-card">
                <div class="value" style="color: var(--green);">{ensemble_m.get('f1_score', '-')}</div>
                <div class="label">Ensemble F1</div>
            </div>
        </div>

        <div class="chart-grid">
            <div class="chart-card full-width">
                <h3>Sensor Time Series</h3>
                <div class="sensor-tabs" id="sensorTabs"></div>
                <div id="timeseriesChart" style="height: 350px;"></div>
            </div>

            <div class="chart-card">
                <h3>Reconstruction Error Comparison</h3>
                <div id="reconChart" style="height: 280px;"></div>
            </div>

            <div class="chart-card">
                <h3>Anomaly Type Distribution</h3>
                <div id="pieChart" style="height: 280px;"></div>
            </div>

            <div class="chart-card full-width">
                <h3>Model Comparison (4 models)</h3>
                <table class="metrics-table">
                    <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Type</th></tr>
                    <tr>
                        <td><strong>LSTM Autoencoder</strong></td>
                        <td>{lstm_m.get('accuracy', '-')}</td><td>{lstm_m.get('precision', '-')}</td>
                        <td>{lstm_m.get('recall', '-')}</td><td><strong>{lstm_m.get('f1_score', '-')}</strong></td>
                        <td><span class="badge badge-blue">Deep Learning</span></td>
                    </tr>
                    <tr>
                        <td><strong>Transformer Autoencoder</strong></td>
                        <td>{trans_m.get('accuracy', '-')}</td><td>{trans_m.get('precision', '-')}</td>
                        <td>{trans_m.get('recall', '-')}</td><td><strong>{trans_m.get('f1_score', '-')}</strong></td>
                        <td><span class="badge badge-purple">Attention-Based</span></td>
                    </tr>
                    <tr>
                        <td><strong>Isolation Forest</strong></td>
                        <td>{iforest_m.get('accuracy', '-')}</td><td>{iforest_m.get('precision', '-')}</td>
                        <td>{iforest_m.get('recall', '-')}</td><td><strong>{iforest_m.get('f1_score', '-')}</strong></td>
                        <td><span class="badge badge-orange">Traditional ML</span></td>
                    </tr>
                    <tr>
                        <td><strong>Ensemble (soft-vote)</strong></td>
                        <td>{ensemble_m.get('accuracy', '-')}</td><td>{ensemble_m.get('precision', '-')}</td>
                        <td>{ensemble_m.get('recall', '-')}</td><td><strong>{ensemble_m.get('f1_score', '-')}</strong></td>
                        <td><span class="badge badge-green">Ensemble</span></td>
                    </tr>
                </table>
            </div>
        </div>
    </div>

    <script>
        const plotlyLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8', size: 11 }},
            margin: {{ t: 20, r: 30, b: 40, l: 60 }},
            xaxis: {{ gridcolor: '#1e293b' }}, yaxis: {{ gridcolor: '#1e293b' }},
        }};
        const plotlyConfig = {{ responsive: true, displayModeBar: true }};

        const sensorLabels = {json.dumps(sensor_labels)};
        const sensorTraces = [];
        {sensor_traces}

        const tabContainer = document.getElementById('sensorTabs');
        sensorLabels.forEach((label, idx) => {{
            const tab = document.createElement('div');
            tab.className = 'sensor-tab' + (idx === 0 ? ' active' : '');
            tab.textContent = label;
            tab.onclick = () => {{
                document.querySelectorAll('.sensor-tab').forEach((t, i) => t.classList.toggle('active', i === idx));
                Plotly.restyle('timeseriesChart', {{ visible: sensorTraces.map(t => t.sensorGroup === idx) }});
            }};
            tabContainer.appendChild(tab);
        }});

        Plotly.newPlot('timeseriesChart', sensorTraces,
            {{...plotlyLayout, showlegend: true, legend: {{x: 0, y: 1.15, orientation: 'h'}}}}, plotlyConfig);

        Plotly.newPlot('reconChart', [
            {{ x: {json.dumps(timestamps)}, y: {json.dumps(recon_lstm)},
               type: 'scatter', mode: 'lines', fill: 'tozeroy',
               fillcolor: 'rgba(59,130,246,0.1)', line: {{color: '#3b82f6', width: 1}}, name: 'LSTM' }},
            {{ x: {json.dumps(timestamps)}, y: {json.dumps(recon_trans)},
               type: 'scatter', mode: 'lines', fill: 'tozeroy',
               fillcolor: 'rgba(139,92,246,0.1)', line: {{color: '#8b5cf6', width: 1}}, name: 'Transformer' }},
            {{ x: ['{timestamps[0]}', '{timestamps[-1]}'], y: [{lstm_threshold}, {lstm_threshold}],
               type: 'scatter', mode: 'lines', line: {{color: '#ef4444', width: 2, dash: 'dash'}}, name: 'LSTM Threshold' }},
            {{ x: ['{timestamps[0]}', '{timestamps[-1]}'], y: [{trans_threshold}, {trans_threshold}],
               type: 'scatter', mode: 'lines', line: {{color: '#f59e0b', width: 2, dash: 'dot'}}, name: 'Trans Threshold' }}
        ], plotlyLayout, plotlyConfig);

        Plotly.newPlot('pieChart', [{{
            labels: {json.dumps(anom_type_labels)}, values: {json.dumps(anom_type_values)},
            type: 'pie', marker: {{ colors: ['#ef4444', '#f59e0b', '#8b5cf6', '#10b981', '#3b82f6'] }},
            textinfo: 'label+percent', textfont: {{ color: '#e2e8f0' }}, hole: 0.4,
        }}], {{ ...plotlyLayout, showlegend: false }}, plotlyConfig);
    </script>
</body>
</html>"""

    return html


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Building interactive dashboard...')

    lstm_df = pd.read_csv(os.path.join(DATA_DIR, 'test_with_lstm_preds.csv'))
    transformer_df = pd.read_csv(os.path.join(DATA_DIR, 'test_with_transformer_preds.csv'))
    iforest_df = pd.read_csv(os.path.join(DATA_DIR, 'test_with_iforest_preds.csv'))

    test_df = lstm_df.copy()
    test_df['transformer_pred'] = transformer_df['transformer_pred']
    test_df['transformer_error'] = transformer_df['transformer_error']
    test_df['iforest_pred'] = iforest_df['iforest_pred']
    test_df['iforest_score'] = iforest_df['iforest_score']

    metrics_path = os.path.join(OUTPUT_DIR, 'comparison_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {'overall': [{}, {}, {}, {}]}

    html = generate_dashboard_html(test_df, metrics)

    output_path = os.path.join(OUTPUT_DIR, 'dashboard.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'  Dashboard saved to {output_path} ({size_mb:.1f} MB)')
    print('  Open in a browser to explore the interactive charts.')
    print('Done.')


if __name__ == '__main__':
    main()
