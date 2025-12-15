"""
Master Pipeline Runner
=======================

Runs the complete IoT anomaly detection pipeline end-to-end:
  1. Generate synthetic sensor data
  2. Train LSTM Autoencoder
  3. Train Transformer Autoencoder
  4. Train Isolation Forest baseline
  5. Evaluate all models + soft-vote ensemble
  6. Build interactive HTML dashboard

Just run this one script and go grab a coffee.

Usage:
    python run_all.py
"""

import os
import sys
import time
import subprocess


def run_step(step_num: int, description: str, script: str):
    """Run a pipeline step and handle errors."""
    print(f'\n{"="*60}')
    print(f'  Step {step_num}: {description}')
    print(f'{"="*60}\n')

    start = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f'\n  ✗ Step {step_num} FAILED after {elapsed:.1f}s')
        print(f'  Check the error above and fix before re-running.')
        sys.exit(1)

    print(f'\n  ✓ Step {step_num} completed in {elapsed:.1f}s')


def main():
    print('='*60)
    print('  IoT Anomaly Detection — LSTM · Transformer · IForest · Ensemble')
    print('='*60)

    total_start = time.time()

    steps = [
        (1, 'Generate synthetic IoT sensor data', 'src/generate_data.py'),
        (2, 'Train LSTM Autoencoder', 'src/lstm_autoencoder.py'),
        (3, 'Train Transformer Autoencoder', 'src/transformer_autoencoder.py'),
        (4, 'Train Isolation Forest baseline', 'src/isolation_forest.py'),
        (5, 'Evaluate all models + soft-vote ensemble', 'src/evaluate.py'),
        (6, 'Build interactive dashboard', 'src/build_dashboard.py'),
    ]

    for step_num, desc, script in steps:
        run_step(step_num, desc, script)

    total_elapsed = time.time() - total_start

    print(f'\n{"="*60}')
    print(f'  All steps completed in {total_elapsed:.1f}s')
    print(f'{"="*60}')
    print(f'\n  Outputs:')
    print(f'    data/               — sensor datasets')
    print(f'    models/             — trained model weights')
    print(f'    outputs/            — plots, metrics, dashboard')
    print(f'\n  Next steps:')
    print(f'    1. Open outputs/dashboard.html in a browser')
    print(f'    2. Run "streamlit run src/streamlit_app.py" for real-time demo')


if __name__ == '__main__':
    main()
