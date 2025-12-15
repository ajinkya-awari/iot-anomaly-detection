"""
LSTM Autoencoder for Time-Series Anomaly Detection
====================================================

Trains a seq2seq LSTM autoencoder where anomalies are flagged based on
reconstruction error exceeding a learned threshold. The threshold is set
at the 95th percentile of training reconstruction errors — a standard
approach from the literature (Malhotra et al., 2016).

The model learns to reconstruct normal operational patterns; anything it
can't reconstruct well is flagged as anomalous.

Outputs:
    models/lstm_autoencoder.pt       — trained model weights
    models/lstm_threshold.npy        — anomaly threshold value
    models/norm_stats.json           — normalization statistics
    outputs/lstm_training_loss.png   — training loss curve
    outputs/lstm_recon_errors.png    — reconstruction error distribution

Usage:
    python src/lstm_autoencoder.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    LSTMAutoencoder, build_model, set_seeds, normalize_data,
    SENSOR_COLS, SEQUENCE_LENGTH, SEED
)


# ── Hyperparameters ────────────────────────────────────────────────────

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50                 # usually converges around epoch 30-35
THRESHOLD_PERCENTILE = 95       # following Malhotra et al. 2016

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = 'models'
OUTPUT_DIR = 'outputs'
DATA_DIR = 'data'


# ── Dataset ────────────────────────────────────────────────────────────

class SensorSequenceDataset(Dataset):
    """
    Sliding window dataset for time-series sequences.
    Nothing fancy — just chop the series into overlapping windows.
    """

    def __init__(self, data: np.ndarray, seq_length: int):
        self.data = data.astype(np.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx: int):
        seq = self.data[idx : idx + self.seq_length]
        return torch.tensor(seq)


# ── Training ───────────────────────────────────────────────────────────

def train_model(model: nn.Module, train_loader: DataLoader,
                num_epochs: int, lr: float) -> list:
    """Standard training loop with LR scheduling and gradient clipping."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # TODO: try cosine annealing — plateau scheduler sometimes gets stuck
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    model.train()
    losses = []
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()

            # clip gradients — LSTMs can be finicky with exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.detach().item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  Epoch [{epoch+1:3d}/{num_epochs}]  Loss: {avg_loss:.6f}  '
                  f'LR: {current_lr:.2e}  Best: {best_loss:.6f}')

    return losses


def compute_reconstruction_errors(model: nn.Module, data: np.ndarray,
                                   seq_length: int) -> np.ndarray:
    """
    Compute per-sample reconstruction error.
    Returns array of length (n_samples - seq_length).
    """
    model.eval()
    dataset = SensorSequenceDataset(data, seq_length)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            # MSE per sequence, averaged over features and time
            mse = torch.mean((batch - recon) ** 2, dim=(1, 2))
            errors.extend(mse.cpu().numpy())

    return np.array(errors)


# ── Plotting ───────────────────────────────────────────────────────────

def plot_training_loss(losses: list, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(losses) + 1), losses, color='#2980b9', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('LSTM Autoencoder Training Loss')
    ax.grid(True, alpha=0.3)
    min_idx = np.argmin(losses)
    ax.axvline(x=min_idx + 1, color='#e74c3c', linestyle='--', alpha=0.5,
               label=f'Min loss @ epoch {min_idx + 1}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {save_path}')


def plot_error_distribution(train_errors: np.ndarray, test_errors: np.ndarray,
                            threshold: float, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(train_errors, bins=80, alpha=0.7, color='#27ae60', edgecolor='none')
    axes[0].axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    axes[0].set_title('Training Reconstruction Errors')
    axes[0].set_xlabel('MSE')
    axes[0].legend()

    axes[1].hist(test_errors, bins=80, alpha=0.7, color='#2980b9', edgecolor='none')
    axes[1].axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    axes[1].set_title('Test Reconstruction Errors')
    axes[1].set_xlabel('MSE')
    axes[1].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {save_path}')


# ── Main ───────────────────────────────────────────────────────────────

def main():
    set_seeds()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading sensor data...')
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv'))

    train_values = train_df[SENSOR_COLS].values
    test_values = test_df[SENSOR_COLS].values

    print('Normalizing...')
    train_norm, test_norm, norm_stats = normalize_data(train_values, test_values)

    # save normalization stats — needed for inference in Streamlit and eval
    with open(os.path.join(MODEL_DIR, 'norm_stats.json'), 'w') as f:
        json.dump(norm_stats, f)

    print(f'Building LSTM Autoencoder (device: {DEVICE})...')
    model = build_model('lstm', DEVICE)

    # training
    train_dataset = SensorSequenceDataset(train_norm, SEQUENCE_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               drop_last=True)

    print(f'\nTraining for {NUM_EPOCHS} epochs...')
    losses = train_model(model, train_loader, NUM_EPOCHS, LEARNING_RATE)

    # save model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'lstm_autoencoder.pt'))
    print(f'  Model saved to {MODEL_DIR}/lstm_autoencoder.pt')

    # compute reconstruction errors
    print('\nComputing reconstruction errors...')
    train_errors = compute_reconstruction_errors(model, train_norm, SEQUENCE_LENGTH)
    test_errors = compute_reconstruction_errors(model, test_norm, SEQUENCE_LENGTH)

    threshold = float(np.percentile(train_errors, THRESHOLD_PERCENTILE))
    np.save(os.path.join(MODEL_DIR, 'lstm_threshold.npy'), threshold)
    print(f'  Threshold (p{THRESHOLD_PERCENTILE}): {threshold:.6f}')
    print(f'  Train anomalies flagged: {(train_errors > threshold).sum()} / {len(train_errors)}')
    print(f'  Test anomalies flagged: {(test_errors > threshold).sum()} / {len(test_errors)}')

    # save test predictions for later comparison
    test_preds = np.zeros(len(test_df))
    test_preds[SEQUENCE_LENGTH:] = (test_errors > threshold).astype(int)
    test_df['lstm_pred'] = test_preds
    test_df['lstm_error'] = np.concatenate([np.zeros(SEQUENCE_LENGTH), test_errors])
    test_df.to_csv(os.path.join(DATA_DIR, 'test_with_lstm_preds.csv'), index=False)

    # plots
    plot_training_loss(losses, os.path.join(OUTPUT_DIR, 'lstm_training_loss.png'))
    plot_error_distribution(train_errors, test_errors, threshold,
                            os.path.join(OUTPUT_DIR, 'lstm_recon_errors.png'))

    print('\nLSTM Autoencoder training complete.')


if __name__ == '__main__':
    main()
