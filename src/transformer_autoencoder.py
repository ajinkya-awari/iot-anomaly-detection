"""
Transformer Autoencoder for Time-Series Anomaly Detection
==========================================================

Implements a Transformer-based autoencoder that uses self-attention to
capture long-range temporal dependencies in sensor streams. The key advantage
over LSTM: attention can directly relate distant time steps without
propagating through intermediate hidden states, which helps with detecting
anomalies that span longer windows (e.g., gradual drifts).

Based on ideas from:
  - Zerveas et al., "A Transformer-based Framework for Multivariate
    Time Series Representation Learning" (KDD 2021)
  - Tuli et al., "TranAD: Deep Transformer Networks for Anomaly
    Detection in Multivariate Time Series Data" (VLDB 2022)

Our architecture is simpler than TranAD — we use a standard bottleneck
autoencoder with transformer blocks instead of adversarial training.
Keeps things interpretable and easier to debug.

Outputs:
    models/transformer_autoencoder.pt       — trained model weights
    models/transformer_threshold.npy        — anomaly threshold
    outputs/transformer_training_loss.png   — loss curve
    outputs/transformer_recon_errors.png    — error distribution
    outputs/attention_heatmap.png           — attention weight visualization

Usage:
    python src/transformer_autoencoder.py
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
import matplotlib.gridspec as gridspec

# local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    TransformerAutoencoder, build_model, set_seeds, normalize_data,
    SENSOR_COLS, SEQUENCE_LENGTH, SEED
)


# ── Hyperparameters ────────────────────────────────────────────────────

BATCH_SIZE = 64
LEARNING_RATE = 5e-4            # transformers like slightly lower LR than LSTMs
NUM_EPOCHS = 60                 # needs a bit more epochs to converge than LSTM
THRESHOLD_PERCENTILE = 95
WARMUP_EPOCHS = 5               # linear warmup helps transformer training a lot

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = 'models'
OUTPUT_DIR = 'outputs'
DATA_DIR = 'data'


# ── Dataset ────────────────────────────────────────────────────────────

class SensorSequenceDataset(Dataset):
    """Sliding window dataset — same as LSTM version."""

    def __init__(self, data: np.ndarray, seq_length: int):
        self.data = data.astype(np.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx: int):
        seq = self.data[idx : idx + self.seq_length]
        return torch.tensor(seq)


# ── Training with Warmup ──────────────────────────────────────────────

class WarmupScheduler:
    """
    Linear warmup then hand off to a cosine decay.
    
    Transformers are notoriously sensitive to the initial learning rate —
    without warmup they often diverge in the first few epochs.
    See Popel & Bojar 2018 for a good discussion on this.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # cosine decay after warmup
            progress = (self.current_epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


def train_model(model: nn.Module, train_loader: DataLoader,
                num_epochs: int, lr: float) -> list:
    """Training loop with warmup + cosine decay scheduling."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = WarmupScheduler(optimizer, WARMUP_EPOCHS, num_epochs, lr)

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

            # clip gradients — less critical than LSTM but still good practice
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.detach().item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss

        current_lr = scheduler.get_lr()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  Epoch [{epoch+1:3d}/{num_epochs}]  Loss: {avg_loss:.6f}  '
                  f'LR: {current_lr:.2e}  Best: {best_loss:.6f}')

    return losses


def compute_reconstruction_errors(model: nn.Module, data: np.ndarray,
                                   seq_length: int) -> np.ndarray:
    """Compute per-sample reconstruction error."""
    model.eval()
    dataset = SensorSequenceDataset(data, seq_length)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            mse = torch.mean((batch - recon) ** 2, dim=(1, 2))
            errors.extend(mse.cpu().numpy())

    return np.array(errors)


def extract_attention_weights(model: TransformerAutoencoder,
                               sample: np.ndarray) -> np.ndarray:
    """
    Extract attention weights from the first encoder layer for visualization.
    
    This gives us interpretability — we can see which time steps the model
    attends to when encoding each position. Useful for debugging and also
    looks great in a paper/README.
    
    TODO: average attention across all heads vs show per-head — both are
          informative but per-head shows specialization patterns
    """
    model.eval()
    tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # hook into the first encoder layer's self-attention
    attention_weights = []

    def hook_fn(module, input, output):
        # TransformerEncoderLayer uses MultiheadAttention internally
        # we need to call it separately to get attention weights
        pass

    # alternative approach: manually run the first layer's attention
    with torch.no_grad():
        projected = model.input_projection(tensor)
        projected = model.pos_encoder(projected)

        # get attention from first layer
        first_layer = model.transformer_encoder.layers[0]

        # multihead attention forward with need_weights=True
        attn_output, attn_weights = first_layer.self_attn(
            projected, projected, projected, need_weights=True
        )

    # attn_weights shape: (batch, seq_len, seq_len) — averaged over heads
    return attn_weights.squeeze(0).cpu().numpy()


# ── Plotting ───────────────────────────────────────────────────────────

def plot_training_loss(losses: list, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(losses) + 1), losses, color='#8e44ad', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Transformer Autoencoder Training Loss')
    ax.grid(True, alpha=0.3)

    # mark warmup phase
    ax.axvspan(0, WARMUP_EPOCHS, alpha=0.1, color='orange', label='Warmup phase')
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

    axes[0].hist(train_errors, bins=80, alpha=0.7, color='#8e44ad', edgecolor='none')
    axes[0].axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    axes[0].set_title('Training Reconstruction Errors (Transformer)')
    axes[0].set_xlabel('MSE')
    axes[0].legend()

    axes[1].hist(test_errors, bins=80, alpha=0.7, color='#2980b9', edgecolor='none')
    axes[1].axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    axes[1].set_title('Test Reconstruction Errors (Transformer)')
    axes[1].set_xlabel('MSE')
    axes[1].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {save_path}')


def plot_attention_heatmap(attn_weights: np.ndarray, save_path: str):
    """
    Visualize what the transformer attends to.
    
    In a well-trained model you should see:
      - Strong diagonal (each step attends to itself)
      - Bands at periodic offsets (diurnal pattern awareness)
      - Diffuse attention at anomaly locations
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attn_weights, cmap='viridis', aspect='auto')
    ax.set_xlabel('Key Position (time step)')
    ax.set_ylabel('Query Position (time step)')
    ax.set_title('Transformer Self-Attention Weights (Encoder Layer 1)')
    plt.colorbar(im, ax=ax, fraction=0.046)
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
    # reuse the same norm stats saved by the LSTM script
    stats_path = os.path.join(MODEL_DIR, 'norm_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            norm_stats = json.load(f)
        mean = np.array(norm_stats['mean'])
        std = np.array(norm_stats['std'])
        train_norm = (train_values - mean) / std
        test_norm = (test_values - mean) / std
        print('  Using pre-computed normalization stats from LSTM step')
    else:
        train_norm, test_norm, norm_stats = normalize_data(train_values, test_values)
        with open(stats_path, 'w') as f:
            json.dump(norm_stats, f)

    print(f'Building Transformer Autoencoder (device: {DEVICE})...')
    model = build_model('transformer', DEVICE)

    # training
    train_dataset = SensorSequenceDataset(train_norm, SEQUENCE_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               drop_last=True)

    print(f'\nTraining for {NUM_EPOCHS} epochs (warmup: {WARMUP_EPOCHS})...')
    losses = train_model(model, train_loader, NUM_EPOCHS, LEARNING_RATE)

    # save model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'transformer_autoencoder.pt'))
    print(f'  Model saved to {MODEL_DIR}/transformer_autoencoder.pt')

    # compute reconstruction errors
    print('\nComputing reconstruction errors...')
    train_errors = compute_reconstruction_errors(model, train_norm, SEQUENCE_LENGTH)
    test_errors = compute_reconstruction_errors(model, test_norm, SEQUENCE_LENGTH)

    threshold = float(np.percentile(train_errors, THRESHOLD_PERCENTILE))
    np.save(os.path.join(MODEL_DIR, 'transformer_threshold.npy'), threshold)
    print(f'  Threshold (p{THRESHOLD_PERCENTILE}): {threshold:.6f}')
    print(f'  Train anomalies flagged: {(train_errors > threshold).sum()} / {len(train_errors)}')
    print(f'  Test anomalies flagged: {(test_errors > threshold).sum()} / {len(test_errors)}')

    # save test predictions
    test_preds = np.zeros(len(test_df))
    test_preds[SEQUENCE_LENGTH:] = (test_errors > threshold).astype(int)
    test_df['transformer_pred'] = test_preds
    test_df['transformer_error'] = np.concatenate([np.zeros(SEQUENCE_LENGTH), test_errors])
    test_df.to_csv(os.path.join(DATA_DIR, 'test_with_transformer_preds.csv'), index=False)

    # extract and plot attention weights on a sample
    print('\nExtracting attention weights...')
    sample_window = train_norm[100:100 + SEQUENCE_LENGTH]
    try:
        attn_weights = extract_attention_weights(model, sample_window)
        plot_attention_heatmap(attn_weights, os.path.join(OUTPUT_DIR, 'attention_heatmap.png'))
    except Exception as e:
        # attention extraction can sometimes fail depending on PyTorch version
        print(f'  Warning: Could not extract attention weights: {e}')
        print(f'  Skipping attention heatmap (non-critical)')

    # plots
    plot_training_loss(losses, os.path.join(OUTPUT_DIR, 'transformer_training_loss.png'))
    plot_error_distribution(train_errors, test_errors, threshold,
                            os.path.join(OUTPUT_DIR, 'transformer_recon_errors.png'))

    print('\nTransformer Autoencoder training complete.')


if __name__ == '__main__':
    main()
