"""
Shared Utilities & Model Definitions
======================================

Central place for model architectures, normalization helpers, and common
functions used across training scripts and the Streamlit app. Keeps things
DRY — the old version had the LSTM model copy-pasted in two files which
was asking for trouble when tuning hyperparameters.

Usage:
    from utils import LSTMAutoencoder, TransformerAutoencoder, normalize_data
"""

import os
import json
import math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ── Configuration ──────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / 'configs' / 'config.yaml'


def load_config(path: Optional[Path] = None) -> dict:
    """
    Load the centralized YAML configuration file.

    Looks for configs/config.yaml relative to the project root by default.
    Returns an empty dict if the file isn't found or PyYAML isn't installed
    so that importing utils never crashes in minimal environments.

    Args:
        path: Override the default config path (useful for testing).

    Returns:
        Parsed config as a nested dict.

    Example:
        cfg = load_config()
        lr = cfg.get('lstm', {}).get('learning_rate', 1e-3)
    """
    if not _YAML_AVAILABLE:
        return {}
    config_file = path or _CONFIG_PATH
    if not config_file.exists():
        return {}
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


# Load once at import time — avoids repeated disk reads inside training loops.
# Scripts can do `from utils import CFG` and access values without re-parsing.
CFG: dict = load_config()


# ── Constants ──────────────────────────────────────────────────────────
# These pull from config when available so a single edit to config.yaml
# propagates everywhere automatically.

SEED: int = CFG.get('data', {}).get('seed', 42)
SEQUENCE_LENGTH: int = CFG.get('model', {}).get('sequence_length', 30)  # ~2.5 hrs at 5-min intervals
SENSOR_COLS = ['temperature_C', 'pressure_kPa', 'vibration_mm_s', 'current_draw_A']
NUM_SENSORS: int = len(SENSOR_COLS)


# ── Reproducibility ───────────────────────────────────────────────────

def set_seeds(seed: int = SEED):
    """Lock down RNGs. Call this before any training run."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Normalization ──────────────────────────────────────────────────────

def normalize_data(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Z-score normalization fit on training data only — no data leakage.
    Returns normalized arrays and stats dict for inference.
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    # avoid div by zero for constant columns — shouldn't happen but better safe
    std[std == 0] = 1.0
    stats = {'mean': mean.tolist(), 'std': std.tolist()}
    return (train - mean) / std, (test - mean) / std, stats


def apply_normalization(data: np.ndarray, stats: dict) -> np.ndarray:
    """Apply pre-computed normalization stats (for inference)."""
    mean = np.array(stats['mean'])
    std = np.array(stats['std'])
    return (data - mean) / std


# ══════════════════════════════════════════════════════════════════════
#  LSTM AUTOENCODER
# ══════════════════════════════════════════════════════════════════════

class LSTMEncoder(nn.Module):
    """Encodes a sequence into a fixed-size latent vector."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        # grab the last layer's hidden state
        latent = self.fc_latent(hidden[-1])
        return latent


class LSTMDecoder(nn.Module):
    """Reconstructs sequence from latent representation."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 seq_length: int, num_layers: int, dropout: float):
        super().__init__()
        self.seq_length = seq_length
        self.fc_expand = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        expanded = self.fc_expand(z).unsqueeze(1).repeat(1, self.seq_length, 1)
        decoded, _ = self.lstm(expanded)
        output = self.fc_out(decoded)
        return output


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM autoencoder — reconstruction error is the anomaly score.
    High error = the model hasn't seen this pattern during training.
    """

    def __init__(self, input_dim: int = NUM_SENSORS, hidden_dim: int = 64,
                 latent_dim: int = 32, seq_length: int = SEQUENCE_LENGTH,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim,
                                    num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim,
                                    seq_length, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed


# ══════════════════════════════════════════════════════════════════════
#  TRANSFORMER AUTOENCODER
# ══════════════════════════════════════════════════════════════════════
# Loosely follows the architecture from:
#   Zerveas et al., "A Transformer-based Framework for Multivariate
#   Time Series Representation Learning" (KDD 2021)
#
# The key insight: self-attention lets the model weigh which time steps
# matter for reconstructing each position — great for catching long-range
# dependencies that LSTMs sometimes fumble on.

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from Vaswani et al. 2017.
    
    We tried learnable positional embeddings too but sinusoidal works
    just as well for sequences this short and generalizes better.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # the classic log-space trick for computing the division terms
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    """
    Transformer-based autoencoder for time-series anomaly detection.
    
    Architecture:
      - Input projection: sensor_dim -> d_model
      - Positional encoding (sinusoidal)
      - Encoder: N transformer encoder layers with multi-head self-attention
      - Bottleneck: mean-pool over time -> linear -> latent -> expand
      - Decoder: N transformer decoder layers
      - Output projection: d_model -> sensor_dim
    
    The bottleneck forces information compression — same principle as the
    LSTM version but attention handles long-range patterns way better.
    
    TODO: experiment with masking random time steps during training (MAE-style)
          — could improve robustness to missing sensor readings
    """

    def __init__(self, input_dim: int = NUM_SENSORS, d_model: int = 64,
                 n_heads: int = 4, num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2, dim_feedforward: int = 128,
                 latent_dim: int = 32, seq_length: int = SEQUENCE_LENGTH,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model

        # project raw sensor features up to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length + 10,
                                               dropout=dropout)

        # encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',  # gelu slightly outperforms relu here empirically
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # bottleneck — compress temporal dimension via mean pooling then linear
        self.fc_to_latent = nn.Linear(d_model, latent_dim)
        self.fc_from_latent = nn.Linear(latent_dim, d_model * seq_length)

        # decoder stack — using encoder layers since we don't need cross-attention
        # for autoencoding (not doing seq2seq translation)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # project back to sensor space
        self.output_projection = nn.Linear(d_model, input_dim)

        # layer norm before output — stabilizes reconstruction quality
        self.output_norm = nn.LayerNorm(d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent vector."""
        projected = self.input_projection(x)
        projected = self.pos_encoder(projected)
        encoded = self.transformer_encoder(projected)

        # mean pool over time → latent
        # TODO: try attention-weighted pooling instead of mean
        pooled = encoded.mean(dim=1)
        latent = self.fc_to_latent(pooled)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to sequence."""
        expanded = self.fc_from_latent(z)
        expanded = expanded.view(-1, self.seq_length, self.d_model)

        decoded = self.transformer_decoder(expanded)
        decoded = self.output_norm(decoded)
        output = self.output_projection(decoded)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed


# ── Factory function ───────────────────────────────────────────────────

def build_model(model_type: str, device: torch.device,
                input_dim: int = NUM_SENSORS,
                seq_length: int = SEQUENCE_LENGTH,
                cfg: Optional[dict] = None) -> nn.Module:
    """
    Convenience function for instantiating models by name.
    Hyperparameters are pulled from config.yaml when available, with
    safe fallbacks to sensible defaults so existing scripts keep working.

    Args:
        model_type: 'lstm' or 'transformer'.
        device:     Target device (CPU or CUDA).
        input_dim:  Number of sensor channels (default: NUM_SENSORS).
        seq_length: Sliding window length (default: SEQUENCE_LENGTH).
        cfg:        Config dict override — uses module-level CFG if None.
    """
    _cfg = cfg if cfg is not None else CFG

    if model_type == 'lstm':
        lc = _cfg.get('lstm', {})
        mc = _cfg.get('model', {})
        model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=lc.get('hidden_dim', 64),
            latent_dim=mc.get('latent_dim', 32),
            seq_length=seq_length,
            num_layers=lc.get('num_layers', 2),
            dropout=lc.get('dropout', 0.2),
        )
    elif model_type == 'transformer':
        tc = _cfg.get('transformer', {})
        mc = _cfg.get('model', {})
        model = TransformerAutoencoder(
            input_dim=input_dim,
            d_model=tc.get('d_model', 64),
            n_heads=tc.get('n_heads', 4),
            num_encoder_layers=tc.get('num_encoder_layers', 2),
            num_decoder_layers=tc.get('num_decoder_layers', 2),
            dim_feedforward=tc.get('dim_feedforward', 128),
            latent_dim=mc.get('latent_dim', 32),
            seq_length=seq_length,
            dropout=tc.get('dropout', 0.1),
        )
    else:
        raise ValueError(f'Unknown model type: {model_type!r}. Use "lstm" or "transformer".')

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  [{model_type.upper()}] Parameters: {total_params:,}')
    return model
