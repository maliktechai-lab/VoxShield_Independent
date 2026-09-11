"""
VoxShieldNet — Custom Dual-Branch CNN+GRU Architecture
=======================================================

Architecture:
  RAW WAVEFORM  → 1D CNN stack → temporal features
  LOG-MEL SPEC  → 2D CNN stack → spectral features
  FUSION        → Bidirectional GRU → Temporal Attention → Binary Classifier

Constraints (non-negotiable):
  - Trained from random initialization only.
  - No pretrained speech encoder, speaker encoder, or LLM components.
  - No Wav2Vec, HuBERT, Whisper, WavLM, AASIST.
  - No external AI inference APIs.
  - Model returns RAW LOGITS (no sigmoid inside forward()).
  - Training: BCEWithLogitsLoss.
  - Inference: sigmoid(logit).
"""

from __future__ import annotations

import math
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Default configuration (stored in checkpoint for reproducibility)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CONFIG: Dict[str, Any] = {
    "name": "VoxShieldNet",
    "version": "1.0.0",
    # Audio
    "sample_rate": 16000,
    "max_length_sec": 4.0,           # clip / pad to this duration
    # Log-Mel spectrogram
    "n_mels": 80,
    "n_fft": 512,
    "hop_length": 160,               # 10 ms at 16 kHz
    "win_length": 400,               # 25 ms at 16 kHz
    # Raw-waveform branch
    "raw_channels": [64, 128, 128, 256],
    "raw_kernel_sizes": [15, 9, 7, 5],
    "raw_strides": [4, 2, 2, 2],
    # Spectral branch
    "spec_channels": [32, 64, 128, 256],
    "spec_kernel": 3,
    # GRU
    "gru_hidden": 256,
    "gru_layers": 2,
    "gru_dropout": 0.3,
    # Classifier
    "classifier_hidden": 256,
    "dropout": 0.4,
    # Misc
    "pretrained": False,
}


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class _Conv1dBlock(nn.Module):
    """1D conv → BN → GELU → optional residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1,
                 padding: int | None = None):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        # Residual projection only when spatial dims and channels change.
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False)
            if (in_ch != out_ch or stride != 1)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.conv(x)))
        if self.residual is not None:
            x = self.residual(x)
            # Align length in case of rounding differences
            if x.shape[-1] != out.shape[-1]:
                x = x[..., :out.shape[-1]]
        return out + x


class _Conv2dBlock(nn.Module):
    """2D conv → BN → GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.residual = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if (in_ch != out_ch or stride != 1)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.conv(x)))
        if self.residual is not None:
            x = self.residual(x)
            if x.shape != out.shape:
                x = x[..., :out.shape[-2], :out.shape[-1]]
        return out + x


class TemporalAttention(nn.Module):
    """Learnable temporal attention: weights each time step."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        scores = self.attn(x)          # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        out = (x * weights).sum(dim=1)  # (B, H)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class VoxShieldNet(nn.Module):
    """
    Dual-branch anti-spoofing network.

    Raw waveform branch: 1D CNN captures fine-grained temporal artifacts.
    Log-Mel spectrogram branch: 2D CNN captures spectral patterns.
    Fusion: concatenated features → BiGRU → temporal attention → classifier.

    Returns RAW LOGITS (no sigmoid).
    Apply sigmoid() at inference time for probability.
    Use BCEWithLogitsLoss during training.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__()
        cfg = {**MODEL_CONFIG, **(config or {})}
        self.config = cfg

        sr = cfg["sample_rate"]
        max_len = cfg["max_length_sec"]
        self._max_samples = int(sr * max_len)

        # ── Raw-waveform branch ───────────────────────────────────────────────
        raw_chs = cfg["raw_channels"]
        raw_ks = cfg["raw_kernel_sizes"]
        raw_st = cfg["raw_strides"]
        raw_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, ks, st in zip(raw_chs, raw_ks, raw_st):
            raw_layers.append(_Conv1dBlock(in_ch, out_ch, ks, stride=st))
            in_ch = out_ch
        self.raw_branch = nn.Sequential(*raw_layers)
        self._raw_out_ch = raw_chs[-1]

        # ── Log-Mel spectrogram branch ────────────────────────────────────────
        spec_chs = cfg["spec_channels"]
        spec_k = cfg["spec_kernel"]
        spec_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in spec_chs:
            spec_layers.append(_Conv2dBlock(in_ch, out_ch, spec_k, stride=2))
            in_ch = out_ch
        self.spec_branch = nn.Sequential(*spec_layers)
        self._spec_out_ch = spec_chs[-1]

        # Compute spec output height (frequency axis) after 4 stride-2 layers
        n_mels = cfg["n_mels"]
        spec_freq_out = n_mels
        for _ in spec_chs:
            spec_freq_out = math.ceil(spec_freq_out / 2)
        # Projection to align channel dimension
        self.spec_proj = nn.Conv1d(
            self._spec_out_ch * spec_freq_out, self._raw_out_ch, 1, bias=False
        )

        # ── Fusion GRU ────────────────────────────────────────────────────────
        gru_in = self._raw_out_ch * 2    # raw + spec
        gru_h = cfg["gru_hidden"]
        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=gru_h,
            num_layers=cfg["gru_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=cfg["gru_dropout"] if cfg["gru_layers"] > 1 else 0.0,
        )

        gru_out_dim = gru_h * 2  # bidirectional

        # ── Temporal Attention ────────────────────────────────────────────────
        self.attention = TemporalAttention(gru_out_dim)

        # ── Classifier ────────────────────────────────────────────────────────
        cls_h = cfg["classifier_hidden"]
        dropout = cfg["dropout"]
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Linear(gru_out_dim, cls_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_h, cls_h // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(cls_h // 2, 1),
            # NO sigmoid — returns raw logit
        )

        self._init_weights()

    # ── Weight initialization ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    # ── Preprocessing helpers ─────────────────────────────────────────────────

    def _prepare_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        """Pad or clip raw waveform to fixed length, return (B,1,T)."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(1)  # (B,C,T) → keep first channel
        # Take first channel if stereo
        wav = wav[:, :1, :]
        T = wav.shape[-1]
        if T < self._max_samples:
            pad = self._max_samples - T
            wav = F.pad(wav, (0, pad))
        else:
            wav = wav[..., : self._max_samples]
        return wav

    def _mel_spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram from (B,1,T) → (B,1,n_mels,T')."""
        cfg = self.config
        hop = cfg["hop_length"]
        n_fft = cfg["n_fft"]
        win = cfg["win_length"]
        n_mels = cfg["n_mels"]
        sr = cfg["sample_rate"]

        B = wav.shape[0]
        wav_flat = wav[:, 0, :]  # (B, T)

        # STFT → power spec
        window = torch.hann_window(win, device=wav.device, dtype=wav.dtype)
        # Process each sample; torch.stft operates on 1D/2D
        specs = []
        for i in range(B):
            stft = torch.stft(
                wav_flat[i],
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                window=window,
                return_complex=True,
            )
            power = stft.abs().pow(2)  # (F, T')
            specs.append(power)
        power_batch = torch.stack(specs, dim=0)  # (B, F, T')

        # Mel filterbank
        mel_fb = self._get_mel_filterbank(n_mels, n_fft, sr, wav.device, wav.dtype)
        mel_spec = torch.matmul(mel_fb, power_batch)  # (B, n_mels, T')
        log_mel = torch.log(mel_spec + 1e-9)
        return log_mel.unsqueeze(1)  # (B, 1, n_mels, T')

    @staticmethod
    @torch.no_grad()
    def _get_mel_filterbank(n_mels: int, n_fft: int, sr: int,
                             device: torch.device,
                             dtype: torch.dtype) -> torch.Tensor:
        """Build mel filterbank using triangular filters."""
        f_min = 0.0
        f_max = sr / 2.0
        n_freqs = n_fft // 2 + 1

        def hz_to_mel(f: float) -> float:
            return 2595.0 * math.log10(1.0 + f / 700.0)

        def mel_to_hz(m: float) -> float:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mel_pts = [mel_min + i * (mel_max - mel_min) / (n_mels + 1)
                   for i in range(n_mels + 2)]
        hz_pts = [mel_to_hz(m) for m in mel_pts]
        bin_pts = [int(round(h * (n_fft + 1) / sr)) for h in hz_pts]

        fb = torch.zeros(n_mels, n_freqs, dtype=dtype, device=device)
        for m in range(1, n_mels + 1):
            lo, ctr, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
            for k in range(lo, ctr):
                if ctr != lo:
                    fb[m - 1, k] = (k - lo) / (ctr - lo)
            for k in range(ctr, hi):
                if hi != ctr:
                    fb[m - 1, k] = (hi - k) / (hi - ctr)
        return fb

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: Raw waveform tensor. Accepted shapes:
                 (T,), (B,T), (B,1,T), (1,T)

        Returns:
            logits: (B,) raw logits. Use sigmoid() for probability.
        """
        wav = self._prepare_waveform(wav)  # (B, 1, T)

        # ── Raw branch ────────────────────────────────────────────────────────
        raw_feat = self.raw_branch(wav)    # (B, C_raw, T')

        # ── Spectral branch ───────────────────────────────────────────────────
        mel = self._mel_spectrogram(wav)   # (B, 1, n_mels, T_spec)
        spec_feat = self.spec_branch(mel)  # (B, C_spec, freq', T_spec')

        B, C, F, Ts = spec_feat.shape
        # Collapse frequency dim, keep time dim
        spec_feat = spec_feat.view(B, C * F, Ts)   # (B, C*F, T_spec')
        spec_feat = self.spec_proj(spec_feat)        # (B, C_raw, T_spec')

        # ── Align time dimensions ─────────────────────────────────────────────
        T_raw = raw_feat.shape[-1]
        T_spec = spec_feat.shape[-1]
        T_min = min(T_raw, T_spec)
        raw_feat = raw_feat[..., :T_min]
        spec_feat = spec_feat[..., :T_min]

        # ── Fusion ────────────────────────────────────────────────────────────
        fused = torch.cat([raw_feat, spec_feat], dim=1)  # (B, 2*C, T_min)
        fused = fused.permute(0, 2, 1)                   # (B, T_min, 2*C)

        # ── BiGRU ────────────────────────────────────────────────────────────
        gru_out, _ = self.gru(fused)  # (B, T_min, 2*gru_hidden)

        # ── Temporal Attention ────────────────────────────────────────────────
        context = self.attention(gru_out)  # (B, 2*gru_hidden)

        # ── Classifier ────────────────────────────────────────────────────────
        logits = self.classifier(context)  # (B, 1)
        return logits.squeeze(-1)           # (B,)

    # ── Utility ───────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        n = self.count_parameters()
        return (
            f"VoxShieldNet v{self.config['version']} | "
            f"Parameters: {n:,} | "
            f"Pretrained: False | "
            f"Branches: Raw-1DCNN + Spectral-2DCNN | "
            f"Temporal: BiGRU + Attention"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(config: Dict[str, Any] | None = None) -> VoxShieldNet:
    """Construct a VoxShieldNet from random initialization."""
    model = VoxShieldNet(config)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    model = build_model()
    print(model.summary())

    # Batch of 4 clips, 16000 samples each (1 second)
    dummy = torch.randn(4, 16000)
    logits = model(dummy)
    probs = torch.sigmoid(logits)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits: {logits.detach()}")
    print(f"Probabilities: {probs.detach()}")
    assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"
    print("Smoke test passed.")
