"""
VoxShield Audio Preprocessor
==============================

Handles loading audio from files or bytes and preparing it for VoxShieldNet.

Supports:
    .flac, .wav, .mp3, .ogg, .m4a (anything soundfile or scipy can read)

Constraints:
    - 16 kHz mono
    - Fixed 4-second window (64000 samples)
    - No pretrained speech encoder
    - All processing is deterministic and local
"""

from __future__ import annotations

import io
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SAMPLE_RATE   = 16_000
MAX_SEC       = 4.0
MAX_SAMPLES   = int(SAMPLE_RATE * MAX_SEC)
MIN_SAMPLES   = int(SAMPLE_RATE * 0.1)   # at least 0.1 s of audio
SUPPORTED_EXT = {".flac", ".wav", ".ogg", ".mp3", ".m4a", ".mp4"}


class AudioPreprocessorError(ValueError):
    """Raised when audio cannot be preprocessed for a clear, reportable reason."""
    pass


class AudioPreprocessor:
    """
    Load, validate, and prepare audio for VoxShieldNet.

    Usage:
        prep = AudioPreprocessor()
        waveform = prep.from_file("audio.flac")      # → (64000,) tensor
        waveform = prep.from_bytes(raw_bytes, ".wav") # → (64000,) tensor
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        max_samples: int = MAX_SAMPLES,
        min_samples: int = MIN_SAMPLES,
    ):
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.min_samples = min_samples

    # ── Public API ────────────────────────────────────────────────────────────

    def from_file(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Load audio from a file path.

        Returns:
            Tensor of shape (max_samples,), float32, 16 kHz mono.

        Raises:
            AudioPreprocessorError: if the file cannot be decoded or is too short.
        """
        path = Path(path)
        if not path.exists():
            raise AudioPreprocessorError(f"Audio file not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXT:
            raise AudioPreprocessorError(
                f"Unsupported audio format: {suffix}. "
                f"Supported: {sorted(SUPPORTED_EXT)}"
            )

        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as e:
            # Try scipy as fallback for wav
            if suffix == ".wav":
                try:
                    from scipy.io import wavfile
                    sr, data = wavfile.read(str(path))
                    data = data.astype(np.float32)
                    if data.max() > 1.0 or data.min() < -1.0:
                        data = data / max(abs(data.max()), abs(data.min()), 32768.0)
                except Exception as e2:
                    raise AudioPreprocessorError(
                        f"Cannot decode audio file '{path}': {e2}"
                    ) from e2
            else:
                raise AudioPreprocessorError(
                    f"Cannot decode audio file '{path}': {e}"
                ) from e

        return self._process(data, sr, source=str(path))

    def from_bytes(
        self,
        raw_bytes: bytes,
        extension: str = ".wav",
    ) -> torch.Tensor:
        """
        Load audio from raw bytes (e.g., uploaded file content).

        Args:
            raw_bytes: Raw audio file bytes.
            extension: File extension hint for format detection.

        Returns:
            Tensor of shape (max_samples,), float32, 16 kHz mono.
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        if ext not in SUPPORTED_EXT:
            raise AudioPreprocessorError(
                f"Unsupported audio format: {ext}"
            )

        # Try soundfile directly from buffer
        try:
            with io.BytesIO(raw_bytes) as buf:
                data, sr = sf.read(buf, dtype="float32", always_2d=False)
            return self._process(data, sr, source="<bytes>")
        except Exception:
            pass

        # Fallback: write to temp file
        suffix = ext if ext.startswith(".") else "." + ext
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False
            ) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            return self.from_file(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def from_numpy(self, data: np.ndarray, sr: int) -> torch.Tensor:
        """Process a raw numpy array (float32, any sample rate, mono or stereo)."""
        return self._process(data, sr, source="<numpy>")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process(
        self,
        data: np.ndarray,
        sr: int,
        source: str = "",
    ) -> torch.Tensor:
        """Validate, normalize, resample, fix length."""
        if data is None or len(data) == 0:
            raise AudioPreprocessorError(f"Empty audio data from {source}")

        # Ensure float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Mono: average channels
        if data.ndim == 2:
            data = data.mean(axis=1)

        # Normalize amplitude (avoid silent / clipped inputs)
        peak = np.abs(data).max()
        if peak < 1e-8:
            raise AudioPreprocessorError(
                "Audio is silent (peak amplitude < 1e-8). "
                "Please provide a valid audio sample."
            )
        if peak > 1.0:
            data = data / peak  # peak normalize

        wav = torch.from_numpy(data)

        # Resample if needed
        if sr != self.sample_rate:
            logger.info(f"Resampling from {sr} Hz → {self.sample_rate} Hz")
            try:
                import torchaudio.functional as AF
                wav = AF.resample(wav.unsqueeze(0), sr, self.sample_rate).squeeze(0)
            except Exception:
                # scipy fallback
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(sr, self.sample_rate)
                up, down = self.sample_rate // g, sr // g
                data_rs = resample_poly(wav.numpy(), up, down).astype(np.float32)
                wav = torch.from_numpy(data_rs)

        # Check minimum length
        if wav.shape[0] < self.min_samples:
            raise AudioPreprocessorError(
                f"Audio too short: {wav.shape[0] / self.sample_rate:.2f}s "
                f"(minimum {self.min_samples / self.sample_rate:.2f}s required)."
            )

        # Pad or truncate to fixed length
        n = wav.shape[0]
        if n >= self.max_samples:
            wav = wav[: self.max_samples]
        else:
            wav = F.pad(wav, (0, self.max_samples - n))

        return wav  # (max_samples,)

    def validate_upload_size(
        self,
        size_bytes: int,
        max_mb: float = 25.0,
    ) -> None:
        """Raise AudioPreprocessorError if upload is too large."""
        max_bytes = int(max_mb * 1024 * 1024)
        if size_bytes > max_bytes:
            raise AudioPreprocessorError(
                f"Upload size {size_bytes / 1e6:.1f} MB exceeds "
                f"maximum {max_mb:.0f} MB."
            )

    def info(self, path: Union[str, Path]) -> dict:
        """Return audio file metadata without decoding the full waveform."""
        try:
            info = sf.info(str(path))
            return {
                "sample_rate":  info.samplerate,
                "channels":     info.channels,
                "duration_sec": info.duration,
                "frames":       info.frames,
                "format":       info.format,
                "subtype":      info.subtype,
            }
        except Exception as e:
            return {"error": str(e)}
