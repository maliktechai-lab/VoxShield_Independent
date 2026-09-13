"""
VoxShield Long-Audio Inference
================================

Applies the fixed 4-second VoxShieldNet classifier to an arbitrary-length
waveform using an overlapping sliding-window strategy.

Key design:
- Full audio is decoded WITHOUT truncation (unlike AudioPreprocessor.from_bytes,
  which truncates to 4 s for the /predict endpoint).
- Audio is resampled to 16 kHz mono.
- Overlapping 4-second windows with configurable hop size.
- Each window is padded with zeros if shorter than 4 seconds (end of file).
- Window-level inference is run through the existing InferenceEngine.
- Aggregation: mean of the top-K spoof scores (K = ceil(N * top_fraction)).
  This is conservative: it reflects the worst portion of the audio.
- Final verdict uses the same threshold as InferenceEngine.
- Returns rich metadata including per-window results.

Usage:
    from inference.long_audio import LongAudioAnalyzer, decode_full_audio
    analyzer = LongAudioAnalyzer(engine)
    wav = decode_full_audio(raw_bytes, ".flac")
    result = analyzer.analyze(wav)
"""

from __future__ import annotations

import io
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLE_RATE: int = 16_000
WINDOW_SEC: float = 4.0
HOP_SEC: float = 2.0
TOP_FRACTION: float = 0.25
MAX_WINDOWS: int = 120
# Maximum upload size checked externally; cap internal decode at 300 s to avoid
# runaway memory on very large files.
MAX_AUDIO_DURATION_SEC: float = 300.0
MIN_AMPLITUDE: float = 1e-8
SUPPORTED_EXT = frozenset({".flac", ".wav", ".ogg", ".mp3", ".m4a", ".mp4"})


class LongAudioError(ValueError):
    """Raised when long-audio decoding or analysis fails for a reportable reason."""
    pass


# ── Full-audio decoder (NO truncation) ────────────────────────────────────────

def decode_full_audio(
    raw_bytes: bytes,
    extension: str = ".wav",
    *,
    sample_rate: int = SAMPLE_RATE,
    max_duration_sec: float = MAX_AUDIO_DURATION_SEC,
) -> torch.Tensor:
    """
    Decode audio bytes to a full-length float32 16 kHz mono tensor.

    Unlike AudioPreprocessor.from_bytes / _process, this function does NOT
    truncate the audio to 4 seconds.  The caller is responsible for windowing.

    Args:
        raw_bytes:        Raw audio file content.
        extension:        File extension hint (.flac, .wav, etc.)
        sample_rate:      Target sample rate (default 16 000 Hz).
        max_duration_sec: Reject audio longer than this (safety limit).

    Returns:
        1-D float32 tensor at the requested sample rate.

    Raises:
        LongAudioError: if audio cannot be decoded, is empty, silent, or
                        exceeds max_duration_sec after resampling.
    """
    import soundfile as sf

    ext = extension.lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext not in SUPPORTED_EXT:
        raise LongAudioError(
            f"Unsupported audio format: '{ext}'. "
            f"Accepted: {sorted(SUPPORTED_EXT)}"
        )

    if not raw_bytes:
        raise LongAudioError("Empty audio data (0 bytes).")

    # ── Attempt 1: soundfile from BytesIO ─────────────────────────────────────
    data: Optional[np.ndarray] = None
    sr: Optional[int] = None

    try:
        with io.BytesIO(raw_bytes) as buf:
            data, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception as sf_exc:
        # ── Attempt 2: temp file (handles formats needing file-backed I/O) ────
        tmp_path: Optional[str] = None
        try:
            suffix = ext if ext.startswith(".") else "." + ext
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False
            ) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            data, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
        except Exception:
            # ── Attempt 3: scipy WAV fallback ─────────────────────────────────
            if ext == ".wav":
                try:
                    import scipy.io.wavfile as wavfile
                    from io import BytesIO
                    sr_raw, raw_data = wavfile.read(BytesIO(raw_bytes))
                    raw_data = raw_data.astype(np.float32)
                    if raw_data.max() > 1.0 or raw_data.min() < -1.0:
                        scale = max(
                            abs(float(raw_data.max())),
                            abs(float(raw_data.min())),
                            32768.0,
                        )
                        raw_data = raw_data / scale
                    data, sr = raw_data, sr_raw
                except Exception as scipy_exc:
                    raise LongAudioError(
                        f"Cannot decode audio (tried soundfile + scipy): "
                        f"soundfile={sf_exc}, scipy={scipy_exc}"
                    ) from scipy_exc
            else:
                raise LongAudioError(
                    f"Cannot decode '{ext}' audio: {sf_exc}"
                ) from sf_exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── Validate raw decoded data ──────────────────────────────────────────────
    if data is None or len(data) == 0:
        raise LongAudioError("Decoded audio is empty.")

    # Ensure float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Mono: average channels
    if data.ndim == 2:
        data = data.mean(axis=1)
    elif data.ndim != 1:
        raise LongAudioError(
            f"Unexpected audio shape after decode: {data.shape}"
        )

    # Check for non-finite values
    if not np.isfinite(data).all():
        n_bad = int(np.sum(~np.isfinite(data)))
        logger.warning(
            f"Audio contains {n_bad} non-finite samples; replacing with zeros."
        )
        data = np.where(np.isfinite(data), data, 0.0)

    # Silence check
    peak = float(np.abs(data).max())
    if peak < MIN_AMPLITUDE:
        raise LongAudioError(
            "Audio is silent (peak amplitude < 1e-8). "
            "Please provide a valid audio sample."
        )

    # Peak normalise so all windows see amplitude ≤ 1
    if peak > 1.0:
        data = data / peak

    wav = torch.from_numpy(data)

    # ── Resample if needed ─────────────────────────────────────────────────────
    assert sr is not None
    if sr != sample_rate:
        logger.info(f"Long-audio: resampling {sr} Hz → {sample_rate} Hz")
        try:
            import torchaudio.functional as AF
            wav = AF.resample(
                wav.unsqueeze(0), orig_freq=sr, new_freq=sample_rate
            ).squeeze(0)
        except Exception:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), sample_rate)
            up = sample_rate // g
            down = int(sr) // g
            resampled = resample_poly(
                wav.numpy(), up, down
            ).astype(np.float32)
            wav = torch.from_numpy(resampled)

    # ── Duration cap ───────────────────────────────────────────────────────────
    max_samples = int(max_duration_sec * sample_rate)
    if wav.numel() > max_samples:
        logger.warning(
            f"Long-audio: clipping at {max_duration_sec:.0f}s "
            f"({wav.numel()} → {max_samples} samples)."
        )
        wav = wav[:max_samples]

    return wav  # (N,) float32 at 16 kHz


# ── Long-audio analyzer ────────────────────────────────────────────────────────

class LongAudioAnalyzer:
    """
    Apply the fixed 4-second VoxShieldNet classifier over an arbitrary-length
    waveform using overlapping sliding windows.

    Args:
        engine:       InferenceEngine instance (already loaded).
        window_sec:   Window length in seconds (default 4.0).
        hop_sec:      Hop / stride between window starts in seconds (default 2.0).
        top_fraction: Fraction of highest-spoof windows used for the final score.
                      0.25 → top 25 % (conservative / security-oriented).
        max_windows:  Hard cap on window count.  If the audio would yield more
                      windows after sliding, evenly-spaced windows are sampled
                      so the total does not exceed this limit.
    """

    def __init__(
        self,
        engine: Any,
        window_sec: float = WINDOW_SEC,
        hop_sec: float = HOP_SEC,
        top_fraction: float = TOP_FRACTION,
        max_windows: int = MAX_WINDOWS,
    ) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be > 0")
        if hop_sec <= 0:
            raise ValueError("hop_sec must be > 0")
        if hop_sec > window_sec:
            raise ValueError("hop_sec must be ≤ window_sec")
        if not (0.0 < top_fraction <= 1.0):
            raise ValueError("top_fraction must be in (0, 1]")
        if max_windows < 1:
            raise ValueError("max_windows must be ≥ 1")

        self.engine = engine
        self.sample_rate: int = SAMPLE_RATE
        self.window_samples: int = int(round(window_sec * self.sample_rate))
        self.hop_samples: int = int(round(hop_sec * self.sample_rate))
        self.top_fraction: float = top_fraction
        self.max_windows: int = max_windows

    # ── Window generation ──────────────────────────────────────────────────────

    def _make_windows(
        self, wav: torch.Tensor
    ) -> List[Tuple[int, torch.Tensor]]:
        """
        Slice wav into overlapping fixed-length windows.

        Returns:
            List of (start_sample, window_tensor) tuples.
            Each window_tensor has exactly self.window_samples elements.
            The final window is zero-padded if the audio is shorter.
        """
        if wav.dim() != 1:
            wav = wav.flatten()

        total: int = int(wav.numel())

        # ── Short audio: single zero-padded window ─────────────────────────────
        if total <= self.window_samples:
            pad_len = self.window_samples - total
            padded = F.pad(wav, (0, pad_len))
            return [(0, padded)]

        # ── Build start positions ──────────────────────────────────────────────
        # Standard sliding window up to the last full-length window.
        max_start = total - self.window_samples
        starts: List[int] = list(
            range(0, max_start + 1, self.hop_samples)
        )

        # Always include the last aligned window so the tail is covered.
        if starts[-1] != max_start:
            starts.append(max_start)

        # ── Apply max_windows cap ──────────────────────────────────────────────
        if len(starts) > self.max_windows:
            # Evenly subsample, always keeping the first and last.
            stride = math.ceil((len(starts) - 1) / (self.max_windows - 1))
            capped: List[int] = starts[::stride]
            if capped[-1] != max_start:
                if len(capped) < self.max_windows:
                    capped.append(max_start)
                else:
                    capped[-1] = max_start
            starts = capped[: self.max_windows]

        return [
            (start, wav[start : start + self.window_samples])
            for start in starts
        ]

    # ── Analysis ───────────────────────────────────────────────────────────────

    def analyze(self, wav: torch.Tensor) -> Dict[str, Any]:
        """
        Run per-window inference and aggregate into a final verdict.

        Args:
            wav: 1-D float32 tensor at 16 kHz (from decode_full_audio).

        Returns:
            Dict with full result including window_results list.
        """
        if wav.dim() != 1:
            wav = wav.flatten()

        audio_duration_sec: float = round(
            float(wav.numel()) / self.sample_rate, 3
        )
        window_seconds: float = self.window_samples / self.sample_rate
        hop_seconds: float = self.hop_samples / self.sample_rate

        windows = self._make_windows(wav)
        window_results: List[Dict[str, Any]] = []

        for start, window in windows:
            # Each window is exactly window_samples long (padded if needed).
            per_result = self.engine.predict(window)
            window_results.append(
                {
                    "start_sec": round(start / self.sample_rate, 3),
                    "end_sec": round(
                        (start + self.window_samples) / self.sample_rate, 3
                    ),
                    "classification":    per_result.get("classification"),
                    "spoof_probability": per_result.get("spoof_probability"),
                    "confidence":        per_result.get("confidence"),
                    "risk_score":        per_result.get("risk_score"),
                    "threat_level":      per_result.get("threat_level"),
                    "recommended_action": per_result.get("recommended_action"),
                    "latency_ms":        per_result.get("latency_ms"),
                }
            )

        # ── Collect valid numeric spoof scores ─────────────────────────────────
        valid_scores: List[float] = [
            float(w["spoof_probability"])
            for w in window_results
            if isinstance(w.get("spoof_probability"), (int, float))
            and math.isfinite(float(w["spoof_probability"]))
        ]

        # ── Handle model-unavailable / all-error case ──────────────────────────
        if not valid_scores:
            return {
                "classification":       "UNAVAILABLE",
                "status":               "unavailable",
                "spoof_probability":    None,
                "bona_fide_probability": None,
                "confidence":           None,
                "decision_threshold":   (
                    float(self.engine.threshold)
                    if self.engine.is_ready()
                    else None
                ),
                "risk_score":           None,
                "threat_level":         "UNKNOWN",
                "recommended_action":   (
                    "Model not available. Train VoxShieldNet first."
                ),
                "risk_breakdown":       {},
                "model_version":        self.engine.checkpoint_info.get(
                    "model_version", None
                ),
                "inference_mode":       self.engine.inference_mode,
                "device":               str(self.engine.device),
                "audio_duration_sec":   audio_duration_sec,
                "windows_analyzed":     len(window_results),
                "window_seconds":       window_seconds,
                "hop_seconds":          hop_seconds,
                "aggregation": {
                    "method":       "top_fraction_mean",
                    "top_fraction": self.top_fraction,
                    "windows_used_for_final": 0,
                },
                "window_results": window_results,
                "error": "Model unavailable — no valid predictions returned.",
            }

        # ── Aggregation: mean of top-K by spoof score ──────────────────────────
        # Using the top fraction is conservative and security-oriented:
        # even a minority of high-risk windows elevates the verdict.
        scores_tensor = torch.tensor(valid_scores, dtype=torch.float32)
        k = max(1, math.ceil(len(valid_scores) * self.top_fraction))
        top_scores, _ = torch.topk(scores_tensor, k=k)
        final_prob: float = float(top_scores.mean().item())

        threshold: float = float(self.engine.threshold)
        classification = "SPOOF" if final_prob >= threshold else "BONA_FIDE"
        confidence = abs(final_prob - 0.5) * 2.0  # normalised distance from 0.5

        from inference.risk_engine import RiskEngine
        risk = RiskEngine.score(final_prob, threshold)

        return {
            "classification":       classification,
            "spoof_probability":    round(final_prob, 6),
            "bona_fide_probability": round(1.0 - final_prob, 6),
            "confidence":           round(confidence, 4),
            "decision_threshold":   round(threshold, 6),
            "risk_score":           risk["risk_score"],
            "threat_level":         risk["threat_level"],
            "recommended_action":   risk["recommended_action"],
            "risk_breakdown":       risk["breakdown"],
            "status":               "ok",
            "model_version":        self.engine.checkpoint_info.get(
                "model_version", "1.0.0"
            ),
            "inference_mode":       self.engine.inference_mode,
            "device":               str(self.engine.device),
            "audio_duration_sec":   audio_duration_sec,
            "windows_analyzed":     len(window_results),
            "window_seconds":       window_seconds,
            "hop_seconds":          hop_seconds,
            "aggregation": {
                "method":                 "top_fraction_mean",
                "top_fraction":           self.top_fraction,
                "windows_used_for_final": k,
                "all_window_scores":      [round(s, 6) for s in valid_scores],
            },
            "window_results": window_results,
            "error": None,
        }
