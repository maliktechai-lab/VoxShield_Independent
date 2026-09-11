"""
VoxShield Inference Engine
===========================

Checkpoint resolution order:
    1. VOXSHIELD_CHECKPOINT environment variable (explicit path)
    2. checkpoints/asvspoof5/best.pt   (preferred — real trained model)
    3. checkpoints/asvspoof5/last.pt   (fallback — last epoch)
    4. checkpoints/best.pt             (demo fallback)
    5. checkpoints/last.pt             (demo fallback)

Behaviour when no checkpoint is found:
    - Returns a clear UNAVAILABLE response.
    - NEVER fabricates a prediction.
    - NEVER silently falls back to random outputs.

Inference output fields:
    classification      "SPOOF" | "BONA_FIDE"
    spoof_probability   float [0, 1]
    bona_fide_probability float [0, 1]
    confidence          float [0, 1]  — distance from 0.5
    decision_threshold  float         — from checkpoint (EER threshold on val)
    risk_score          float [0, 1]
    threat_level        "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    recommended_action  str
    risk_breakdown      dict
    latency_ms          float
    device              str
    model_version       str
    inference_mode      "asvspoof5" | "demo" | "unavailable"
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)

# Checkpoint search order
_DEFAULT_SEARCH_PATHS = [
    "checkpoints/asvspoof5/best.pt",
    "checkpoints/asvspoof5/last.pt",
    "checkpoints/best.pt",
    "checkpoints/last.pt",
]


def _resolve_checkpoint(project_root: Path) -> Optional[Path]:
    """Resolve the checkpoint to load in priority order."""
    # Explicit environment override
    env_path = os.environ.get("VOXSHIELD_CHECKPOINT", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            logger.info(f"Using checkpoint from VOXSHIELD_CHECKPOINT: {p}")
            return p
        else:
            logger.warning(
                f"VOXSHIELD_CHECKPOINT={env_path} does not exist."
            )

    # Search in priority order
    for rel in _DEFAULT_SEARCH_PATHS:
        p = project_root / rel
        if p.exists():
            logger.info(f"Checkpoint resolved: {p}")
            return p

    return None


class InferenceEngine:
    """
    Stateful inference engine. Loads model and checkpoint once at init.

    Usage:
        engine = InferenceEngine()
        result = engine.predict(waveform_tensor)
    """

    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        import sys
        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]
        self.project_root = Path(project_root)

        # Add project root to path so imports resolve
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        # Device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = None
        self.checkpoint_path: Optional[Path] = None
        self.checkpoint_info: Dict = {}
        self.threshold: float = 0.5
        self.inference_mode: str = "unavailable"
        self._model_ready: bool = False

        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        from model.voxshieldnet import build_model

        ckpt_path = _resolve_checkpoint(self.project_root)
        if ckpt_path is None:
            logger.warning(
                "No checkpoint found. "
                "Inference will return UNAVAILABLE until a model is trained. "
                "Run: python -m training.train --asvspoof5-dir <DATASET>"
            )
            self._model_ready = False
            self.inference_mode = "unavailable"
            return

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
            self._model_ready = False
            return

        # Build model using saved config
        model_config = ckpt.get("model_config", None)
        try:
            self.model = build_model(model_config).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to restore model from {ckpt_path}: {e}")
            self._model_ready = False
            return

        # Extract metadata
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.checkpoint_path = ckpt_path
        ds_type = (
            ckpt.get("dataset_info", {}).get("dataset_type", "unknown")
            or ckpt.get("metadata", {}).get("dataset_type", "unknown")
        )
        self.inference_mode = "asvspoof5" if ds_type == "ASVspoof5" else "demo"
        self.checkpoint_info = {
            "path":           str(ckpt_path),
            "epoch":          ckpt.get("epoch", "?"),
            "val_metrics":    ckpt.get("val_metrics", {}),
            "threshold":      self.threshold,
            "dataset_type":   ds_type,
            "training_time":  ckpt.get("metadata", {}).get("training_time", "unknown"),
            "model_version":  ckpt.get("metadata", {}).get("version",
                              ckpt.get("model_config", {}).get("version", "1.0.0")),
            "pretrained":     ckpt.get("metadata", {}).get("pretrained", False),
        }
        self._model_ready = True

        logger.info(
            f"Model loaded: {self.model.summary()} | "
            f"Threshold={self.threshold:.4f} | "
            f"Mode={self.inference_mode} | "
            f"Device={self.device}"
        )

    def reload(self) -> None:
        """Reload the model from disk (e.g., after training finishes)."""
        self._model_ready = False
        self.model = None
        self._load()

    # ── Prediction ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, wav: torch.Tensor) -> Dict:
        """
        Run inference on a preprocessed waveform.

        Args:
            wav: Float32 tensor of shape (N,) or (B,N) at 16 kHz.
                 N should be 64000 (4 seconds).

        Returns:
            Dict with all inference output fields.
        """
        from inference.risk_engine import RiskEngine

        if not self._model_ready or self.model is None:
            return self._unavailable_response()

        # Timing
        t0 = time.perf_counter()

        # Shape handling
        if wav.dim() == 1:
            wav_input = wav.unsqueeze(0)  # (1, N)
        elif wav.dim() == 2:
            wav_input = wav
        else:
            return self._error_response(
                f"Unexpected waveform shape: {wav.shape}"
            )

        wav_input = wav_input.to(self.device)

        try:
            logit = self.model(wav_input)    # (B,)
            prob  = torch.sigmoid(logit)     # (B,) probabilities
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return self._error_response(f"Inference failed: {e}")

        latency_ms = (time.perf_counter() - t0) * 1000

        spoof_prob = float(prob[0].item())
        bf_prob    = 1.0 - spoof_prob
        thr        = self.threshold

        # Classification
        classification = "SPOOF" if spoof_prob >= thr else "BONA_FIDE"
        # Confidence = how far from the decision boundary (0.5 normalized)
        confidence = abs(spoof_prob - 0.5) * 2.0  # [0, 1]

        # Risk engine
        risk_result = RiskEngine.score(spoof_prob, thr)

        result = {
            "classification":       classification,
            "spoof_probability":    round(spoof_prob, 6),
            "bona_fide_probability": round(bf_prob, 6),
            "confidence":           round(confidence, 4),
            "decision_threshold":   round(thr, 6),
            "risk_score":           risk_result["risk_score"],
            "threat_level":         risk_result["threat_level"],
            "recommended_action":   risk_result["recommended_action"],
            "risk_breakdown":       risk_result["breakdown"],
            "latency_ms":           round(latency_ms, 2),
            "device":               str(self.device),
            "model_version":        self.checkpoint_info.get("model_version", "1.0.0"),
            "inference_mode":       self.inference_mode,
            "status":               "ok",
            "error":                None,
        }
        return result

    # ── Status ────────────────────────────────────────────────────────────────

    def model_info(self) -> Dict:
        """Return detailed model and checkpoint information."""
        if not self._model_ready or self.model is None:
            return {
                "status":       "unavailable",
                "message":      "No trained model found. Run training first.",
                "pretrained":   False,
            }

        val_metrics = self.checkpoint_info.get("val_metrics", {})
        ds_info = {}
        if self.checkpoint_path:
            try:
                raw = torch.load(
                    self.checkpoint_path, map_location="cpu", weights_only=False
                )
                ds_info = raw.get("dataset_info", {})
            except Exception:
                pass

        return {
            "status":             "ready",
            "model_name":         "VoxShieldNet",
            "model_version":      self.checkpoint_info.get("model_version", "1.0.0"),
            "parameter_count":    self.model.count_parameters(),
            "device":             str(self.device),
            "checkpoint":         str(self.checkpoint_path),
            "dataset_type":       self.checkpoint_info.get("dataset_type", "unknown"),
            "dataset_source":     ds_info.get("dataset_dir", "unknown"),
            "training_timestamp": self.checkpoint_info.get("training_time", "unknown"),
            "epoch":              self.checkpoint_info.get("epoch", "?"),
            "val_metrics":        val_metrics,
            "dataset_counts": {
                "train": ds_info.get("train_stats", {}).get("total", "?"),
                "val":   ds_info.get("val_stats", {}).get("total", "?"),
            },
            "decision_threshold": self.threshold,
            "inference_mode":     self.inference_mode,
            "pretrained":         False,
            "architecture": {
                "type":     "Dual-branch CNN+BiGRU+Attention",
                "branches": ["Raw 1D CNN", "Log-Mel 2D CNN"],
                "fusion":   "Bidirectional GRU + Temporal Attention",
            },
        }

    def is_ready(self) -> bool:
        return self._model_ready

    # ── Private ───────────────────────────────────────────────────────────────

    def _unavailable_response(self) -> Dict:
        return {
            "classification":       "UNAVAILABLE",
            "spoof_probability":    None,
            "bona_fide_probability": None,
            "confidence":           None,
            "decision_threshold":   None,
            "risk_score":           None,
            "threat_level":         "UNKNOWN",
            "recommended_action":   (
                "Model not available. Train VoxShieldNet first: "
                "python -m training.train --asvspoof5-dir <DATASET>"
            ),
            "risk_breakdown":       {},
            "latency_ms":           0.0,
            "device":               str(self.device),
            "model_version":        None,
            "inference_mode":       "unavailable",
            "status":               "unavailable",
            "error":                "No checkpoint found",
        }

    def _error_response(self, msg: str) -> Dict:
        return {
            "classification":       "ERROR",
            "spoof_probability":    None,
            "bona_fide_probability": None,
            "confidence":           None,
            "decision_threshold":   self.threshold,
            "risk_score":           None,
            "threat_level":         "UNKNOWN",
            "recommended_action":   "Audio processing error. Check input file.",
            "risk_breakdown":       {},
            "latency_ms":           0.0,
            "device":               str(self.device),
            "model_version":        self.checkpoint_info.get("model_version"),
            "inference_mode":       self.inference_mode,
            "status":               "error",
            "error":                msg,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Singleton (lazy-loaded at first use by backend)
# ──────────────────────────────────────────────────────────────────────────────

_engine_instance: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = InferenceEngine()
    return _engine_instance


def reload_engine() -> InferenceEngine:
    global _engine_instance
    _engine_instance = None
    return get_engine()
