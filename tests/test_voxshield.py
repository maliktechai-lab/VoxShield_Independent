"""
VoxShield Test Suite
=====================

Tests are organized to run without ASVspoof5 dataset.
Integration tests requiring the full dataset are marked with @pytest.mark.integration.

Run all unit tests:
    python -m pytest tests/ -v

Run with coverage:
    python -m pytest tests/ -v --tb=short

Skip integration tests (default):
    python -m pytest tests/ -v -m "not integration"
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ── Make project root importable ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Test: Model architecture
# ──────────────────────────────────────────────────────────────────────────────

class TestVoxShieldNet:
    """Tests for the VoxShieldNet model architecture."""

    def test_model_builds(self):
        from model.voxshieldnet import build_model
        model = build_model()
        assert model is not None

    def test_model_output_shape_single(self):
        from model.voxshieldnet import build_model
        model = build_model()
        model.eval()
        with torch.no_grad():
            wav = torch.randn(1, 64000)
            logits = model(wav)
        assert logits.shape == (1,), f"Expected (1,), got {logits.shape}"

    def test_model_output_shape_batch(self):
        from model.voxshieldnet import build_model
        model = build_model()
        model.eval()
        with torch.no_grad():
            wav = torch.randn(4, 64000)
            logits = model(wav)
        assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"

    def test_model_returns_raw_logits_not_probabilities(self):
        """Logits should NOT be bounded to [0,1] by sigmoid inside the model."""
        from model.voxshieldnet import build_model
        torch.manual_seed(42)
        model = build_model()
        model.eval()
        with torch.no_grad():
            wav = torch.randn(8, 64000)
            logits = model(wav)
        # If sigmoid were inside the model, all values would be in [0, 1].
        # After applying raw noise, at least some logits should be > 1 or < 0.
        # We force this by checking the raw distribution.
        probs_if_sigmoid = torch.sigmoid(logits)
        # Verify sigmoid mapping is different from raw logits
        assert not torch.allclose(logits, probs_if_sigmoid, atol=0.01), \
            "Model appears to be outputting post-sigmoid values internally"

    def test_probability_from_logit_in_valid_range(self):
        from model.voxshieldnet import build_model
        model = build_model()
        model.eval()
        with torch.no_grad():
            wav = torch.randn(4, 64000)
            logits = model(wav)
            probs = torch.sigmoid(logits)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_model_config_stored(self):
        from model.voxshieldnet import build_model, MODEL_CONFIG
        model = build_model()
        assert model.config is not None
        assert model.config["name"] == "VoxShieldNet"
        assert model.config["pretrained"] is False

    def test_parameter_count_reasonable(self):
        from model.voxshieldnet import build_model
        model = build_model()
        n = model.count_parameters()
        # Should be 1M–20M for this architecture
        assert 500_000 < n < 30_000_000, f"Unexpected parameter count: {n:,}"

    def test_no_pretrained_components(self):
        """Verify no pretrained speech encoder is used."""
        from model.voxshieldnet import VoxShieldNet
        # Check module names — none should reference forbidden architectures
        forbidden = ['wav2vec', 'hubert', 'whisper', 'wavlm', 'aasist',
                     'encodec', 'speechbrain']
        model = VoxShieldNet()
        module_names = [n.lower() for n, _ in model.named_modules()]
        module_str = ' '.join(module_names)
        for f in forbidden:
            assert f not in module_str, f"Found forbidden component: {f}"

    def test_model_handles_short_audio(self):
        """Model should pad audio shorter than max_samples."""
        from model.voxshieldnet import build_model
        model = build_model()
        model.eval()
        with torch.no_grad():
            # Only 0.5 seconds of audio
            wav = torch.randn(1, 8000)
            logits = model(wav)
        assert logits.shape == (1,)

    def test_model_handles_long_audio(self):
        """Model should truncate audio longer than max_samples."""
        from model.voxshieldnet import build_model
        model = build_model()
        model.eval()
        with torch.no_grad():
            # 10 seconds of audio
            wav = torch.randn(1, 160000)
            logits = model(wav)
        assert logits.shape == (1,)

    def test_random_initialization(self):
        """Two models with different seeds should have different weights."""
        from model.voxshieldnet import build_model
        torch.manual_seed(1)
        m1 = build_model()
        torch.manual_seed(999)
        m2 = build_model()
        w1 = list(m1.parameters())[0].data
        w2 = list(m2.parameters())[0].data
        assert not torch.allclose(w1, w2), "Models with different seeds should differ"

    def test_summary_string(self):
        from model.voxshieldnet import build_model
        model = build_model()
        s = model.summary()
        assert 'VoxShieldNet' in s
        assert 'Pretrained: False' in s
        assert 'BiGRU' in s


# ──────────────────────────────────────────────────────────────────────────────
# Test: Risk Engine
# ──────────────────────────────────────────────────────────────────────────────

class TestRiskEngine:
    """Tests for the deterministic risk scoring policy."""

    def test_low_risk_bonafide(self):
        from inference.risk_engine import RiskEngine
        result = RiskEngine.score(0.02, 0.42)
        assert result["threat_level"] == "LOW"
        assert result["risk_score"] < 0.2

    def test_medium_risk_borderline(self):
        from inference.risk_engine import RiskEngine
        result = RiskEngine.score(0.50, 0.42)
        assert result["threat_level"] in ("MEDIUM", "HIGH")

    def test_high_risk(self):
        from inference.risk_engine import RiskEngine
        result = RiskEngine.score(0.80, 0.42)
        assert result["threat_level"] in ("HIGH", "CRITICAL")

    def test_critical_risk(self):
        from inference.risk_engine import RiskEngine
        result = RiskEngine.score(0.97, 0.42)
        assert result["threat_level"] == "CRITICAL"

    def test_risk_score_monotone(self):
        """Higher spoof probability should give higher risk score."""
        from inference.risk_engine import RiskEngine
        probs = [0.0, 0.2, 0.5, 0.7, 0.95]
        scores = [RiskEngine.score(p, 0.5)["risk_score"] for p in probs]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i+1], \
                f"Risk not monotone: {probs[i]} → {scores[i]}, {probs[i+1]} → {scores[i+1]}"

    def test_risk_score_bounds(self):
        """Risk score should be in [0, 1]."""
        from inference.risk_engine import RiskEngine
        for p in [0.0, 0.5, 1.0]:
            r = RiskEngine.score(p, 0.5)
            assert 0.0 <= r["risk_score"] <= 1.0

    def test_breakdown_fields(self):
        from inference.risk_engine import RiskEngine
        r = RiskEngine.score(0.7, 0.5)
        assert "breakdown" in r
        bd = r["breakdown"]
        assert bd["ml_model"] is False
        assert bd["policy_engine"] == "deterministic"
        assert bd["risk_basis"] == "model_probability"

    def test_recommended_action_not_empty(self):
        from inference.risk_engine import RiskEngine
        for p in [0.01, 0.50, 0.75, 0.98]:
            r = RiskEngine.score(p, 0.5)
            assert len(r["recommended_action"]) > 0

    def test_deterministic(self):
        """Same inputs should always give same outputs."""
        from inference.risk_engine import RiskEngine
        r1 = RiskEngine.score(0.72, 0.45)
        r2 = RiskEngine.score(0.72, 0.45)
        assert r1 == r2


# ──────────────────────────────────────────────────────────────────────────────
# Test: Preprocessor
# ──────────────────────────────────────────────────────────────────────────────

class TestAudioPreprocessor:
    """Tests for audio preprocessing."""

    def test_from_numpy_mono(self):
        from inference.preprocessor import AudioPreprocessor
        prep = AudioPreprocessor()
        data = np.random.randn(32000).astype(np.float32) * 0.5
        wav = prep.from_numpy(data, sr=16000)
        assert wav.shape == (64000,)
        assert wav.dtype == torch.float32

    def test_from_numpy_stereo(self):
        from inference.preprocessor import AudioPreprocessor
        prep = AudioPreprocessor()
        data = np.random.randn(32000, 2).astype(np.float32) * 0.5
        wav = prep.from_numpy(data, sr=16000)
        assert wav.shape == (64000,)

    def test_silent_audio_raises(self):
        from inference.preprocessor import AudioPreprocessor, AudioPreprocessorError
        prep = AudioPreprocessor()
        data = np.zeros(32000, dtype=np.float32)
        with pytest.raises(AudioPreprocessorError, match="silent"):
            prep.from_numpy(data, sr=16000)

    def test_too_short_raises(self):
        from inference.preprocessor import AudioPreprocessor, AudioPreprocessorError
        prep = AudioPreprocessor()
        data = np.random.randn(100).astype(np.float32)
        with pytest.raises(AudioPreprocessorError, match="short"):
            prep.from_numpy(data, sr=16000)

    def test_padding_to_max_length(self):
        from inference.preprocessor import AudioPreprocessor, MAX_SAMPLES
        prep = AudioPreprocessor()
        data = np.random.randn(8000).astype(np.float32)
        wav = prep.from_numpy(data, sr=16000)
        assert wav.shape[0] == MAX_SAMPLES

    def test_truncation_to_max_length(self):
        from inference.preprocessor import AudioPreprocessor, MAX_SAMPLES
        prep = AudioPreprocessor()
        data = np.random.randn(200000).astype(np.float32)
        wav = prep.from_numpy(data, sr=16000)
        assert wav.shape[0] == MAX_SAMPLES

    def test_file_not_found_raises(self):
        from inference.preprocessor import AudioPreprocessor, AudioPreprocessorError
        prep = AudioPreprocessor()
        with pytest.raises(AudioPreprocessorError, match="not found"):
            prep.from_file("/nonexistent/path/audio.wav")

    def test_from_bytes_wav(self):
        """Write a minimal WAV to bytes and decode."""
        import io
        import scipy.io.wavfile as wavfile
        from inference.preprocessor import AudioPreprocessor
        prep = AudioPreprocessor()

        # Create a simple WAV in memory
        data = (np.random.randn(32000) * 10000).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, 16000, data)
        raw_bytes = buf.getvalue()

        wav = prep.from_bytes(raw_bytes, extension=".wav")
        assert wav.shape == (64000,)

    def test_upload_size_limit(self):
        from inference.preprocessor import AudioPreprocessor, AudioPreprocessorError
        prep = AudioPreprocessor()
        with pytest.raises(AudioPreprocessorError, match="exceeds"):
            prep.validate_upload_size(30 * 1024 * 1024, max_mb=25.0)

    def test_upload_within_limit(self):
        from inference.preprocessor import AudioPreprocessor
        prep = AudioPreprocessor()
        prep.validate_upload_size(10 * 1024 * 1024, max_mb=25.0)  # should not raise

    def test_peak_normalization(self):
        """Clipped audio (peak > 1) should be normalized."""
        from inference.preprocessor import AudioPreprocessor
        prep = AudioPreprocessor()
        data = np.random.randn(32000).astype(np.float32) * 5.0  # peak > 1
        wav = prep.from_numpy(data, sr=16000)
        assert float(wav.abs().max()) <= 1.0 + 1e-5

    def test_resampling(self):
        from inference.preprocessor import AudioPreprocessor, MAX_SAMPLES
        prep = AudioPreprocessor()
        data = np.random.randn(32000).astype(np.float32) * 0.5
        wav = prep.from_numpy(data, sr=8000)  # resample from 8kHz → 16kHz
        assert wav.shape == (MAX_SAMPLES,)


# ──────────────────────────────────────────────────────────────────────────────
# Test: Inference Engine
# ──────────────────────────────────────────────────────────────────────────────

class TestInferenceEngine:
    """Tests for the InferenceEngine (no real checkpoint required)."""

    def test_engine_unavailable_without_checkpoint(self, tmp_path):
        """Engine should gracefully handle missing checkpoint."""
        from inference.engine import InferenceEngine
        engine = InferenceEngine(project_root=tmp_path)
        assert not engine.is_ready()
        assert engine.inference_mode == "unavailable"

    def test_predict_returns_unavailable_dict(self, tmp_path):
        from inference.engine import InferenceEngine
        engine = InferenceEngine(project_root=tmp_path)
        wav = torch.randn(64000)
        result = engine.predict(wav)
        assert result["status"] == "unavailable"
        assert result["classification"] == "UNAVAILABLE"
        assert result["spoof_probability"] is None
        assert result["error"] is not None

    def test_model_info_unavailable(self, tmp_path):
        from inference.engine import InferenceEngine
        engine = InferenceEngine(project_root=tmp_path)
        info = engine.model_info()
        assert info["status"] == "unavailable"
        assert info["pretrained"] is False

    def test_engine_with_synthetic_checkpoint(self, tmp_path):
        """Create a minimal real checkpoint and verify engine loads it."""
        from model.voxshieldnet import build_model, MODEL_CONFIG
        from inference.engine import InferenceEngine
        import time

        ckpt_dir = tmp_path / "checkpoints" / "asvspoof5"
        ckpt_dir.mkdir(parents=True)

        model = build_model()
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_config":     model.config,
            "epoch":            1,
            "threshold":        0.42,
            "val_metrics":      {"roc_auc": 0.78, "eer": 0.21, "f1": 0.74},
            "dataset_info":     {"dataset_type": "ASVspoof5"},
            "metadata":         {
                "model_name": "VoxShieldNet",
                "version": "1.0.0",
                "dataset_type": "ASVspoof5",
                "pretrained": False,
                "training_time": "2026-01-01T00:00:00Z",
            },
        }
        ckpt_path = ckpt_dir / "best.pt"
        torch.save(ckpt, ckpt_path)

        engine = InferenceEngine(project_root=tmp_path)
        assert engine.is_ready()
        assert engine.threshold == 0.42
        assert engine.inference_mode == "asvspoof5"

    def test_threshold_from_checkpoint_not_hardcoded(self, tmp_path):
        """Threshold must come from checkpoint, not hardcoded 0.5."""
        from model.voxshieldnet import build_model
        from inference.engine import InferenceEngine

        ckpt_dir = tmp_path / "checkpoints" / "asvspoof5"
        ckpt_dir.mkdir(parents=True)
        model = build_model()
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_config":     model.config,
            "epoch":            5,
            "threshold":        0.3751,   # non-trivial threshold
            "dataset_info":     {"dataset_type": "ASVspoof5"},
            "metadata":         {"pretrained": False, "training_time": ""},
        }
        torch.save(ckpt, ckpt_dir / "best.pt")

        engine = InferenceEngine(project_root=tmp_path)
        assert engine.threshold == pytest.approx(0.3751, abs=1e-4)
        assert engine.threshold != 0.5

    def test_predict_uses_threshold_for_classification(self, tmp_path):
        """When spoof_prob >= threshold → SPOOF, else → BONA_FIDE."""
        from model.voxshieldnet import build_model
        from inference.engine import InferenceEngine
        import torch.nn as nn

        ckpt_dir = tmp_path / "checkpoints" / "asvspoof5"
        ckpt_dir.mkdir(parents=True)
        model = build_model()
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_config":     model.config,
            "epoch":            1,
            "threshold":        0.5,
            "dataset_info":     {"dataset_type": "ASVspoof5"},
            "metadata":         {"pretrained": False, "training_time": ""},
        }
        torch.save(ckpt, ckpt_dir / "best.pt")

        engine = InferenceEngine(project_root=tmp_path)
        assert engine.is_ready()

        wav = torch.randn(64000)
        result = engine.predict(wav)
        assert result["status"] == "ok"
        sp = result["spoof_probability"]
        thr = result["decision_threshold"]
        expected_cls = "SPOOF" if sp >= thr else "BONA_FIDE"
        assert result["classification"] == expected_cls

    def test_env_checkpoint_override(self, tmp_path, monkeypatch):
        """VOXSHIELD_CHECKPOINT env var should override default path."""
        from model.voxshieldnet import build_model
        from inference.engine import InferenceEngine

        ckpt_path = tmp_path / "my_model.pt"
        model = build_model()
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_config":     model.config,
            "epoch":            1,
            "threshold":        0.6,
            "dataset_info":     {"dataset_type": "ASVspoof5"},
            "metadata":         {"pretrained": False, "training_time": ""},
        }
        torch.save(ckpt, ckpt_path)

        monkeypatch.setenv("VOXSHIELD_CHECKPOINT", str(ckpt_path))
        engine = InferenceEngine(project_root=tmp_path)
        assert engine.is_ready()
        assert engine.checkpoint_path == ckpt_path


# ──────────────────────────────────────────────────────────────────────────────
# Test: Incident Logger
# ──────────────────────────────────────────────────────────────────────────────

class TestIncidentLogger:
    """Tests for the SQLite incident logger."""

    @pytest.fixture
    def logger(self, tmp_path):
        from backend.incident_logger import IncidentLogger
        return IncidentLogger(db_path=tmp_path / "test.db")

    def _make_result(self, classification="SPOOF"):
        return {
            "classification":       classification,
            "spoof_probability":    0.87,
            "bona_fide_probability": 0.13,
            "confidence":           0.74,
            "risk_score":           0.82,
            "threat_level":         "HIGH",
            "decision_threshold":   0.42,
            "recommended_action":   "CHALLENGE",
            "latency_ms":           42.5,
            "device":               "cpu",
            "model_version":        "1.0.0",
            "inference_mode":       "asvspoof5",
            "status":               "ok",
            "error":                None,
        }

    def test_log_incident(self, logger):
        result = self._make_result()
        row_id = logger.log_incident(result, filename="test.flac")
        assert row_id > 0

    def test_get_incidents(self, logger):
        logger.log_incident(self._make_result("SPOOF"))
        logger.log_incident(self._make_result("BONA_FIDE"))
        incidents = logger.get_incidents(limit=10)
        assert len(incidents) == 2

    def test_get_stats(self, logger):
        logger.log_incident(self._make_result("SPOOF"))
        logger.log_incident(self._make_result("SPOOF"))
        logger.log_incident(self._make_result("BONA_FIDE"))
        stats = logger.get_stats()
        assert stats["total"] == 3
        assert stats["by_classification"].get("SPOOF", 0) == 2
        assert stats["by_classification"].get("BONA_FIDE", 0) == 1

    def test_timeline(self, logger):
        for i in range(5):
            logger.log_incident(self._make_result())
        tl = logger.get_recent_timeline(3)
        assert len(tl) == 3

    def test_empty_db_stats(self, logger):
        stats = logger.get_stats()
        assert stats["total"] == 0

    def test_filename_sanitized(self, logger):
        result = self._make_result()
        # Attempt path traversal in filename
        logger.log_incident(result, filename="../../etc/passwd")
        incidents = logger.get_incidents(limit=1)
        # Should only keep the basename
        assert incidents[0]["filename"] == "passwd"

    def test_filter_by_classification(self, logger):
        logger.log_incident(self._make_result("SPOOF"))
        logger.log_incident(self._make_result("BONA_FIDE"))
        spoof_only = logger.get_incidents(classification="SPOOF")
        assert all(i["classification"] == "SPOOF" for i in spoof_only)

    def test_multiple_writes_incremental_ids(self, logger):
        ids = []
        for _ in range(5):
            ids.append(logger.log_incident(self._make_result()))
        assert len(set(ids)) == 5
        assert ids == sorted(ids)


# ──────────────────────────────────────────────────────────────────────────────
# Test: Checkpoint handling
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckpointHandling:
    """Tests for checkpoint saving, loading, and isolation."""

    def test_atomic_save_creates_file(self, tmp_path):
        from training.train import _atomic_save
        path = tmp_path / "test.pt"
        _atomic_save({"key": "value"}, path)
        assert path.exists()
        assert not (tmp_path / "test.tmp").exists()

    def test_atomic_save_content_correct(self, tmp_path):
        from training.train import _atomic_save
        path = tmp_path / "test.pt"
        _atomic_save({"epoch": 42, "threshold": 0.37}, path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["epoch"] == 42
        assert data["threshold"] == pytest.approx(0.37)

    def test_load_checkpoint_missing(self, tmp_path):
        from training.train import load_checkpoint
        result = load_checkpoint(tmp_path / "nonexistent.pt")
        assert result is None

    def test_checkpoint_includes_model_config(self, tmp_path):
        """Saved checkpoint must contain model_config for reconstruction."""
        from model.voxshieldnet import build_model
        from training.train import save_checkpoint

        model = build_model()
        opt = torch.optim.Adam(model.parameters())
        ckpt_path = tmp_path / "test.pt"

        save_checkpoint(
            ckpt_path, model, opt, None, None,
            epoch=1,
            val_metrics={"roc_auc": 0.75},
            dataset_info={"dataset_type": "ASVspoof5"},
            threshold=0.44,
            best_auc=0.75,
            training_history=[],
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_config" in ckpt
        assert ckpt["model_config"]["pretrained"] is False
        assert ckpt["threshold"] == pytest.approx(0.44)

    def test_checkpoint_pretrained_false(self, tmp_path):
        from model.voxshieldnet import build_model
        from training.train import save_checkpoint

        model = build_model()
        opt = torch.optim.Adam(model.parameters())
        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(
            ckpt_path, model, opt, None, None,
            epoch=1, val_metrics={},
            dataset_info={"dataset_type": "ASVspoof5"},
            threshold=0.5, best_auc=0.0, training_history=[],
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["metadata"]["pretrained"] is False

    def test_dataset_type_stored_in_checkpoint(self, tmp_path):
        from model.voxshieldnet import build_model
        from training.train import save_checkpoint

        model = build_model()
        opt = torch.optim.Adam(model.parameters())
        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(
            ckpt_path, model, opt, None, None,
            epoch=3,
            val_metrics={"roc_auc": 0.82},
            dataset_info={"dataset_type": "ASVspoof5", "dataset_dir": "/data"},
            threshold=0.38, best_auc=0.82, training_history=[],
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["dataset_info"]["dataset_type"] == "ASVspoof5"


# ──────────────────────────────────────────────────────────────────────────────
# Test: API (FastAPI test client)
# ──────────────────────────────────────────────────────────────────────────────

class TestAPI:
    """Integration tests for the FastAPI backend."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a test client with a fresh empty project root (no checkpoint)."""
        from fastapi.testclient import TestClient
        import backend.app as app_module

        # Point engine to empty tmp dir
        original_engine = app_module._engine
        from inference.engine import InferenceEngine
        app_module._engine = InferenceEngine(project_root=tmp_path)
        app_module._incident_log = None  # reset to trigger re-init

        from backend.incident_logger import IncidentLogger
        app_module._incident_log = IncidentLogger(db_path=tmp_path / "test.db")

        client = TestClient(app_module.app, raise_server_exceptions=False)
        yield client

        app_module._engine = original_engine

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "model_ready" in data

    def test_health_model_not_ready_without_checkpoint(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["model_ready"] is False

    def test_model_info_endpoint(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("pretrained") is False

    def test_model_info_shows_unavailable(self, client):
        resp = client.get("/model-info")
        data = resp.json()
        assert data["status"] == "unavailable"

    def test_incidents_empty(self, client):
        resp = client.get("/incidents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["incidents"] == []
        assert data["count"] == 0

    def test_incidents_stats_empty(self, client):
        resp = client.get("/incidents/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_predict_without_file_returns_422(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 422

    def test_predict_returns_unavailable_without_checkpoint(self, client):
        """Without a checkpoint, predict should return unavailable, not crash."""
        import io, scipy.io.wavfile as wavfile
        data = (np.random.randn(16000) * 1000).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, 16000, data)
        wav_bytes = buf.getvalue()

        resp = client.post(
            "/predict",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "unavailable"
        assert body["classification"] == "UNAVAILABLE"
        # No fabricated prediction
        assert body["spoof_probability"] is None

    def test_predict_unsupported_format_returns_400(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )
        assert resp.status_code == 400

    def test_predict_logs_incident(self, client):
        import io, scipy.io.wavfile as wavfile
        data = (np.random.randn(16000) * 1000).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, 16000, data)

        client.post(
            "/predict",
            files={"file": ("test.wav", buf.getvalue(), "audio/wav")},
        )

        resp = client.get("/incidents")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_timeline_endpoint(self, client):
        resp = client.get("/incidents/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert "timeline" in data

    def test_reload_model_endpoint(self, client):
        resp = client.post("/reload-model")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "reloaded"


# ──────────────────────────────────────────────────────────────────────────────
# Test: Evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluationMetrics:
    """Tests for the evaluation metric functions."""

    def test_eer_perfect_separation(self):
        from evaluation.evaluator import compute_eer
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        eer, thr = compute_eer(labels, scores)
        assert eer < 0.01  # near-zero EER for perfect separation

    def test_eer_random_chance(self):
        from evaluation.evaluator import compute_eer
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, 1000)
        scores = rng.uniform(0, 1, 1000)
        eer, _ = compute_eer(labels, scores)
        assert 0.35 < eer < 0.65  # random ~ 0.5

    def test_compute_metrics_returns_all_fields(self):
        from evaluation.evaluator import compute_all_metrics
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 200)
        scores = rng.uniform(0, 1, 200)
        m = compute_all_metrics(labels, scores, threshold=0.5)
        required = ['roc_auc', 'eer', 'accuracy', 'f1', 'precision',
                    'recall', 'fpr', 'fnr', 'confusion_matrix',
                    'threshold_used', 'threshold_source']
        for field in required:
            assert field in m, f"Missing metric: {field}"

    def test_threshold_source_is_validation(self):
        from evaluation.evaluator import compute_all_metrics
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 100)
        scores = rng.uniform(0, 1, 100)
        m = compute_all_metrics(labels, scores, threshold=0.45)
        assert m["threshold_source"] == "validation_eer"
        assert m["threshold_used"] == pytest.approx(0.45)

    def test_per_generator_report(self):
        from evaluation.evaluator import per_generator_report
        records = [
            {"attack_type": "A01", "label": 1, "label_str": "spoof"},
            {"attack_type": "A01", "label": 1, "label_str": "spoof"},
            {"attack_type": "bonafide", "label": 0, "label_str": "bonafide"},
        ]
        labels = np.array([1, 1, 0])
        scores = np.array([0.8, 0.9, 0.2])
        result = per_generator_report(records, labels, scores, threshold=0.5)
        assert "A01" in result
        assert "bonafide" in result
        assert result["A01"]["n_spoof"] == 2
        assert result["A01"]["spoof_recall"] == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Test: Dataset utilities
# ──────────────────────────────────────────────────────────────────────────────

class TestDatasetUtilities:
    """Tests for dataset indexing and statistics (no actual audio needed)."""

    def test_dataset_stats(self):
        from training.dataset import dataset_stats
        records = [
            {"speaker_id": "S01", "label": 0, "attack_type": "bonafide"},
            {"speaker_id": "S01", "label": 1, "attack_type": "A01"},
            {"speaker_id": "S02", "label": 1, "attack_type": "A02"},
        ]
        stats = dataset_stats(records)
        assert stats["total"] == 3
        assert stats["bonafide"] == 1
        assert stats["spoof"] == 2
        assert stats["unique_speakers"] == 2
        assert "bonafide" in stats["attack_types"]

    def test_class_weights_balanced(self):
        """For equal class split, weights should be equal."""
        from training.dataset import ASVspoof5Dataset
        # Create a tiny mock dataset
        records = [
            {"speaker_id": "S01", "utterance_id": "U01", "gender": "M",
             "codec": None, "attack_type": "bonafide", "label": 0,
             "label_str": "bonafide", "path": "/fake/1.flac"},
            {"speaker_id": "S02", "utterance_id": "U02", "gender": "F",
             "codec": None, "attack_type": "A01", "label": 1,
             "label_str": "spoof", "path": "/fake/2.flac"},
        ]
        ds = ASVspoof5Dataset.__new__(ASVspoof5Dataset)
        ds.records = records
        cw = ds.class_weights()
        assert cw.shape == (2,)
        assert cw[0] == pytest.approx(cw[1], abs=0.01)

    def test_speaker_disjoint_split(self):
        from training.dataset import _speaker_disjoint_split
        records = [
            {"speaker_id": f"S{i:03d}", "label": i % 2} for i in range(100)
        ]
        train, val = _speaker_disjoint_split(records, 0.2, seed=42)
        train_speakers = {r["speaker_id"] for r in train}
        val_speakers   = {r["speaker_id"] for r in val}
        assert len(train_speakers & val_speakers) == 0, \
            "Speaker leak between train and validation!"
        assert len(train) + len(val) == 100

    def test_collate_fn(self):
        from training.dataset import collate_fn
        batch = [
            (torch.randn(64000), 0, {"speaker_id": "S1"}),
            (torch.randn(64000), 1, {"speaker_id": "S2"}),
        ]
        wavs, labels, metas = collate_fn(batch)
        assert wavs.shape == (2, 64000)
        assert labels.shape == (2,)
        assert labels.dtype == torch.float32
        assert len(metas) == 2

    def test_labels_0_and_1_only(self):
        """All labels should be 0 (bonafide) or 1 (spoof)."""
        from training.dataset import _parse_protocol
        # Simulate parsing a few lines
        import tempfile
        lines = [
            "T_4850 T_0000000000 F - - - AC3 A05 spoof -",
            "T_3734 T_0000000011 F - - - - bonafide bonafide -",
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("\n".join(lines))
            fpath = f.name
        try:
            records = _parse_protocol(Path(fpath))
            labels = {r["label"] for r in records}
            assert labels.issubset({0, 1}), f"Unexpected labels: {labels}"
            spoof_records = [r for r in records if r["label_str"] == "spoof"]
            bf_records    = [r for r in records if r["label_str"] == "bonafide"]
            assert all(r["label"] == 1 for r in spoof_records)
            assert all(r["label"] == 0 for r in bf_records)
        finally:
            Path(fpath).unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Test: No hardcoded secrets / path traversal safety
# ──────────────────────────────────────────────────────────────────────────────

class TestSecurity:
    """Basic security checks."""

    def test_no_hardcoded_api_keys(self):
        """Check that source files don't contain obvious API key patterns."""
        import re
        pattern = re.compile(r'(?i)(api[_-]?key|secret[_-]?key|openai|anthropic)\s*=\s*["\'][^"\']{10,}["\']')
        src_dirs = ["model", "training", "inference", "backend", "evaluation"]
        for d in src_dirs:
            for f in (PROJECT_ROOT / d).glob("*.py"):
                text = f.read_text()
                m = pattern.search(text)
                assert m is None, f"Possible hardcoded secret in {f}: {m.group()}"

    def test_no_external_inference_imports(self):
        """Check that no forbidden model libraries are imported."""
        import ast
        forbidden_modules = {
            'transformers', 'openai', 'anthropic', 'cohere',
            'huggingface_hub', 'speechbrain',
        }
        # Only allow transformers if it's in a comment
        src_dirs = ["model", "training", "inference", "backend"]
        for d in src_dirs:
            for f in (PROJECT_ROOT / d).glob("*.py"):
                try:
                    tree = ast.parse(f.read_text())
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.Import):
                            names = [alias.name.split('.')[0] for alias in node.names]
                        else:
                            names = [node.module.split('.')[0]] if node.module else []
                        for name in names:
                            assert name not in forbidden_modules, \
                                f"Forbidden import '{name}' in {f}"

    def test_incident_logger_sanitizes_path_in_filename(self, tmp_path):
        """Path traversal attempts in filename should be sanitized."""
        from backend.incident_logger import IncidentLogger
        logger_inst = IncidentLogger(db_path=tmp_path / "sec_test.db")
        result = {
            "classification": "SPOOF", "spoof_probability": 0.9,
            "bona_fide_probability": 0.1, "confidence": 0.8,
            "risk_score": 0.85, "threat_level": "HIGH",
            "decision_threshold": 0.5, "recommended_action": "BLOCK",
            "latency_ms": 10.0, "device": "cpu", "model_version": "1.0",
            "inference_mode": "asvspoof5", "status": "ok", "error": None,
        }
        logger_inst.log_incident(result, filename="../../../etc/shadow")
        incidents = logger_inst.get_incidents(limit=1)
        assert incidents[0]["filename"] == "shadow"


# ──────────────────────────────────────────────────────────────────────────────
# Test: Frontend build (smoke test)
# ──────────────────────────────────────────────────────────────────────────────

class TestFrontendBuild:
    """Verify the frontend build artifact exists."""

    def test_dist_index_exists(self):
        dist_html = PROJECT_ROOT / "frontend" / "dist" / "index.html"
        assert dist_html.exists(), (
            f"Frontend dist not found at {dist_html}. "
            "Run: cd frontend && npm run build"
        )

    def test_dist_has_js_asset(self):
        assets_dir = PROJECT_ROOT / "frontend" / "dist" / "assets"
        js_files = list(assets_dir.glob("*.js"))
        assert len(js_files) > 0, "No JS bundle found in frontend/dist/assets/"

    def test_dist_js_not_empty(self):
        assets_dir = PROJECT_ROOT / "frontend" / "dist" / "assets"
        js_files = list(assets_dir.glob("*.js"))
        for f in js_files:
            assert f.stat().st_size > 10_000, f"JS bundle suspiciously small: {f}"


# ──────────────────────────────────────────────────────────────────────────────
# Integration markers
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestASVspoof5Integration:
    """Integration tests that require actual ASVspoof5 dataset."""

    DATASET_DIR = Path("/mnt/d/Datasets/ASVspoof5")

    def test_dataset_accessible(self):
        assert self.DATASET_DIR.exists()
        assert (self.DATASET_DIR / "protocols" / "ASVspoof5.train.tsv").exists()

    def test_can_build_index(self):
        from training.dataset import build_index, dataset_stats
        train, val = build_index(self.DATASET_DIR, split="train", seed=42)
        assert len(train) > 10000
        assert len(val) > 1000
        t = dataset_stats(train)
        assert t["bonafide"] > 0
        assert t["spoof"] > 0

    def test_no_speaker_overlap(self):
        from training.dataset import build_index
        train, val = build_index(self.DATASET_DIR, split="train", seed=42)
        train_sp = {r["speaker_id"] for r in train}
        val_sp   = {r["speaker_id"] for r in val}
        assert len(train_sp & val_sp) == 0

    def test_can_load_sample(self):
        from training.dataset import build_index, ASVspoof5Dataset
        _, val = build_index(self.DATASET_DIR, split="train", seed=42)
        ds = ASVspoof5Dataset(val[:10], augment=False)
        wav, label, meta = ds[0]
        assert wav.shape == (64000,)
        assert label in (0, 1)
        assert "speaker_id" in meta
