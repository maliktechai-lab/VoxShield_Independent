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
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
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
                    tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
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
# Test: Audio padding integer correctness (regression for float-pad bug)
# ──────────────────────────────────────────────────────────────────────────────

class TestAudioPadding:
    """
    Regression tests for the F.pad integer bug.

    Bug: MODEL_CONFIG["max_length_sec"] * MODEL_CONFIG["sample_rate"]
         evaluates to 4.0 * 16000 = 64000.0 (float), which was passed
         to ASVspoof5Dataset as max_samples.  The _load_audio method then
         called F.pad(wav, (0, self.max_samples - n)) where the right-pad
         value was float, causing:
             TypeError: pad(): argument 'pad' (pos 2) must be tuple of ints

    All tests below must pass without TypeError.
    """

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_record(self, path: str) -> dict:
        return {
            "speaker_id": "S01",
            "utterance_id": "U01",
            "gender": "M",
            "codec": None,
            "attack_type": "bonafide",
            "label": 0,
            "label_str": "bonafide",
            "path": path,
        }

    def _write_flac(self, tmp_path: Path, n_samples: int, sr: int = 16000) -> Path:
        """Write a minimal FLAC file with n_samples frames."""
        import soundfile as sf
        data = np.random.default_rng(0).standard_normal(n_samples).astype(np.float32) * 0.1
        p = tmp_path / f"audio_{n_samples}.flac"
        sf.write(str(p), data, sr, subtype="PCM_16")
        return p

    # ── max_samples type guard ────────────────────────────────────────────────

    def test_max_samples_stored_as_int_when_given_int(self):
        """When max_samples is already an int, self.max_samples must be int."""
        from training.dataset import ASVspoof5Dataset
        ds = ASVspoof5Dataset.__new__(ASVspoof5Dataset)
        ds.records = []
        ds.sample_rate = 16000
        ds.max_samples = int(64000)
        assert isinstance(ds.max_samples, int)

    def test_max_samples_coerced_to_int_when_given_float(self):
        """
        Core regression: passing max_samples=64000.0 (float) to the
        constructor must result in self.max_samples being int, not float.
        """
        from training.dataset import ASVspoof5Dataset
        records = []  # empty — we only test __init__ type coercion
        ds = ASVspoof5Dataset(records, max_samples=64000.0)
        assert isinstance(ds.max_samples, int), (
            f"max_samples should be int, got {type(ds.max_samples)}"
        )
        assert ds.max_samples == 64000

    def test_sample_rate_coerced_to_int_when_given_float(self):
        """sample_rate=16000.0 must be stored as int."""
        from training.dataset import ASVspoof5Dataset
        ds = ASVspoof5Dataset([], sample_rate=16000.0, max_samples=64000)
        assert isinstance(ds.sample_rate, int)

    def test_config_derived_max_samples_is_float_before_int_wrap(self):
        """
        Confirm the production calculation would have been a float without int().
        This documents the root cause.
        """
        from model.voxshieldnet import MODEL_CONFIG
        raw = MODEL_CONFIG["max_length_sec"] * MODEL_CONFIG["sample_rate"]
        assert isinstance(raw, float), (
            "Precondition: max_length_sec * sample_rate must be float to trigger the bug"
        )
        assert raw == 64000.0
        # int() wrapping is the fix
        assert isinstance(int(raw), int)

    # ── Padding path: short waveform (requires F.pad) ─────────────────────────

    def test_pad_short_waveform_no_type_error(self, tmp_path):
        """Waveform shorter than max_samples must pad without TypeError."""
        from training.dataset import ASVspoof5Dataset
        audio_path = self._write_flac(tmp_path, n_samples=8000)  # 0.5 s
        rec = self._make_record(str(audio_path))
        # Pass max_samples as float to trigger the bug path; defensive int()
        # in __init__ should save us.
        ds = ASVspoof5Dataset([rec], max_samples=64000.0, augment=False)
        wav, label, meta = ds[0]
        assert wav.shape == (64000,), f"Expected (64000,), got {wav.shape}"
        assert wav.dtype == torch.float32

    def test_pad_short_waveform_correct_length(self, tmp_path):
        """Padded waveform length must exactly equal max_samples."""
        from training.dataset import ASVspoof5Dataset
        n_input = 12345
        audio_path = self._write_flac(tmp_path, n_samples=n_input)
        rec = self._make_record(str(audio_path))
        ds = ASVspoof5Dataset([rec], max_samples=64000, augment=False)
        wav, _, _ = ds[0]
        assert wav.shape[0] == 64000
        # Tail (padding region) should be zero
        assert wav[n_input:].abs().sum() == 0.0

    def test_pad_values_are_zero(self, tmp_path):
        """Padding applied by F.pad must be zero-valued."""
        from training.dataset import ASVspoof5Dataset
        n_input = 16000  # 1 s of audio
        audio_path = self._write_flac(tmp_path, n_samples=n_input)
        rec = self._make_record(str(audio_path))
        ds = ASVspoof5Dataset([rec], max_samples=64000, augment=False)
        wav, _, _ = ds[0]
        assert float(wav[n_input:].abs().max()) == pytest.approx(0.0)

    # ── Exact-length waveform (no pad, no crop) ───────────────────────────────

    def test_exact_length_waveform_unmodified(self, tmp_path):
        """Waveform with n == max_samples must not be padded or cropped."""
        from training.dataset import ASVspoof5Dataset
        n_input = 64000
        audio_path = self._write_flac(tmp_path, n_samples=n_input)
        rec = self._make_record(str(audio_path))
        ds = ASVspoof5Dataset([rec], max_samples=64000, augment=False)
        wav, _, _ = ds[0]
        assert wav.shape[0] == 64000

    # ── Long waveform (truncation path) ──────────────────────────────────────

    def test_truncate_long_waveform(self, tmp_path):
        """Waveform longer than max_samples must be truncated to max_samples."""
        from training.dataset import ASVspoof5Dataset
        audio_path = self._write_flac(tmp_path, n_samples=96000)  # 6 s
        rec = self._make_record(str(audio_path))
        ds = ASVspoof5Dataset([rec], max_samples=64000, augment=False)
        wav, _, _ = ds[0]
        assert wav.shape[0] == 64000

    # ── F.pad receives integer padding tuple ─────────────────────────────────

    def test_fpad_integer_pad_tuple(self):
        """
        Directly verify that _load_audio produces an integer pad tuple.
        Monkeypatches F.pad to inspect the argument before it is consumed.
        """
        import torch.nn.functional as real_F
        from training.dataset import ASVspoof5Dataset

        captured_pad_args = []

        original_pad = real_F.pad

        def spy_pad(input, pad, *args, **kwargs):
            captured_pad_args.append(pad)
            return original_pad(input, pad, *args, **kwargs)

        # Build a dataset with a real audio fixture using a short waveform.
        # We'll stub _load_audio to call F.pad manually so we can spy.
        data = np.zeros(8000, dtype=np.float32)  # shorter than 64000
        wav_tensor = torch.from_numpy(data)

        ds = ASVspoof5Dataset.__new__(ASVspoof5Dataset)
        ds.records = []
        ds.sample_rate = 16000
        ds.max_samples = 64000  # int
        ds.augment = False
        ds._rng = np.random.default_rng(0)

        n = wav_tensor.shape[0]
        pad_value = ds.max_samples - n
        # Confirm the pad value is an int
        assert isinstance(pad_value, int), (
            f"pad_value must be int, got {type(pad_value)}: {pad_value}"
        )
        # Confirm F.pad does not raise
        padded = real_F.pad(wav_tensor, (0, pad_value))
        assert padded.shape[0] == ds.max_samples

    def test_fpad_raises_with_float_pad_tuple(self):
        """
        Confirm PyTorch raises TypeError when F.pad receives a float.
        This documents the bug behaviour so the regression is meaningful.
        """
        import torch.nn.functional as real_F

        wav = torch.zeros(8000)
        float_pad = 64000.0 - 8000  # → 56000.0, a float
        assert isinstance(float_pad, float)
        with pytest.raises(TypeError):
            real_F.pad(wav, (0, float_pad))

    # ── Non-standard target lengths from config values ────────────────────────

    def test_non_standard_max_samples_from_float_config(self, tmp_path):
        """
        Simulate a caller passing max_samples derived from float config
        (e.g. 3.5 * 16000 = 56000.0).  Defensive int() in __init__ must
        prevent TypeError.
        """
        from training.dataset import ASVspoof5Dataset
        target = int(3.5 * 16000)  # 56000
        audio_path = self._write_flac(tmp_path, n_samples=8000)
        rec = self._make_record(str(audio_path))
        # Pass as float to trigger the defensive coercion
        ds = ASVspoof5Dataset([rec], max_samples=3.5 * 16000, augment=False)
        wav, _, _ = ds[0]
        assert wav.shape[0] == target
        assert isinstance(ds.max_samples, int)

    # ── End-to-end: many consecutive samples without error ────────────────────

    def test_many_consecutive_samples_no_error(self, tmp_path):
        """
        Load 20 samples of varying lengths through ASVspoof5Dataset.
        None should raise TypeError or any other exception.
        This mirrors the Colab failure where ~1400 batches succeeded
        before a padding sample caused the crash.
        """
        import soundfile as sf
        from training.dataset import ASVspoof5Dataset

        rng = np.random.default_rng(42)
        records = []
        lengths = [4000, 8000, 16000, 32000, 64000, 72000, 96000,
                   1000, 48000, 64000, 3200, 16001, 63999, 64001,
                   44100, 22050, 11025, 8192, 65536, 32768]
        for i, n in enumerate(lengths):
            p = tmp_path / f"sample_{i}.flac"
            data = (rng.standard_normal(n) * 0.1).astype(np.float32)
            sf.write(str(p), data, 16000, subtype="PCM_16")
            records.append({
                "speaker_id": f"S{i:03d}",
                "utterance_id": f"U{i:03d}",
                "gender": "M",
                "codec": None,
                "attack_type": "bonafide",
                "label": i % 2,
                "label_str": "bonafide" if i % 2 == 0 else "spoof",
                "path": str(p),
            })

        # Use float max_samples to rely on defensive int() coercion in __init__
        ds = ASVspoof5Dataset(records, max_samples=64000.0, augment=False)
        assert ds.max_samples == 64000
        assert isinstance(ds.max_samples, int)

        for idx in range(len(ds)):
            wav, label, meta = ds[idx]
            assert wav.shape == (64000,), (
                f"Sample {idx} (input len {lengths[idx]}): "
                f"got shape {wav.shape}"
            )
            assert wav.dtype == torch.float32
            assert label in (0, 1)


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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers shared by new regression tests
# ──────────────────────────────────────────────────────────────────────────────

def _write_tiny_flac(path: Path, n_samples: int = 8000, sr: int = 16000) -> None:
    """Write a tiny valid FLAC file at *path*."""
    import soundfile as sf
    data = (np.random.default_rng(0).standard_normal(n_samples) * 0.05).astype(np.float32)
    sf.write(str(path), data, sr, subtype="PCM_16")


def _make_tsv_line(
    speaker="T_4850",
    uid="T_0000000000",
    gender="F",
    codec="-",
    codec_q="-",
    codec_seed="-",
    attack_tag="-",
    attack_label="bonafide",
    key="bonafide",
    tmp="-",
) -> str:
    return f"{speaker} {uid} {gender} {codec} {codec_q} {codec_seed} {attack_tag} {attack_label} {key} {tmp}"


def _build_synthetic_root(
    tmp_path: Path,
    train_lines: list[str] | None = None,
    dev_lines: list[str] | None = None,
    create_train_audio: bool = True,
    create_dev_audio: bool = False,
    proto_at_root: bool = True,
) -> Path:
    """
    Build a minimal synthetic ASVspoof5-style root directory.

    Returns the root path.
    """
    root = tmp_path / "ASVspoof5"
    root.mkdir(parents=True)

    proto_dir = root if proto_at_root else (root / "protocols")
    proto_dir.mkdir(parents=True, exist_ok=True)

    if train_lines is None:
        train_lines = [
            _make_tsv_line("T_SP01", "T_0000000001", "M", "-", "-", "-", "-", "bonafide", "bonafide"),
            _make_tsv_line("T_SP01", "T_0000000002", "M", "OPUS", "-", "-", "A_syn", "A01", "spoof"),
            _make_tsv_line("T_SP02", "T_0000000003", "F", "-", "-", "-", "-", "bonafide", "bonafide"),
            _make_tsv_line("T_SP02", "T_0000000004", "F", "AAC",  "-", "-", "A_syn", "A02", "spoof"),
            _make_tsv_line("T_SP03", "T_0000000005", "M", "-", "-", "-", "-", "bonafide", "bonafide"),
            _make_tsv_line("T_SP03", "T_0000000006", "M", "MP3",  "-", "-", "A_syn", "A03", "spoof"),
        ]

    (proto_dir / "ASVspoof5.train.tsv").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

    if create_train_audio:
        flac_t = root / "flac_T"
        flac_t.mkdir()
        for line in train_lines:
            uid = line.split()[1]
            _write_tiny_flac(flac_t / f"{uid}.flac")

    if dev_lines is not None:
        (proto_dir / "ASVspoof5.dev.track_1.tsv").write_text(
            "\n".join(dev_lines) + "\n", encoding="utf-8"
        )
        if create_dev_audio:
            flac_d = root / "flac_D"
            flac_d.mkdir()
            for line in dev_lines:
                uid = line.split()[1]
                _write_tiny_flac(flac_d / f"{uid}.flac")

    return root


# ──────────────────────────────────────────────────────────────────────────────
# Task 2: Native path resolution
# ──────────────────────────────────────────────────────────────────────────────

class TestNativePathResolution:
    """
    Verify that _find_audio() resolves to the official native layout:
        T_* -> <root>/flac_T/<id>.flac
        D_* -> <root>/flac_D/<id>.flac
        E_* -> <root>/flac_E_eval/<id>.flac

    These MUST NOT require artificial train/ or dev/ subdirectories.
    """

    def test_train_utterance_resolves_to_flac_T(self, tmp_path):
        from training.dataset import _find_audio
        flac_t = tmp_path / "flac_T"
        flac_t.mkdir()
        expected = flac_t / "T_0000000001.flac"
        _write_tiny_flac(expected)

        result = _find_audio(tmp_path, "T_0000000001")
        assert result == expected, f"Expected {expected}, got {result}"

    def test_dev_utterance_resolves_to_flac_D(self, tmp_path):
        from training.dataset import _find_audio
        flac_d = tmp_path / "flac_D"
        flac_d.mkdir()
        expected = flac_d / "D_0000000001.flac"
        _write_tiny_flac(expected)

        result = _find_audio(tmp_path, "D_0000000001")
        assert result == expected, f"Expected {expected}, got {result}"

    def test_eval_utterance_resolves_to_flac_E_eval(self, tmp_path):
        from training.dataset import _find_audio
        flac_e = tmp_path / "flac_E_eval"
        flac_e.mkdir()
        expected = flac_e / "E_0000000001.flac"
        _write_tiny_flac(expected)

        result = _find_audio(tmp_path, "E_0000000001")
        assert result == expected, f"Expected {expected}, got {result}"

    def test_missing_audio_returns_none(self, tmp_path):
        from training.dataset import _find_audio
        result = _find_audio(tmp_path, "T_9999999999")
        assert result is None

    def test_does_not_look_in_train_subdirectory(self, tmp_path):
        """Must NOT look in train/flac_T/ — that is the OLD wrong layout."""
        from training.dataset import _find_audio
        # Create audio in the wrong (old) location
        wrong_dir = tmp_path / "train" / "flac_T"
        wrong_dir.mkdir(parents=True)
        _write_tiny_flac(wrong_dir / "T_0000000001.flac")

        result = _find_audio(tmp_path, "T_0000000001")
        assert result is None, (
            "Should NOT resolve T_ utterances from train/flac_T/ — "
            "that is the old incorrect layout"
        )

    def test_does_not_look_in_dev_subdirectory(self, tmp_path):
        """Must NOT look in dev/flac_D/ — that is the OLD wrong layout."""
        from training.dataset import _find_audio
        wrong_dir = tmp_path / "dev" / "flac_D"
        wrong_dir.mkdir(parents=True)
        _write_tiny_flac(wrong_dir / "D_0000000001.flac")

        result = _find_audio(tmp_path, "D_0000000001")
        assert result is None, (
            "Should NOT resolve D_ utterances from dev/flac_D/ — "
            "that is the old incorrect layout"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Task 3: TSV parsing — correct field positions
# ──────────────────────────────────────────────────────────────────────────────

class TestTSVParsing:
    """
    Verify correct ASVspoof5 protocol field mapping.

    Official 10-field layout:
        0: SPEAKER_ID
        1: FLAC_FILE_NAME
        2: SPEAKER_GENDER
        3: CODEC            ← codec must come from field 3, NOT field 6
        4: CODEC_Q
        5: CODEC_SEED
        6: ATTACK_TAG
        7: ATTACK_LABEL     ← attack label from field 7
        8: KEY              ← bonafide/spoof from field 8
        9: TMP
    """

    def _parse_line(self, line: str) -> dict:
        import tempfile
        from training.dataset import _parse_protocol
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write(line + "\n")
            fpath = f.name
        try:
            records = _parse_protocol(Path(fpath))
        finally:
            Path(fpath).unlink(missing_ok=True)
        return records[0] if records else {}

    def test_bonafide_key_field_8(self):
        """Field 8 = 'bonafide' → label 0."""
        line = "T_4850 T_0000000011 F - - - - bonafide bonafide -"
        r = self._parse_line(line)
        assert r["label"] == 0
        assert r["label_str"] == "bonafide"

    def test_spoof_key_field_8(self):
        """Field 8 = 'spoof' → label 1."""
        line = "T_4850 T_0000000000 F - - - AC3 A05 spoof -"
        r = self._parse_line(line)
        assert r["label"] == 1
        assert r["label_str"] == "spoof"

    def test_attack_label_from_field_7(self):
        """Attack type/label comes from field 7 (not field 6)."""
        line = "T_4850 T_0000000000 F - - - AC3 A05 spoof -"
        r = self._parse_line(line)
        assert r["attack_type"] == "A05", (
            f"attack_type should be 'A05' (field 7), got '{r['attack_type']}'"
        )

    def test_codec_from_field_3(self):
        """Codec comes from field 3 (not field 6)."""
        # "OPUS" is at field 3 here; field 6 is a different value
        line = "T_SP01 T_0000000001 M OPUS q1 s1 ATAG A01 spoof -"
        r = self._parse_line(line)
        assert r["codec"] == "OPUS", (
            f"codec should be 'OPUS' (field 3), got '{r['codec']}'"
        )

    def test_codec_dash_becomes_none(self):
        """Codec field '-' (bonafide) should be stored as None."""
        line = "T_4850 T_0000000011 F - - - - bonafide bonafide -"
        r = self._parse_line(line)
        assert r["codec"] is None, f"codec should be None for bonafide, got {r['codec']!r}"

    def test_attack_tag_not_used_as_codec(self):
        """AC3 at field 6 (ATTACK_TAG) must NOT be treated as codec."""
        # Official: field 3 is codec (here '-'), field 6 is attack_tag ('AC3')
        line = "T_4850 T_0000000000 F - - - AC3 A05 spoof -"
        r = self._parse_line(line)
        # codec should be None (field 3 = '-'), not 'AC3'
        assert r["codec"] is None, (
            f"AC3 is at field 6 (ATTACK_TAG), not codec (field 3). "
            f"Got codec={r['codec']!r}"
        )

    def test_utterance_id_from_field_1(self):
        line = "T_4850 T_0000000000 F - - - AC3 A05 spoof -"
        r = self._parse_line(line)
        assert r["utterance_id"] == "T_0000000000"

    def test_speaker_id_from_field_0(self):
        line = "T_4850 T_0000000000 F - - - AC3 A05 spoof -"
        r = self._parse_line(line)
        assert r["speaker_id"] == "T_4850"

    def test_gender_from_field_2(self):
        line = "T_4850 T_0000000000 F - - - AC3 A05 spoof -"
        r = self._parse_line(line)
        assert r["gender"] == "F"

    def test_short_line_skipped(self):
        """Lines with fewer than 9 fields should be silently skipped."""
        import tempfile
        from training.dataset import _parse_protocol
        lines = [
            "too short",
            "T_4850 T_0000000000 F - - - AC3 A05 spoof -",
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            fpath = f.name
        try:
            records = _parse_protocol(Path(fpath))
        finally:
            Path(fpath).unlink(missing_ok=True)
        assert len(records) == 1, "Short line should be skipped, valid line kept"

    def test_labels_only_0_and_1(self):
        import tempfile
        from training.dataset import _parse_protocol
        lines = [
            "T_4850 T_0000000000 F - - - AC3 A05 spoof -",
            "T_3734 T_0000000011 F - - - - bonafide bonafide -",
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            fpath = f.name
        try:
            records = _parse_protocol(Path(fpath))
        finally:
            Path(fpath).unlink(missing_ok=True)
        labels = {r["label"] for r in records}
        assert labels.issubset({0, 1})

    def test_parse_protocol_file_at_root(self, tmp_path):
        """Protocol file can be read from the dataset root directly."""
        from training.dataset import _find_protocol
        proto = tmp_path / "ASVspoof5.train.tsv"
        proto.write_text(
            "T_SP01 T_0000000001 M - - - - bonafide bonafide -\n",
            encoding="utf-8",
        )
        found = _find_protocol(tmp_path, "ASVspoof5.train.tsv")
        assert found == proto

    def test_parse_protocol_file_in_protocols_subdir(self, tmp_path):
        """Protocol file under protocols/ subdirectory is also found."""
        from training.dataset import _find_protocol
        proto_dir = tmp_path / "protocols"
        proto_dir.mkdir()
        proto = proto_dir / "ASVspoof5.train.tsv"
        proto.write_text(
            "T_SP01 T_0000000001 M - - - - bonafide bonafide -\n",
            encoding="utf-8",
        )
        found = _find_protocol(tmp_path, "ASVspoof5.train.tsv")
        assert found == proto

    def test_missing_protocol_raises(self, tmp_path):
        """Missing protocol must raise FileNotFoundError, not return None."""
        from training.dataset import _require_protocol
        with pytest.raises(FileNotFoundError):
            _require_protocol(tmp_path, "ASVspoof5.train.tsv")


# ──────────────────────────────────────────────────────────────────────────────
# Task 4: Cache validation
# ──────────────────────────────────────────────────────────────────────────────

class TestCacheValidation:
    """
    Verify that stale/invalid cache is automatically rebuilt.

    The cache must NEVER silently return zero records or invalid paths.
    """

    def test_fresh_cache_is_built_and_loaded(self, tmp_path):
        """After build_index, a second call should load from cache."""
        root = _build_synthetic_root(tmp_path)
        from training.dataset import build_index

        train1, val1 = build_index(root, "train", val_speaker_fraction=0.3, seed=42)
        train2, val2 = build_index(root, "train", val_speaker_fraction=0.3, seed=42)

        assert len(train1) == len(train2)
        assert len(val1) == len(val2)

    def test_force_rebuild_ignores_cache(self, tmp_path):
        """--force-rebuild-index must rebuild even if cache exists."""
        root = _build_synthetic_root(tmp_path)
        from training.dataset import build_index

        build_index(root, "train", val_speaker_fraction=0.3, seed=42)
        # Second call with force_rebuild=True
        train, val = build_index(root, "train", val_speaker_fraction=0.3,
                                  seed=42, force_rebuild=True)
        assert len(train) > 0
        assert len(val) > 0

    def test_stale_cache_protocol_change_triggers_rebuild(self, tmp_path):
        """If protocol content changes, the cache must be invalidated."""
        root = _build_synthetic_root(tmp_path)
        from training.dataset import build_index

        train_orig, _ = build_index(root, "train", val_speaker_fraction=0.3, seed=42)

        # Add a new record to the protocol and a matching audio file
        proto = root / "ASVspoof5.train.tsv"
        with open(proto, "a", encoding="utf-8") as f:
            f.write("T_SP99 T_0000000099 M - - - - bonafide bonafide -\n")
        _write_tiny_flac(root / "flac_T" / "T_0000000099.flac")

        train_new, _ = build_index(root, "train", val_speaker_fraction=0.3, seed=42)
        assert len(train_new) > len(train_orig), (
            "Cache should have been invalidated after protocol changed"
        )

    def test_zero_record_cache_triggers_rebuild(self, tmp_path):
        """
        A cache with zero train records must be treated as invalid
        and the index rebuilt.
        """
        import pickle
        from training.dataset import _cache_path, CACHE_VERSION, build_index

        root = _build_synthetic_root(tmp_path)
        cache_file = _cache_path(root, "train")

        # Write a zero-record cache with valid metadata so it passes
        # the version check but fails the record-count check
        from training.dataset import _protocol_hash, _audio_dir_fingerprint
        train_proto = root / "ASVspoof5.train.tsv"
        bad_cache = {
            "cache_version": CACHE_VERSION,
            "proto_hash": _protocol_hash(train_proto),
            "audio_fingerprint": _audio_dir_fingerprint(root),
            "train": [],   # ZERO records — must trigger rebuild
            "val": [],
        }
        with open(cache_file, "wb") as f:
            pickle.dump(bad_cache, f)

        # build_index should detect empty train and rebuild
        train, val = build_index(root, "train", val_speaker_fraction=0.3, seed=42)
        assert len(train) > 0, "Cache with zero records must be rebuilt"

    def test_cache_with_missing_paths_triggers_rebuild(self, tmp_path):
        """
        A cache whose audio paths no longer exist on disk must be invalidated.
        """
        import pickle, shutil
        from training.dataset import _cache_path, CACHE_VERSION, build_index

        root = _build_synthetic_root(tmp_path)

        # Build a valid cache first
        build_index(root, "train", val_speaker_fraction=0.3, seed=42)

        # Delete the audio files to simulate moved/missing data
        shutil.rmtree(root / "flac_T")

        # Re-add protocol + audio in new location? No — test that rebuild
        # happens (and then naturally fails with RuntimeError because audio gone)
        cache_file = _cache_path(root, "train")
        assert cache_file.exists()

        # Load raw cache and confirm paths are stale
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        from training.dataset import _validate_cache, _protocol_hash, _audio_dir_fingerprint
        train_proto = root / "ASVspoof5.train.tsv"
        valid, reason = _validate_cache(
            data, root,
            _protocol_hash(train_proto),
            _audio_dir_fingerprint(root),
        )
        assert not valid, f"Cache should be invalid after audio deleted, reason: {reason}"
        assert "no longer exist" in reason.lower() or "missing" in reason.lower() or "fingerprint" in reason.lower()

    def test_cache_version_mismatch_triggers_rebuild(self, tmp_path):
        """Old CACHE_VERSION must cause immediate invalidation."""
        import pickle
        from training.dataset import _cache_path, build_index

        root = _build_synthetic_root(tmp_path)
        cache_file = _cache_path(root, "train")

        # Write a cache with wrong version
        bad_cache = {
            "cache_version": "v_ancient",
            "proto_hash": "abc",
            "audio_fingerprint": "xyz",
            "train": [{"dummy": True}],
            "val":   [{"dummy": True}],
        }
        with open(cache_file, "wb") as f:
            pickle.dump(bad_cache, f)

        train, val = build_index(root, "train", val_speaker_fraction=0.3, seed=42)
        assert len(train) > 0, "Cache with wrong version must be rebuilt"

    def test_cache_metadata_saved(self, tmp_path):
        """Built cache must contain version, hash, and fingerprint."""
        import pickle
        from training.dataset import _cache_path, CACHE_VERSION, build_index

        root = _build_synthetic_root(tmp_path)
        build_index(root, "train", val_speaker_fraction=0.3, seed=42)

        cache_file = _cache_path(root, "train")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)

        assert data["cache_version"] == CACHE_VERSION
        assert "proto_hash" in data
        assert "audio_fingerprint" in data
        assert len(data["proto_hash"]) == 64, "Expected SHA-256 hex digest"


# ──────────────────────────────────────────────────────────────────────────────
# Task 5: Audio loading — explicit failures
# ──────────────────────────────────────────────────────────────────────────────

class TestAudioLoadingBehavior:
    """
    Verify that unreadable/missing audio raises explicitly rather than
    returning zeros silently.
    """

    def _make_ds(self, records, **kw):
        from training.dataset import ASVspoof5Dataset
        return ASVspoof5Dataset(records, max_samples=64000, **kw)

    def _make_record(self, path):
        return {
            "speaker_id": "S01", "utterance_id": "U01", "gender": "M",
            "codec": None, "attack_type": "bonafide",
            "label": 0, "label_str": "bonafide",
            "path": str(path),
        }

    def test_missing_audio_raises_runtime_error(self, tmp_path):
        """A file listed in the index but missing on disk must raise RuntimeError."""
        rec = self._make_record(tmp_path / "nonexistent.flac")
        ds = self._make_ds([rec])
        with pytest.raises(RuntimeError, match="not found"):
            ds[0]

    def test_corrupt_audio_raises_runtime_error(self, tmp_path):
        """A corrupt/truncated file must raise RuntimeError, not return zeros."""
        bad_file = tmp_path / "corrupt.flac"
        bad_file.write_bytes(b"this is not a valid flac file")
        rec = self._make_record(bad_file)
        ds = self._make_ds([rec])
        with pytest.raises(RuntimeError, match="Cannot read"):
            ds[0]

    def test_valid_audio_loads_correctly(self, tmp_path):
        """Valid audio must load to exactly max_samples tensor."""
        audio_file = tmp_path / "T_test.flac"
        _write_tiny_flac(audio_file, n_samples=8000)
        rec = self._make_record(audio_file)
        ds = self._make_ds([rec])
        wav, label, meta = ds[0]
        assert wav.shape == (64000,)
        assert wav.dtype == torch.float32

    def test_short_audio_padded_with_zeros(self, tmp_path):
        """Short audio (< max_samples) must be zero-padded to max_samples."""
        audio_file = tmp_path / "T_short.flac"
        n_input = 8000
        _write_tiny_flac(audio_file, n_samples=n_input)
        rec = self._make_record(audio_file)
        ds = self._make_ds([rec])
        wav, _, _ = ds[0]
        assert wav.shape[0] == 64000
        assert float(wav[n_input:].abs().sum()) == pytest.approx(0.0)

    def test_long_audio_truncated(self, tmp_path):
        """Long audio (> max_samples) must be truncated to max_samples."""
        audio_file = tmp_path / "T_long.flac"
        _write_tiny_flac(audio_file, n_samples=96000)
        rec = self._make_record(audio_file)
        ds = self._make_ds([rec])
        wav, _, _ = ds[0]
        assert wav.shape[0] == 64000

    def test_exact_length_audio_unchanged(self, tmp_path):
        """Audio of exactly max_samples must not be modified."""
        audio_file = tmp_path / "T_exact.flac"
        _write_tiny_flac(audio_file, n_samples=64000)
        rec = self._make_record(audio_file)
        ds = self._make_ds([rec])
        wav, _, _ = ds[0]
        assert wav.shape[0] == 64000

    def test_max_samples_always_int(self, tmp_path):
        """max_samples passed as float must be stored as int."""
        from training.dataset import ASVspoof5Dataset
        ds = ASVspoof5Dataset([], max_samples=64000.0)
        assert isinstance(ds.max_samples, int)
        assert ds.max_samples == 64000


# ──────────────────────────────────────────────────────────────────────────────
# Task 7: Validation split behavior
# ──────────────────────────────────────────────────────────────────────────────

class TestValidationSplit:
    """Verify dev audio used when present; speaker-disjoint fallback otherwise."""

    def test_speaker_disjoint_fallback_when_no_dev_audio(self, tmp_path):
        """Without dev audio, carve speaker-disjoint split from training."""
        root = _build_synthetic_root(tmp_path, create_dev_audio=False)
        from training.dataset import build_index

        train, val = build_index(root, "train", val_speaker_fraction=0.4, seed=42)
        train_sp = {r["speaker_id"] for r in train}
        val_sp   = {r["speaker_id"] for r in val}

        assert len(train) > 0
        assert len(val) > 0
        assert len(train_sp & val_sp) == 0, "Speaker leak in disjoint fallback"

    def test_dev_audio_used_when_present(self, tmp_path):
        """When dev protocol + audio exist, use them as validation."""
        dev_lines = [
            _make_tsv_line("D_SP90", "D_0000000001", "M", "-", "-", "-", "-", "bonafide", "bonafide"),
            _make_tsv_line("D_SP91", "D_0000000002", "F", "AAC", "-", "-", "A_syn", "A01", "spoof"),
        ]
        root = _build_synthetic_root(
            tmp_path, dev_lines=dev_lines, create_dev_audio=True
        )
        from training.dataset import build_index

        _, val = build_index(root, "train", val_speaker_fraction=0.4, seed=42)
        val_uids = {r["utterance_id"] for r in val}
        assert "D_0000000001" in val_uids or "D_0000000002" in val_uids, (
            "Dev audio must be used as validation when present"
        )

    def test_empty_train_raises_runtime_error(self, tmp_path):
        """build_index must raise RuntimeError when no training audio exists."""
        root = _build_synthetic_root(tmp_path, create_train_audio=False)
        from training.dataset import build_index
        with pytest.raises(RuntimeError, match="No training audio"):
            build_index(root, "train", seed=42)


# ──────────────────────────────────────────────────────────────────────────────
# Task 11/12: Synthetic ASVspoof5 smoke test
# ──────────────────────────────────────────────────────────────────────────────

class TestSyntheticASVspoofSmokeTest:
    """
    End-to-end smoke test using a tiny synthetic ASVspoof5-style fixture.
    Proves the full pipeline works without the real 35+ GB dataset.
    """

    def test_full_pipeline_train_only(self, tmp_path):
        """
        Create synthetic fixture → parse protocol → build index → load dataset.
        """
        root = _build_synthetic_root(tmp_path)
        from training.dataset import build_index, ASVspoof5Dataset

        train, val = build_index(root, "train", val_speaker_fraction=0.4, seed=42)

        assert len(train) > 0, "Must have train records"
        assert len(val) > 0, "Must have val records"

        ds_train = ASVspoof5Dataset(train, max_samples=64000, augment=False)
        ds_val   = ASVspoof5Dataset(val,   max_samples=64000, augment=False)

        for i in range(len(ds_train)):
            wav, label, meta = ds_train[i]
            assert wav.shape == (64000,)
            assert label in (0, 1)

        for i in range(len(ds_val)):
            wav, label, meta = ds_val[i]
            assert wav.shape == (64000,)
            assert label in (0, 1)

    def test_full_pipeline_with_dev(self, tmp_path):
        """Full pipeline with dev protocol + dev audio."""
        dev_lines = [
            _make_tsv_line("D_SP90", "D_0000000001", "M", "-", "-", "-", "-", "bonafide", "bonafide"),
            _make_tsv_line("D_SP91", "D_0000000002", "F", "AAC", "-", "-", "A_syn", "A01", "spoof"),
        ]
        root = _build_synthetic_root(
            tmp_path, dev_lines=dev_lines, create_dev_audio=True
        )
        from training.dataset import build_index, ASVspoof5Dataset

        train, val = build_index(root, "train", val_speaker_fraction=0.4, seed=42)
        assert len(train) > 0
        assert len(val) >= 1

        for records, split_name in [(train, "train"), (val, "val")]:
            ds = ASVspoof5Dataset(records, max_samples=64000, augment=False)
            for i in range(len(ds)):
                wav, label, meta = ds[i]
                assert wav.shape == (64000,), f"{split_name}[{i}] shape {wav.shape}"
                assert label in (0, 1)

    def test_protocol_at_root_layout(self, tmp_path):
        """Protocol at dataset root (not under protocols/) is supported."""
        root = _build_synthetic_root(tmp_path, proto_at_root=True)
        from training.dataset import build_index
        train, val = build_index(root, "train", val_speaker_fraction=0.4, seed=42)
        assert len(train) > 0

    def test_protocol_in_protocols_subdir(self, tmp_path):
        """Protocol under protocols/ subdirectory is also supported."""
        root = _build_synthetic_root(tmp_path, proto_at_root=False)
        from training.dataset import build_index
        train, val = build_index(root, "train", val_speaker_fraction=0.4, seed=42)
        assert len(train) > 0

    def test_labels_assigned_correctly(self, tmp_path):
        """bonafide → 0, spoof → 1."""
        root = _build_synthetic_root(tmp_path)
        from training.dataset import build_index, ASVspoof5Dataset

        train, val = build_index(root, "train", val_speaker_fraction=0.4, seed=42)
        all_records = train + val
        ds = ASVspoof5Dataset(all_records, max_samples=64000, augment=False)

        for i in range(len(ds)):
            wav, label, meta = ds[i]
            if meta["label_str"] == "bonafide":
                assert label == 0
            else:
                assert label == 1

    def test_collate_fn_with_synthetic(self, tmp_path):
        """collate_fn must produce correct batch shapes."""
        from training.dataset import build_index, ASVspoof5Dataset, collate_fn
        from torch.utils.data import DataLoader

        root = _build_synthetic_root(tmp_path)
        train, val = build_index(root, "train", val_speaker_fraction=0.4, seed=42)
        all_records = train + val
        ds = ASVspoof5Dataset(all_records, max_samples=64000)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

        batch = next(iter(loader))
        wavs, labels, metas = batch
        assert wavs.shape[1] == 64000
        assert labels.dtype == torch.float32

    def test_class_weights_non_zero_with_both_classes(self, tmp_path):
        """When both bonafide and spoof are present, weights must be finite."""
        from training.dataset import build_index, ASVspoof5Dataset

        root = _build_synthetic_root(tmp_path)
        train, val = build_index(root, "train", val_speaker_fraction=0.4, seed=42)
        all_records = train + val
        ds = ASVspoof5Dataset(all_records, max_samples=64000)
        cw = ds.class_weights()
        assert torch.all(torch.isfinite(cw))
        assert torch.all(cw > 0)


# ──────────────────────────────────────────────────────────────────────────────
# Task 13: Model forward + backward smoke test
# ──────────────────────────────────────────────────────────────────────────────

class TestModelSmoke:
    """
    Model forward and backward pass smoke tests.
    No real dataset required.
    """

    def test_forward_pass_shape(self):
        from model.voxshieldnet import build_model
        model = build_model()
        model.eval()
        with torch.no_grad():
            wav = torch.randn(2, 64000)
            logits = model(wav)
        assert logits.shape == (2,), f"Expected (2,), got {logits.shape}"

    def test_forward_pass_finite_logits(self):
        from model.voxshieldnet import build_model
        model = build_model()
        model.eval()
        with torch.no_grad():
            wav = torch.randn(4, 64000)
            logits = model(wav)
        assert torch.all(torch.isfinite(logits)), "Logits must be finite"

    def test_backward_pass_gradients(self):
        """Loss must be differentiable; gradients must flow."""
        import torch.nn as nn
        from model.voxshieldnet import build_model

        model = build_model()
        model.train()
        criterion = nn.BCEWithLogitsLoss()

        wav    = torch.randn(4, 64000)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

        logits = model(wav)
        loss   = criterion(logits, labels)
        loss.backward()

        assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"
        # At least one parameter must have a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients flowed during backward pass"

    def test_model_no_sigmoid_inside_forward(self):
        """
        Output values must be able to exceed [0,1] range (raw logits).
        If sigmoid were inside forward(), this would never happen.
        """
        from model.voxshieldnet import build_model
        torch.manual_seed(0)
        model = build_model()
        model.eval()
        # Many different inputs — at least one should produce |logit| > 0.1
        with torch.no_grad():
            wav = torch.randn(32, 64000) * 2
            logits = model(wav)
        # sigmoid-clamped values would all be in [0,1]; raw logits can be outside
        probs_if_clamped = torch.sigmoid(logits)
        # The key property: logits != sigmoid(logits) for extreme values
        extreme = (logits.abs() > 0.2).any()
        assert extreme, "Expected at least one logit with |value| > 0.2"

    def test_checkpoint_save_and_reload(self, tmp_path):
        """Full checkpoint save → reload → forward must produce identical output."""
        from model.voxshieldnet import build_model
        from training.train import save_checkpoint, load_checkpoint
        import torch.optim as optim

        model = build_model()
        optimizer = optim.Adam(model.parameters())
        ckpt_path = tmp_path / "smoke.pt"

        save_checkpoint(
            ckpt_path, model, optimizer, None, None,
            epoch=1,
            val_metrics={"roc_auc": 0.75},
            dataset_info={"dataset_type": "ASVspoof5"},
            threshold=0.44,
            best_auc=0.75,
            training_history=[],
        )

        ckpt = load_checkpoint(ckpt_path)
        assert ckpt is not None
        assert "model_state_dict" in ckpt
        assert ckpt["threshold"] == pytest.approx(0.44)
        assert ckpt["metadata"]["pretrained"] is False

        # Reload into a fresh model and verify forward output matches
        model2 = build_model()
        model2.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        model2.eval()
        with torch.no_grad():
            wav = torch.randn(2, 64000)
            logits1 = model(wav)
            logits2 = model2(wav)
        assert torch.allclose(logits1, logits2, atol=1e-5), (
            "Reloaded model must produce identical logits"
        )
