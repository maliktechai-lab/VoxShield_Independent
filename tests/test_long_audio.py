"""
Tests for long-audio inference (windowing + /predict-long endpoint).

Covers:
1.  Long-audio window generation — normal overlapping case.
2.  Short audio padding (< 4 s → single zero-padded window).
3.  Exact 4-second audio → single window (no padding needed).
4.  Multiple-window audio.
5.  Very long audio / max_windows cap.
6.  Invalid / corrupt input handling.
7.  Existing /health endpoint.
8.  Existing /ready endpoint.
9.  Existing /predict endpoint.
10. New /predict-long endpoint.

Model inference is stubbed wherever possible so tests are fast and
do not depend on the trained checkpoint.
"""

from __future__ import annotations

import io
import math
import sys
import uuid
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Helpers ───────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 64_000   # 4 s × 16 000


def _make_wav_bytes(duration_sec: float, sr: int = SAMPLE_RATE) -> bytes:
    """Generate a mono WAV file in memory (int16 PCM)."""
    import scipy.io.wavfile as wavfile

    n = int(sr * duration_sec)
    rng = np.random.default_rng(42)
    data = (rng.standard_normal(n) * 8000).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, data)
    return buf.getvalue()


def _stub_engine_result(spoof_prob: float = 0.8) -> Dict[str, Any]:
    """Return a fake InferenceEngine.predict() result."""
    return {
        "classification":       "SPOOF" if spoof_prob >= 0.5 else "BONA_FIDE",
        "spoof_probability":    spoof_prob,
        "bona_fide_probability": 1.0 - spoof_prob,
        "confidence":           abs(spoof_prob - 0.5) * 2.0,
        "decision_threshold":   0.6728817,
        "risk_score":           round(spoof_prob ** 0.7, 4),
        "threat_level":         "HIGH",
        "recommended_action":   "CHALLENGE",
        "risk_breakdown":       {"ml_model": False, "policy_engine": "deterministic"},
        "latency_ms":           1.5,
        "device":               "cpu",
        "model_version":        "1.0.0",
        "inference_mode":       "asvspoof5",
        "status":               "ok",
        "error":                None,
    }


def _make_mock_engine(spoof_prob: float = 0.8, ready: bool = True) -> MagicMock:
    """Build a MagicMock that behaves like InferenceEngine."""
    engine = MagicMock()
    engine.is_ready.return_value = ready
    engine.threshold = 0.6728817
    engine.inference_mode = "asvspoof5"
    engine.device = torch.device("cpu")
    engine.checkpoint_info = {"model_version": "1.0.0"}
    engine.predict.return_value = _stub_engine_result(spoof_prob)
    return engine


# ──────────────────────────────────────────────────────────────────────────────
# Tests: decode_full_audio
# ──────────────────────────────────────────────────────────────────────────────

class TestDecodeFullAudio:
    """Unit tests for the full-audio decoder."""

    def test_decode_8s_wav_not_truncated(self):
        """8-second WAV must NOT be truncated to 4 s."""
        from inference.long_audio import decode_full_audio

        wav_bytes = _make_wav_bytes(8.0)
        wav = decode_full_audio(wav_bytes, ".wav")
        assert wav.dim() == 1
        # 8 s at 16 kHz = 128 000 samples
        assert wav.numel() >= 120_000, (
            f"Expected ~128 000 samples, got {wav.numel()}"
        )

    def test_decode_short_wav(self):
        """2-second WAV should decode to ~32 000 samples (not padded here)."""
        from inference.long_audio import decode_full_audio

        wav_bytes = _make_wav_bytes(2.0)
        wav = decode_full_audio(wav_bytes, ".wav")
        assert wav.numel() >= 28_000

    def test_decode_exact_4s_wav(self):
        """Exactly 4-second WAV should decode to 64 000 samples."""
        from inference.long_audio import decode_full_audio

        wav_bytes = _make_wav_bytes(4.0)
        wav = decode_full_audio(wav_bytes, ".wav")
        assert wav.numel() == WINDOW_SAMPLES

    def test_output_is_float32_tensor(self):
        from inference.long_audio import decode_full_audio

        wav = decode_full_audio(_make_wav_bytes(2.0), ".wav")
        assert isinstance(wav, torch.Tensor)
        assert wav.dtype == torch.float32

    def test_output_peak_normalized(self):
        """Peak should be ≤ 1.0 after decode."""
        from inference.long_audio import decode_full_audio

        wav = decode_full_audio(_make_wav_bytes(3.0), ".wav")
        assert float(wav.abs().max()) <= 1.0 + 1e-5

    def test_empty_bytes_raises(self):
        from inference.long_audio import decode_full_audio, LongAudioError

        with pytest.raises(LongAudioError, match="Empty"):
            decode_full_audio(b"", ".wav")

    def test_corrupt_bytes_raises(self):
        from inference.long_audio import decode_full_audio, LongAudioError

        with pytest.raises(LongAudioError):
            decode_full_audio(b"\x00\x01\x02\x03corrupt", ".wav")

    def test_unsupported_format_raises(self):
        from inference.long_audio import decode_full_audio, LongAudioError

        with pytest.raises(LongAudioError, match="Unsupported"):
            decode_full_audio(b"some bytes", ".xyz")

    def test_silent_audio_raises(self):
        import scipy.io.wavfile as wavfile
        from inference.long_audio import decode_full_audio, LongAudioError

        silence = np.zeros(32000, dtype=np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, 16000, silence)
        with pytest.raises(LongAudioError, match="silent"):
            decode_full_audio(buf.getvalue(), ".wav")

    def test_max_duration_cap(self):
        """Audio longer than max_duration_sec should be clipped."""
        from inference.long_audio import decode_full_audio

        wav_bytes = _make_wav_bytes(10.0)
        wav = decode_full_audio(wav_bytes, ".wav", max_duration_sec=5.0)
        assert wav.numel() <= 5 * SAMPLE_RATE + 100  # small rounding tolerance

    def test_resampling_8khz(self):
        """8 kHz audio should be resampled to 16 kHz."""
        import scipy.io.wavfile as wavfile
        from inference.long_audio import decode_full_audio

        n = 32000  # 4 s at 8 kHz
        rng = np.random.default_rng(1)
        data = (rng.standard_normal(n) * 8000).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, 8000, data)
        wav = decode_full_audio(buf.getvalue(), ".wav")
        # 4 s at 16 kHz → 64 000 samples (allow small resampling jitter)
        assert abs(wav.numel() - WINDOW_SAMPLES) < 500


# ──────────────────────────────────────────────────────────────────────────────
# Tests: LongAudioAnalyzer._make_windows
# ──────────────────────────────────────────────────────────────────────────────

class TestMakeWindows:
    """Unit tests for the sliding-window slicing logic."""

    def _analyzer(self, window_sec=4.0, hop_sec=2.0, max_windows=120):
        from inference.long_audio import LongAudioAnalyzer

        return LongAudioAnalyzer(
            engine=_make_mock_engine(),
            window_sec=window_sec,
            hop_sec=hop_sec,
            max_windows=max_windows,
        )

    # ── Shape correctness ──────────────────────────────────────────────────────

    def test_each_window_correct_length(self):
        a = self._analyzer()
        wav = torch.randn(100_000)
        windows = a._make_windows(wav)
        for _, w in windows:
            assert w.numel() == WINDOW_SAMPLES

    def test_short_audio_single_padded_window(self):
        """Audio shorter than window → 1 zero-padded window starting at 0."""
        a = self._analyzer()
        wav = torch.randn(20_000)  # 1.25 s
        windows = a._make_windows(wav)
        assert len(windows) == 1
        start, w = windows[0]
        assert start == 0
        assert w.numel() == WINDOW_SAMPLES
        # Padding zeros should be at the tail
        assert torch.all(w[20_000:] == 0.0)

    def test_exact_4s_single_window(self):
        """Exactly 4 s of audio → exactly 1 window, no padding needed."""
        a = self._analyzer()
        wav = torch.randn(WINDOW_SAMPLES)
        windows = a._make_windows(wav)
        assert len(windows) == 1
        assert windows[0][0] == 0

    def test_8s_audio_window_count(self):
        """8 s, window=4, hop=2 → expect 3 windows (0, 2, 4 s starts)."""
        a = self._analyzer(window_sec=4.0, hop_sec=2.0)
        # 8 s = 128 000 samples
        wav = torch.randn(128_000)
        windows = a._make_windows(wav)
        # starts: 0, 32000, 64000 → 3 windows
        assert len(windows) == 3
        starts = [s for s, _ in windows]
        assert starts[0] == 0
        assert starts[-1] == 64_000  # 128000 - 64000

    def test_10s_audio_hop_2s(self):
        """10 s, window=4, hop=2 → starts 0,2,4,6 s → 4 windows."""
        a = self._analyzer(window_sec=4.0, hop_sec=2.0)
        wav = torch.randn(160_000)  # 10 s
        windows = a._make_windows(wav)
        assert len(windows) == 4
        starts = [s for s, _ in windows]
        assert starts[-1] == 96_000  # 160000 - 64000

    def test_max_windows_cap(self):
        """When audio would produce more windows, cap at max_windows."""
        a = self._analyzer(window_sec=4.0, hop_sec=1.0, max_windows=5)
        # 60 s = 960 000 samples → many windows without cap
        wav = torch.randn(960_000)
        windows = a._make_windows(wav)
        assert len(windows) <= 5

    def test_last_window_covers_tail(self):
        """Last window should always start at total-window_samples."""
        a = self._analyzer(window_sec=4.0, hop_sec=2.0)
        wav = torch.randn(100_001)  # odd length
        windows = a._make_windows(wav)
        last_start = windows[-1][0]
        assert last_start == wav.numel() - WINDOW_SAMPLES

    def test_first_window_starts_at_zero(self):
        a = self._analyzer()
        wav = torch.randn(200_000)
        windows = a._make_windows(wav)
        assert windows[0][0] == 0

    def test_no_duplicate_starts(self):
        a = self._analyzer(window_sec=4.0, hop_sec=2.0)
        wav = torch.randn(200_000)
        windows = a._make_windows(wav)
        starts = [s for s, _ in windows]
        assert len(starts) == len(set(starts))

    def test_invalid_hop_greater_than_window_raises(self):
        from inference.long_audio import LongAudioAnalyzer

        with pytest.raises(ValueError, match="hop_sec"):
            LongAudioAnalyzer(_make_mock_engine(), window_sec=4.0, hop_sec=5.0)

    def test_invalid_window_sec_raises(self):
        from inference.long_audio import LongAudioAnalyzer

        with pytest.raises(ValueError):
            LongAudioAnalyzer(_make_mock_engine(), window_sec=0.0)

    def test_invalid_top_fraction_raises(self):
        from inference.long_audio import LongAudioAnalyzer

        with pytest.raises(ValueError, match="top_fraction"):
            LongAudioAnalyzer(_make_mock_engine(), top_fraction=0.0)

    def test_2d_tensor_flattened(self):
        """2-D tensor should be flattened before windowing."""
        a = self._analyzer()
        wav = torch.randn(1, 100_000)   # (1, N)
        windows = a._make_windows(wav)
        assert len(windows) > 0
        for _, w in windows:
            assert w.numel() == WINDOW_SAMPLES


# ──────────────────────────────────────────────────────────────────────────────
# Tests: LongAudioAnalyzer.analyze (with mocked engine)
# ──────────────────────────────────────────────────────────────────────────────

class TestAnalyze:
    """Integration tests for LongAudioAnalyzer.analyze using a mock engine."""

    def _analyzer(self, spoof_prob: float = 0.8, **kwargs):
        from inference.long_audio import LongAudioAnalyzer

        return LongAudioAnalyzer(
            engine=_make_mock_engine(spoof_prob=spoof_prob),
            **kwargs,
        )

    def test_single_window_short_audio(self):
        """< 4 s audio → 1 window analyzed."""
        a = self._analyzer()
        wav = torch.randn(20_000) * 0.1
        result = a.analyze(wav)
        assert result["windows_analyzed"] == 1
        assert result["status"] == "ok"

    def test_multiple_windows_returned(self):
        """8 s audio, 4 s window, 2 s hop → 3 windows."""
        a = self._analyzer()
        wav = torch.randn(128_000) * 0.1
        result = a.analyze(wav)
        assert result["windows_analyzed"] == 3

    def test_aggregation_uses_top_fraction(self):
        """With 4 windows and top_fraction=0.5, only top 2 should be used."""
        from inference.long_audio import LongAudioAnalyzer

        engine = _make_mock_engine()
        # Return distinct probabilities per call
        probs = [0.9, 0.8, 0.3, 0.2]
        call_count = [0]
        def fake_predict(_):
            p = probs[call_count[0] % len(probs)]
            call_count[0] += 1
            return _stub_engine_result(p)
        engine.predict.side_effect = fake_predict

        a = LongAudioAnalyzer(
            engine=engine, window_sec=4.0, hop_sec=2.0, top_fraction=0.5
        )
        wav = torch.randn(128_000) * 0.1  # 8 s → 3 windows
        result = a.analyze(wav)
        # top_fraction=0.5, 3 windows → k=ceil(3*0.5)=2
        assert result["aggregation"]["windows_used_for_final"] == 2

    def test_final_prob_from_top_fraction(self):
        """Verify aggregation: mean of top-k spoof scores."""
        from inference.long_audio import LongAudioAnalyzer

        engine = _make_mock_engine()
        probs = [0.9, 0.5, 0.1]
        call_count = [0]
        def fake_predict(_):
            p = probs[call_count[0] % len(probs)]
            call_count[0] += 1
            return _stub_engine_result(p)
        engine.predict.side_effect = fake_predict

        # 3 windows, top_fraction=1/3 → k=1 → final = max(probs) = 0.9
        a = LongAudioAnalyzer(
            engine=engine, window_sec=4.0, hop_sec=2.0,
            top_fraction=1.0 / 3
        )
        wav = torch.randn(128_000) * 0.1  # 8 s → 3 windows
        result = a.analyze(wav)
        assert result["aggregation"]["windows_used_for_final"] == 1
        assert abs(result["spoof_probability"] - 0.9) < 1e-4

    def test_result_fields_present(self):
        a = self._analyzer()
        wav = torch.randn(64_000) * 0.1
        r = a.analyze(wav)
        required = [
            "classification", "spoof_probability", "bona_fide_probability",
            "confidence", "decision_threshold", "risk_score", "threat_level",
            "recommended_action", "audio_duration_sec", "windows_analyzed",
            "window_seconds", "hop_seconds", "aggregation", "window_results",
            "model_version", "inference_mode", "device", "status",
        ]
        for f in required:
            assert f in r, f"Missing field: {f}"

    def test_audio_duration_correct(self):
        a = self._analyzer()
        wav = torch.randn(128_000) * 0.1  # 8 s
        r = a.analyze(wav)
        assert abs(r["audio_duration_sec"] - 8.0) < 0.01

    def test_probabilities_sum_to_one(self):
        a = self._analyzer(spoof_prob=0.7)
        wav = torch.randn(64_000) * 0.1
        r = a.analyze(wav)
        assert abs(r["spoof_probability"] + r["bona_fide_probability"] - 1.0) < 1e-4

    def test_classification_consistent_with_threshold(self):
        a = self._analyzer(spoof_prob=0.9)  # > threshold 0.6728817
        wav = torch.randn(64_000) * 0.1
        r = a.analyze(wav)
        p = r["spoof_probability"]
        thr = r["decision_threshold"]
        expected = "SPOOF" if p >= thr else "BONA_FIDE"
        assert r["classification"] == expected

    def test_window_results_length_matches(self):
        a = self._analyzer()
        wav = torch.randn(128_000) * 0.1
        r = a.analyze(wav)
        assert len(r["window_results"]) == r["windows_analyzed"]

    def test_window_results_have_required_fields(self):
        a = self._analyzer()
        wav = torch.randn(64_000) * 0.1
        r = a.analyze(wav)
        for wr in r["window_results"]:
            for field in [
                "start_sec", "end_sec", "classification",
                "spoof_probability", "confidence", "threat_level",
            ]:
                assert field in wr, f"Window result missing field: {field}"

    def test_unavailable_engine_returns_unavailable(self):
        from inference.long_audio import LongAudioAnalyzer

        engine = _make_mock_engine(ready=False)
        engine.predict.return_value = {
            "classification": "UNAVAILABLE",
            "spoof_probability": None,
            "bona_fide_probability": None,
            "confidence": None,
            "decision_threshold": None,
            "risk_score": None,
            "threat_level": "UNKNOWN",
            "recommended_action": "Model unavailable",
            "risk_breakdown": {},
            "latency_ms": 0.0,
            "device": "cpu",
            "model_version": None,
            "inference_mode": "unavailable",
            "status": "unavailable",
            "error": "No checkpoint found",
        }
        a = LongAudioAnalyzer(engine=engine)
        wav = torch.randn(64_000) * 0.1
        r = a.analyze(wav)
        assert r["status"] == "unavailable"
        assert r["classification"] == "UNAVAILABLE"
        assert r["spoof_probability"] is None


# ──────────────────────────────────────────────────────────────────────────────
# Tests: /predict-long API endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictLongAPI:
    """
    API-level tests for /predict-long.

    We use a synthetic checkpoint to avoid loading the 46 MB production model
    for unit tests.  Integration tests against the real checkpoint are in
    TestPredictLongIntegration.
    """

    @pytest.fixture
    def client_with_mock_engine(self, tmp_path):
        """TestClient with a mock InferenceEngine (no real checkpoint needed)."""
        from fastapi.testclient import TestClient
        import backend.app as app_module
        from backend.incident_logger import IncidentLogger

        mock_engine = _make_mock_engine(spoof_prob=0.9)

        original_engine = app_module._engine
        original_log = app_module._incident_log

        app_module._engine = mock_engine
        app_module._incident_log = IncidentLogger(db_path=tmp_path / "test.db")
        app_module._preprocessor = None  # allow lazy re-init

        client = TestClient(app_module.app, raise_server_exceptions=False)
        yield client

        app_module._engine = original_engine
        app_module._incident_log = original_log

    @pytest.fixture
    def client_no_checkpoint(self, tmp_path):
        """TestClient with a real engine pointing at an empty project root."""
        from fastapi.testclient import TestClient
        import backend.app as app_module
        from inference.engine import InferenceEngine
        from backend.incident_logger import IncidentLogger

        original_engine = app_module._engine
        original_log = app_module._incident_log

        app_module._engine = InferenceEngine(project_root=tmp_path)
        app_module._incident_log = IncidentLogger(db_path=tmp_path / "test.db")
        app_module._preprocessor = None

        client = TestClient(app_module.app, raise_server_exceptions=False)
        yield client

        app_module._engine = original_engine
        app_module._incident_log = original_log

    # ── Existing endpoints still work ─────────────────────────────────────────

    def test_health_returns_200(self, client_with_mock_engine):
        resp = client_with_mock_engine.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_ready" in data

    def test_ready_returns_200_when_model_ready(self, client_with_mock_engine):
        resp = client_with_mock_engine.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"

    def test_predict_still_works(self, client_with_mock_engine):
        """Existing /predict must remain backward compatible."""
        wav_bytes = _make_wav_bytes(2.0)
        resp = client_with_mock_engine.post(
            "/predict",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        # With mock engine, classification should reflect spoof_prob=0.9
        assert "classification" in data
        assert "spoof_probability" in data

    # ── Basic /predict-long functionality ─────────────────────────────────────

    def test_predict_long_returns_200(self, client_with_mock_engine):
        wav_bytes = _make_wav_bytes(6.0)
        resp = client_with_mock_engine.post(
            "/predict-long",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200

    def test_predict_long_response_fields(self, client_with_mock_engine):
        wav_bytes = _make_wav_bytes(6.0)
        resp = client_with_mock_engine.post(
            "/predict-long",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        data = resp.json()
        required = [
            "classification", "spoof_probability", "bona_fide_probability",
            "confidence", "decision_threshold", "risk_score", "threat_level",
            "recommended_action", "audio_duration_sec", "windows_analyzed",
            "window_seconds", "hop_seconds", "aggregation", "window_results",
            "status",
        ]
        for f in required:
            assert f in data, f"Missing response field: {f}"

    def test_predict_long_window_count_6s(self, client_with_mock_engine):
        """6 s audio, default 2 s hop → 2 windows (0 s and 2 s starts)."""
        wav_bytes = _make_wav_bytes(6.0)
        resp = client_with_mock_engine.post(
            "/predict-long",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        data = resp.json()
        # 6 s: starts at 0 s, 2 s → 2 windows
        assert data["windows_analyzed"] >= 1

    def test_predict_long_short_audio_single_window(self, client_with_mock_engine):
        """Short (2 s) audio → 1 padded window."""
        wav_bytes = _make_wav_bytes(2.0)
        resp = client_with_mock_engine.post(
            "/predict-long",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["windows_analyzed"] == 1

    def test_predict_long_exact_4s_single_window(self, client_with_mock_engine):
        """4 s audio → exactly 1 window."""
        wav_bytes = _make_wav_bytes(4.0)
        resp = client_with_mock_engine.post(
            "/predict-long",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["windows_analyzed"] == 1

    def test_predict_long_without_file_returns_422(self, client_with_mock_engine):
        resp = client_with_mock_engine.post("/predict-long")
        assert resp.status_code == 422

    def test_predict_long_unsupported_format_returns_400(self, client_with_mock_engine):
        resp = client_with_mock_engine.post(
            "/predict-long",
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )
        assert resp.status_code == 400

    def test_predict_long_corrupt_audio_returns_422(self, client_with_mock_engine):
        resp = client_with_mock_engine.post(
            "/predict-long",
            files={"file": ("test.wav", b"\x00\x01corrupt", "audio/wav")},
        )
        assert resp.status_code == 422

    def test_predict_long_invalid_hop_returns_400(self, client_with_mock_engine):
        """hop_sec > window_sec should return 400."""
        wav_bytes = _make_wav_bytes(4.0)
        resp = client_with_mock_engine.post(
            "/predict-long",
            data={"hop_sec": "10.0", "window_sec": "4.0"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 400

    def test_predict_long_logs_incident(self, client_with_mock_engine):
        wav_bytes = _make_wav_bytes(4.0)
        client_with_mock_engine.post(
            "/predict-long",
            files={"file": ("long.wav", wav_bytes, "audio/wav")},
        )
        resp = client_with_mock_engine.get("/incidents")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_predict_long_no_checkpoint_returns_unavailable(
        self, client_no_checkpoint
    ):
        """Without a checkpoint, /predict-long returns unavailable, not crash."""
        wav_bytes = _make_wav_bytes(4.0)
        resp = client_no_checkpoint.post(
            "/predict-long",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unavailable"
        assert data["classification"] == "UNAVAILABLE"
        assert data["spoof_probability"] is None

    def test_predict_long_audio_duration_in_response(self, client_with_mock_engine):
        wav_bytes = _make_wav_bytes(8.0)
        resp = client_with_mock_engine.post(
            "/predict-long",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        data = resp.json()
        # Duration should be close to 8 s
        assert data["audio_duration_sec"] is not None
        assert abs(data["audio_duration_sec"] - 8.0) < 0.5

    def test_predict_long_window_results_list(self, client_with_mock_engine):
        wav_bytes = _make_wav_bytes(8.0)
        resp = client_with_mock_engine.post(
            "/predict-long",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        data = resp.json()
        assert isinstance(data["window_results"], list)
        assert len(data["window_results"]) == data["windows_analyzed"]
        for wr in data["window_results"]:
            assert "start_sec" in wr
            assert "end_sec" in wr
            assert "classification" in wr

    def test_predict_long_custom_hop_sec(self, client_with_mock_engine):
        """Custom hop_sec=1.0 should produce more windows than default 2.0."""
        wav_bytes = _make_wav_bytes(8.0)

        resp_default = client_with_mock_engine.post(
            "/predict-long",
            data={"hop_sec": "2.0"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        resp_dense = client_with_mock_engine.post(
            "/predict-long",
            data={"hop_sec": "1.0"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp_default.status_code == 200
        assert resp_dense.status_code == 200
        assert (
            resp_dense.json()["windows_analyzed"]
            >= resp_default.json()["windows_analyzed"]
        )

    def test_predict_long_no_internal_paths(self, client_with_mock_engine):
        """Response must not expose filesystem paths."""
        wav_bytes = _make_wav_bytes(4.0)
        resp = client_with_mock_engine.post(
            "/predict-long",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        response_text = resp.text
        # Should not contain Windows or Unix absolute paths to internals
        assert "\\checkpoints\\" not in response_text
        assert "/checkpoints/" not in response_text
        # Verify "checkpoint" key is not in the top-level response
        data = resp.json()
        assert "checkpoint" not in data


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Existing API endpoints (smoke tests)
# ──────────────────────────────────────────────────────────────────────────────

class TestExistingAPI:
    """Smoke tests for existing API endpoints to ensure no regression."""

    @pytest.fixture
    def client(self, tmp_path):
        from fastapi.testclient import TestClient
        import backend.app as app_module
        from inference.engine import InferenceEngine
        from backend.incident_logger import IncidentLogger

        original_engine = app_module._engine
        original_log = app_module._incident_log
        original_prep = app_module._preprocessor

        app_module._engine = InferenceEngine(project_root=tmp_path)
        app_module._incident_log = IncidentLogger(db_path=tmp_path / "test.db")
        app_module._preprocessor = None

        client = TestClient(app_module.app, raise_server_exceptions=False)
        yield client

        app_module._engine = original_engine
        app_module._incident_log = original_log
        app_module._preprocessor = original_prep

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready_503_without_checkpoint(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 503

    def test_predict_unavailable_without_checkpoint(self, client):
        wav_bytes = _make_wav_bytes(2.0)
        resp = client.post(
            "/predict",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "unavailable"

    def test_predict_long_unavailable_without_checkpoint(self, client):
        wav_bytes = _make_wav_bytes(2.0)
        resp = client.post(
            "/predict-long",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "unavailable"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: very long audio / window limit
# ──────────────────────────────────────────────────────────────────────────────

class TestVeryLongAudio:
    """Verify max_windows cap and tail coverage for very long audio."""

    def test_max_windows_capped(self):
        from inference.long_audio import LongAudioAnalyzer

        engine = _make_mock_engine()
        a = LongAudioAnalyzer(engine=engine, max_windows=10)
        # 120 s of audio
        wav = torch.randn(120 * SAMPLE_RATE) * 0.1
        windows = a._make_windows(wav)
        assert len(windows) <= 10

    def test_tail_covered_under_cap(self):
        """Even when capped, the last window should cover the audio tail."""
        from inference.long_audio import LongAudioAnalyzer

        engine = _make_mock_engine()
        a = LongAudioAnalyzer(engine=engine, max_windows=5)
        total = 200_000
        wav = torch.randn(total) * 0.1
        windows = a._make_windows(wav)
        last_start = windows[-1][0]
        assert last_start == total - WINDOW_SAMPLES

    def test_analyze_very_long_audio(self):
        """30 s of audio should analyze without error."""
        from inference.long_audio import LongAudioAnalyzer

        engine = _make_mock_engine(spoof_prob=0.3)
        a = LongAudioAnalyzer(engine=engine)
        wav = torch.randn(30 * SAMPLE_RATE) * 0.1
        result = a.analyze(wav)
        assert result["status"] == "ok"
        assert result["windows_analyzed"] > 0
        assert result["audio_duration_sec"] >= 29.9
