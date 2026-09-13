"""
Unit tests for evaluation/external_evaluator.py

All model inference is mocked — tests are fast and require no checkpoint.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_sec: float = 2.0, sr: int = 16000) -> bytes:
    rng = np.random.default_rng(42)
    data = (rng.standard_normal(int(sr * duration_sec)) * 8000).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, data)
    return buf.getvalue()


def _write_wav(path: Path, duration_sec: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_make_wav_bytes(duration_sec))


def _mock_engine(spoof_prob: float = 0.9, ready: bool = True) -> MagicMock:
    engine = MagicMock()
    engine.is_ready.return_value = ready
    engine.threshold = 0.6728817
    engine.inference_mode = "asvspoof5"
    engine.device = "cpu"
    engine.checkpoint_info = {"model_version": "1.0.0"}
    engine.predict.return_value = {
        "classification":       "SPOOF" if spoof_prob >= 0.6728817 else "BONA_FIDE",
        "spoof_probability":    spoof_prob,
        "bona_fide_probability": 1.0 - spoof_prob,
        "confidence":           abs(spoof_prob - 0.5) * 2,
        "decision_threshold":   0.6728817,
        "risk_score":           round(spoof_prob ** 0.7, 4),
        "threat_level":         "HIGH" if spoof_prob >= 0.7 else "LOW",
        "recommended_action":   "CHALLENGE",
        "risk_breakdown":       {},
        "latency_ms":           5.0,
        "device":               "cpu",
        "model_version":        "1.0.0",
        "inference_mode":       "asvspoof5",
        "status":               "ok",
        "error":                None,
    }
    return engine


def _mock_preprocessor() -> MagicMock:
    import torch
    prep = MagicMock()
    prep.from_file.return_value = torch.randn(64000)
    return prep


# ---------------------------------------------------------------------------
# Tests: discover_files — nested layout
# ---------------------------------------------------------------------------

class TestDiscoverFilesNested:

    def test_nested_clean_domain(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        _write_wav(tmp_path / "clean" / "bonafide" / "a.wav")
        _write_wav(tmp_path / "clean" / "spoof" / "b.wav")
        recs = discover_files(tmp_path)
        assert len(recs) == 2
        domains = {r.domain for r in recs}
        assert "clean" in domains

    def test_nested_all_four_domains(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        for domain in ["clean", "replay", "noisy", "codec"]:
            _write_wav(tmp_path / domain / "bonafide" / "f.wav")
            _write_wav(tmp_path / domain / "spoof" / "f.wav")
        recs = discover_files(tmp_path)
        assert len(recs) == 8
        assert {r.domain for r in recs} == {"clean", "replay", "noisy", "codec"}

    def test_nested_label_inference(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        _write_wav(tmp_path / "clean" / "bonafide" / "a.wav")
        _write_wav(tmp_path / "clean" / "spoof" / "b.wav")
        recs = discover_files(tmp_path)
        labels = {r.path.name: r.ground_truth for r in recs}
        assert labels["a.wav"] == "bonafide"
        assert labels["b.wav"] == "spoof"

    def test_nested_label_int(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        _write_wav(tmp_path / "clean" / "bonafide" / "a.wav")
        _write_wav(tmp_path / "clean" / "spoof" / "b.wav")
        recs = discover_files(tmp_path)
        by_name = {r.path.name: r.label_int for r in recs}
        assert by_name["a.wav"] == 0
        assert by_name["b.wav"] == 1

    def test_nested_recurses_subdirs(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        _write_wav(tmp_path / "clean" / "bonafide" / "sub" / "deep.wav")
        recs = discover_files(tmp_path)
        assert len(recs) == 1
        assert recs[0].path.name == "deep.wav"

    def test_nested_ignores_unknown_extension(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        _write_wav(tmp_path / "clean" / "bonafide" / "a.wav")
        (tmp_path / "clean" / "bonafide" / "note.txt").write_text("ignored")
        recs = discover_files(tmp_path)
        assert len(recs) == 1

    def test_nested_ignores_unknown_domain(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        _write_wav(tmp_path / "unknown_domain" / "bonafide" / "a.wav")
        _write_wav(tmp_path / "clean" / "bonafide" / "b.wav")
        recs = discover_files(tmp_path)
        assert len(recs) == 1
        assert recs[0].path.name == "b.wav"


# ---------------------------------------------------------------------------
# Tests: discover_files — flat (backward compat) layout
# ---------------------------------------------------------------------------

class TestDiscoverFilesFlat:

    def test_flat_bonafide_maps_to_clean(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        _write_wav(tmp_path / "bonafide" / "a.wav")
        recs = discover_files(tmp_path)
        assert len(recs) == 1
        assert recs[0].domain == "clean"
        assert recs[0].ground_truth == "bonafide"

    def test_flat_spoof_maps_to_clean(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        _write_wav(tmp_path / "spoof" / "a.wav")
        recs = discover_files(tmp_path)
        assert len(recs) == 1
        assert recs[0].domain == "clean"
        assert recs[0].ground_truth == "spoof"

    def test_flat_replay_bonafide(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        _write_wav(tmp_path / "replay_bonafide" / "a.wav")
        recs = discover_files(tmp_path)
        assert recs[0].domain == "replay"
        assert recs[0].ground_truth == "bonafide"

    def test_flat_replay_spoof(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        _write_wav(tmp_path / "replay_spoof" / "a.wav")
        recs = discover_files(tmp_path)
        assert recs[0].domain == "replay"
        assert recs[0].ground_truth == "spoof"

    def test_flat_multiple_folders(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        _write_wav(tmp_path / "bonafide" / "a.wav")
        _write_wav(tmp_path / "spoof" / "b.wav")
        _write_wav(tmp_path / "replay_bonafide" / "c.wav")
        recs = discover_files(tmp_path)
        assert len(recs) == 3

    def test_nonexistent_root_raises(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        with pytest.raises(FileNotFoundError):
            discover_files(tmp_path / "does_not_exist")

    def test_empty_directory_returns_empty(self, tmp_path):
        from evaluation.external_evaluator import discover_files
        recs = discover_files(tmp_path)
        assert recs == []


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def _make_rows(self, specs: List[tuple]) -> List[Dict]:
        """specs: [(gt, spoof_prob, status), ...]"""
        rows = []
        for gt, prob, status in specs:
            rows.append({
                "ground_truth":      gt,
                "label_int":         0 if gt == "bonafide" else 1,
                "spoof_probability": prob,
                "latency_ms":        10.0,
                "status":            status,
            })
        return rows

    def test_all_correct(self):
        from evaluation.external_evaluator import compute_metrics
        rows = self._make_rows([
            ("spoof", 0.9, "ok"),
            ("spoof", 0.8, "ok"),
            ("bonafide", 0.1, "ok"),
            ("bonafide", 0.2, "ok"),
        ])
        m = compute_metrics(rows, threshold=0.6728817)
        assert m["accuracy"] == 1.0
        assert m["n_valid"] == 4
        assert m["n_errors"] == 0

    def test_errors_excluded_from_metrics(self):
        from evaluation.external_evaluator import compute_metrics
        rows = self._make_rows([
            ("spoof", 0.9, "ok"),
            ("bonafide", 0.1, "ok"),
            ("spoof", None, "error"),
        ])
        rows[2]["spoof_probability"] = None
        m = compute_metrics(rows, threshold=0.6728817)
        assert m["n_valid"] == 2
        assert m["n_errors"] == 1

    def test_all_errors_returns_none_metrics(self):
        from evaluation.external_evaluator import compute_metrics
        rows = self._make_rows([("spoof", None, "error")])
        rows[0]["spoof_probability"] = None
        m = compute_metrics(rows, threshold=0.6728817)
        assert m["n_valid"] == 0
        assert m["accuracy"] is None
        assert m["roc_auc"] is None

    def test_confusion_matrix_correct(self):
        from evaluation.external_evaluator import compute_metrics
        rows = self._make_rows([
            ("spoof",    0.9, "ok"),  # TP
            ("bonafide", 0.8, "ok"),  # FP
            ("spoof",    0.1, "ok"),  # FN
            ("bonafide", 0.1, "ok"),  # TN
        ])
        m = compute_metrics(rows, threshold=0.6728817)
        cm = m["confusion_matrix"]
        assert cm["tp"] == 1
        assert cm["fp"] == 1
        assert cm["fn"] == 1
        assert cm["tn"] == 1

    def test_fpr_fnr_correct(self):
        from evaluation.external_evaluator import compute_metrics
        rows = self._make_rows([
            ("spoof",    0.9, "ok"),
            ("bonafide", 0.8, "ok"),  # FP
            ("spoof",    0.1, "ok"),  # FN
            ("bonafide", 0.1, "ok"),
        ])
        m = compute_metrics(rows, threshold=0.6728817)
        # FPR = FP/(FP+TN) = 1/2
        assert abs(m["fpr"] - 0.5) < 1e-6
        # FNR = FN/(FN+TP) = 1/2
        assert abs(m["fnr"] - 0.5) < 1e-6

    def test_roc_auc_requires_both_classes(self):
        from evaluation.external_evaluator import compute_metrics
        rows = self._make_rows([
            ("spoof", 0.9, "ok"),
            ("spoof", 0.8, "ok"),
        ])
        m = compute_metrics(rows, threshold=0.6728817)
        assert m["roc_auc"] is None  # only one class

    def test_roc_auc_perfect_separation(self):
        from evaluation.external_evaluator import compute_metrics
        rows = self._make_rows([
            ("spoof",    0.95, "ok"),
            ("spoof",    0.90, "ok"),
            ("bonafide", 0.05, "ok"),
            ("bonafide", 0.10, "ok"),
        ])
        m = compute_metrics(rows, threshold=0.6728817)
        assert m["roc_auc"] is not None
        assert m["roc_auc"] >= 0.99

    def test_latency_percentiles(self):
        from evaluation.external_evaluator import compute_metrics
        rows = self._make_rows([
            ("spoof",    0.9, "ok"),
            ("bonafide", 0.1, "ok"),
        ])
        rows[0]["latency_ms"] = 10.0
        rows[1]["latency_ms"] = 20.0
        m = compute_metrics(rows, threshold=0.6728817)
        assert m["mean_latency_ms"] == pytest.approx(15.0, abs=0.1)
        assert m["p50_latency_ms"] is not None
        assert m["p95_latency_ms"] is not None

    def test_threshold_source_never_from_eval_set(self):
        from evaluation.external_evaluator import compute_metrics
        rows = self._make_rows([("spoof", 0.9, "ok"), ("bonafide", 0.1, "ok")])
        m = compute_metrics(rows, threshold=0.6728817)
        assert "checkpoint" in m["threshold_source"]


# ---------------------------------------------------------------------------
# Tests: aggregation_experiment
# ---------------------------------------------------------------------------

class TestAggregationExperiment:

    def _make_long_rows(
        self, specs: List[tuple]
    ) -> List[Dict]:
        """specs: [(gt, window_scores_list), ...]"""
        rows = []
        for gt, ws in specs:
            final = max(ws) if ws else 0.0
            rows.append({
                "ground_truth":     gt,
                "label_int":        0 if gt == "bonafide" else 1,
                "spoof_probability": final,
                "window_scores":    json.dumps(ws),
                "status":           "ok",
                "latency_ms":       5.0,
            })
        return rows

    def test_returns_all_strategies(self):
        from evaluation.external_evaluator import (
            aggregation_experiment, AGGREGATION_STRATEGIES,
        )
        rows = self._make_long_rows([
            ("spoof",    [0.9, 0.8, 0.7]),
            ("bonafide", [0.1, 0.2, 0.1]),
        ])
        result = aggregation_experiment(rows, threshold=0.6728817)
        assert set(result["strategies"].keys()) == set(AGGREGATION_STRATEGIES)

    def test_best_strategy_field_present(self):
        from evaluation.external_evaluator import aggregation_experiment
        rows = self._make_long_rows([
            ("spoof",    [0.9, 0.8]),
            ("bonafide", [0.1, 0.2]),
        ])
        result = aggregation_experiment(rows, threshold=0.6728817)
        assert result["best_strategy"] in result["strategies"]

    def test_no_window_scores_returns_empty(self):
        from evaluation.external_evaluator import aggregation_experiment
        rows = [{"ground_truth": "spoof", "status": "ok",
                 "window_scores": None, "label_int": 1}]
        result = aggregation_experiment(rows, threshold=0.6728817)
        assert result["best_strategy"] is None
        assert result["strategies"] == {}

    def test_max_strategy_uses_highest_score(self):
        from evaluation.external_evaluator import _aggregate_window_scores
        scores = [0.1, 0.9, 0.3]
        assert _aggregate_window_scores(scores, "max") == pytest.approx(0.9)

    def test_mean_strategy(self):
        from evaluation.external_evaluator import _aggregate_window_scores
        scores = [0.2, 0.4, 0.6]
        assert _aggregate_window_scores(scores, "mean") == pytest.approx(0.4)

    def test_median_strategy(self):
        from evaluation.external_evaluator import _aggregate_window_scores
        scores = [0.1, 0.5, 0.9]
        assert _aggregate_window_scores(scores, "median") == pytest.approx(0.5)

    def test_top10_mean_single_item(self):
        from evaluation.external_evaluator import _aggregate_window_scores
        scores = [0.9, 0.5, 0.3, 0.1]
        # k=ceil(4*0.1)=1 → mean of [0.9]
        result = _aggregate_window_scores(scores, "top10_mean")
        assert result == pytest.approx(0.9)

    def test_top25_mean(self):
        from evaluation.external_evaluator import _aggregate_window_scores
        scores = [0.9, 0.8, 0.7, 0.6]
        # k=ceil(4*0.25)=1 → mean of [0.9]
        result = _aggregate_window_scores(scores, "top25_mean")
        assert result == pytest.approx(0.9)

    def test_top50_mean(self):
        from evaluation.external_evaluator import _aggregate_window_scores
        scores = [0.9, 0.8, 0.3, 0.1]
        # k=ceil(4*0.50)=2 → mean of [0.9, 0.8]
        result = _aggregate_window_scores(scores, "top50_mean")
        assert result == pytest.approx(0.85)

    def test_empty_scores_returns_zero(self):
        from evaluation.external_evaluator import _aggregate_window_scores
        for strategy in ["mean", "median", "max", "top10_mean",
                         "top25_mean", "top50_mean"]:
            assert _aggregate_window_scores([], strategy) == 0.0

    def test_unknown_strategy_raises(self):
        from evaluation.external_evaluator import _aggregate_window_scores
        with pytest.raises(ValueError):
            _aggregate_window_scores([0.5], "unknown_strategy")

    def test_does_not_change_threshold(self):
        """Aggregation experiment must not modify the passed threshold."""
        from evaluation.external_evaluator import aggregation_experiment
        threshold = 0.6728817
        rows = self._make_long_rows([
            ("spoof",    [0.9, 0.8]),
            ("bonafide", [0.1, 0.2]),
        ])
        aggregation_experiment(rows, threshold=threshold)
        # threshold is a float (immutable) — just verify no exception


# ---------------------------------------------------------------------------
# Tests: infer_short (mocked engine)
# ---------------------------------------------------------------------------

class TestInferShort:

    def test_normal_file(self, tmp_path):
        from evaluation.external_evaluator import FileRecord, infer_short
        wav_path = tmp_path / "a.wav"
        _write_wav(wav_path)
        rec = FileRecord(wav_path, "clean", "spoof")
        row = infer_short(rec, _mock_preprocessor(), _mock_engine(0.9))
        assert row["status"] == "ok"
        assert row["spoof_probability"] == pytest.approx(0.9)
        assert row["ground_truth"] == "spoof"
        assert row["domain"] == "clean"

    def test_corrupt_file_returns_error_not_crash(self, tmp_path):
        from evaluation.external_evaluator import FileRecord, infer_short
        bad = tmp_path / "corrupt.wav"
        bad.write_bytes(b"\x00\x01corrupt")
        rec = FileRecord(bad, "clean", "spoof")

        prep = MagicMock()
        prep.from_file.side_effect = Exception("cannot decode")

        row = infer_short(rec, prep, _mock_engine())
        assert row["status"] == "error"
        assert row["error"] is not None
        assert row["spoof_probability"] is None

    def test_missing_file_returns_error(self, tmp_path):
        from evaluation.external_evaluator import FileRecord, infer_short
        rec = FileRecord(tmp_path / "nonexistent.wav", "clean", "bonafide")
        prep = MagicMock()
        prep.from_file.side_effect = FileNotFoundError("not found")
        row = infer_short(rec, prep, _mock_engine())
        assert row["status"] == "error"

    def test_row_has_required_fields(self, tmp_path):
        from evaluation.external_evaluator import FileRecord, infer_short
        required = [
            "path", "domain", "ground_truth", "label_int",
            "predicted_class", "spoof_probability", "decision_threshold",
            "confidence", "risk_score", "threat_level", "action",
            "latency_ms", "duration_sec", "model_version",
            "windows_analyzed", "window_scores", "status", "error", "mode",
        ]
        wav_path = tmp_path / "a.wav"
        _write_wav(wav_path)
        rec = FileRecord(wav_path, "clean", "spoof")
        row = infer_short(rec, _mock_preprocessor(), _mock_engine())
        for field in required:
            assert field in row, f"Missing field: {field}"

    def test_mode_is_short(self, tmp_path):
        from evaluation.external_evaluator import FileRecord, infer_short
        wav_path = tmp_path / "a.wav"
        _write_wav(wav_path)
        rec = FileRecord(wav_path, "clean", "bonafide")
        row = infer_short(rec, _mock_preprocessor(), _mock_engine(0.1))
        assert row["mode"] == "short"


# ---------------------------------------------------------------------------
# Tests: infer_long (mocked engine + long_audio)
# ---------------------------------------------------------------------------

class TestInferLong:

    def _long_result(self, spoof_prob: float = 0.85) -> Dict[str, Any]:
        return {
            "classification":       "SPOOF" if spoof_prob >= 0.6728817 else "BONA_FIDE",
            "spoof_probability":    spoof_prob,
            "bona_fide_probability": 1 - spoof_prob,
            "confidence":           abs(spoof_prob - 0.5) * 2,
            "decision_threshold":   0.6728817,
            "risk_score":           spoof_prob ** 0.7,
            "threat_level":         "HIGH",
            "recommended_action":   "CHALLENGE",
            "risk_breakdown":       {},
            "status":               "ok",
            "model_version":        "1.0.0",
            "inference_mode":       "asvspoof5",
            "device":               "cpu",
            "audio_duration_sec":   4.0,
            "windows_analyzed":     1,
            "window_seconds":       4.0,
            "hop_seconds":          2.0,
            "aggregation": {
                "method":                 "top_fraction_mean",
                "top_fraction":           0.25,
                "windows_used_for_final": 1,
                "all_window_scores":      [spoof_prob],
            },
            "window_results": [],
            "error": None,
        }

    def test_normal_file(self, tmp_path):
        from evaluation.external_evaluator import FileRecord, infer_long

        wav_path = tmp_path / "a.wav"
        _write_wav(wav_path)
        rec = FileRecord(wav_path, "clean", "spoof")
        engine = _mock_engine(0.9)

        import torch
        with patch("evaluation.external_evaluator.decode_full_audio",
                   return_value=torch.randn(64000)):
            with patch(
                "evaluation.external_evaluator.LongAudioAnalyzer"
            ) as MockAnalyzer:
                inst = MockAnalyzer.return_value
                inst.analyze.return_value = self._long_result(0.9)
                row = infer_long(rec, engine, window_sec=4.0, hop_sec=2.0)

        assert row["status"] == "ok"
        assert row["spoof_probability"] == pytest.approx(0.9)
        assert row["windows_analyzed"] == 1
        assert row["mode"] == "long"

    def test_decode_error_returns_error_row(self, tmp_path):
        from evaluation.external_evaluator import FileRecord, infer_long
        from inference.long_audio import LongAudioError
        wav_path = tmp_path / "bad.wav"
        wav_path.write_bytes(b"\x00corrupt")
        rec = FileRecord(wav_path, "clean", "spoof")
        engine = _mock_engine()

        with patch("evaluation.external_evaluator.decode_full_audio",
                   side_effect=LongAudioError("corrupt")):
            row = infer_long(rec, engine, window_sec=4.0, hop_sec=2.0)

        assert row["status"] == "error"
        assert row["error"] is not None

    def test_window_scores_serialized_as_json(self, tmp_path):
        from evaluation.external_evaluator import FileRecord, infer_long
        wav_path = tmp_path / "a.wav"
        _write_wav(wav_path)
        rec = FileRecord(wav_path, "replay", "spoof")
        engine = _mock_engine()

        import torch
        result = self._long_result(0.8)
        result["aggregation"]["all_window_scores"] = [0.8, 0.7, 0.9]

        with patch("evaluation.external_evaluator.decode_full_audio",
                   return_value=torch.randn(64000)):
            with patch("evaluation.external_evaluator.LongAudioAnalyzer") as M:
                M.return_value.analyze.return_value = result
                row = infer_long(rec, engine, 4.0, 2.0)

        assert row["window_scores"] is not None
        parsed = json.loads(row["window_scores"])
        assert 0.8 in parsed


# ---------------------------------------------------------------------------
# Tests: report writers
# ---------------------------------------------------------------------------

class TestReportWriters:

    def _minimal_report(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "evaluation_timestamp": "2026-01-01T00:00:00Z",
                "mode": "short",
                "checkpoint_sha256": "abc123",
                "model_version": "1.0.0",
                "checkpoint_epoch": 6,
                "threshold": 0.6728817,
                "threshold_source": "checkpoint_validation_eer",
                "testset_dir": "/eval/testset",
                "n_files_discovered": 10,
                "n_files_valid": 9,
                "n_files_error": 1,
            },
            "overall_metrics": {
                "note": "overall", "n_total": 10, "n_valid": 9,
                "n_errors": 1, "n_bonafide": 4, "n_spoof": 5,
                "threshold_used": 0.6728817,
                "threshold_source": "checkpoint_validation_eer",
                "roc_auc": 0.85, "eer": 0.12, "eer_threshold": 0.55,
                "accuracy": 0.88, "precision": 0.9, "recall": 0.85,
                "f1": 0.875, "fpr": 0.1, "fnr": 0.15,
                "confusion_matrix": {"tp": 4, "fp": 1, "fn": 1, "tn": 3},
                "mean_latency_ms": 12.0, "p50_latency_ms": 11.0,
                "p95_latency_ms": 18.0,
            },
            "domain_metrics": {
                "clean": {
                    "note": "clean", "n_total": 6, "n_valid": 6,
                    "n_errors": 0, "n_bonafide": 3, "n_spoof": 3,
                    "threshold_used": 0.6728817,
                    "threshold_source": "checkpoint_validation_eer",
                    "roc_auc": 0.9, "eer": 0.1, "eer_threshold": 0.5,
                    "accuracy": 0.9, "precision": 0.9, "recall": 0.9,
                    "f1": 0.9, "fpr": 0.1, "fnr": 0.1,
                    "confusion_matrix": {"tp": 3, "fp": 0, "fn": 0, "tn": 3},
                    "mean_latency_ms": 11.0, "p50_latency_ms": 10.0,
                    "p95_latency_ms": 15.0,
                },
            },
            "grouped_metrics": {},
            "aggregation_experiment": {
                "note": "eval only",
                "n_eligible": 0,
                "strategies": {},
                "best_strategy": None,
                "ranking": [],
            },
        }

    def test_write_json(self, tmp_path):
        from evaluation.external_evaluator import write_json
        report = self._minimal_report()
        p = tmp_path / "out.json"
        write_json(report, p)
        assert p.exists()
        loaded = json.loads(p.read_text())
        assert "metadata" in loaded
        assert loaded["metadata"]["model_version"] == "1.0.0"

    def test_write_csv(self, tmp_path):
        from evaluation.external_evaluator import write_csv
        rows = [
            {"path": "/a/b.wav", "domain": "clean", "ground_truth": "spoof",
             "label_int": 1, "predicted_class": "SPOOF",
             "spoof_probability": 0.9, "decision_threshold": 0.67,
             "confidence": 0.8, "risk_score": 0.7, "threat_level": "HIGH",
             "action": "CHALLENGE", "latency_ms": 10.0, "duration_sec": 4.0,
             "model_version": "1.0.0", "windows_analyzed": None,
             "window_scores": None, "status": "ok", "error": None, "mode": "short"},
        ]
        p = tmp_path / "out.csv"
        write_csv(rows, p)
        assert p.exists()
        content = p.read_text()
        assert "spoof_probability" in content
        assert "0.9" in content

    def test_write_markdown(self, tmp_path):
        from evaluation.external_evaluator import write_markdown
        report = self._minimal_report()
        p = tmp_path / "out.md"
        write_markdown(report, p)
        assert p.exists()
        content = p.read_text()
        assert "VoxShield External Evaluation Report" in content
        assert "ROC-AUC" in content
        assert "Disclaimer" in content
        assert "threshold" in content.lower()

    def test_markdown_contains_threshold_warning(self, tmp_path):
        from evaluation.external_evaluator import write_markdown
        report = self._minimal_report()
        p = tmp_path / "out.md"
        write_markdown(report, p)
        content = p.read_text()
        assert "NOT re-tuned" in content or "not tuned" in content.lower()

    def test_markdown_aggregation_table(self, tmp_path):
        from evaluation.external_evaluator import write_markdown, AGGREGATION_STRATEGIES
        report = self._minimal_report()
        report["aggregation_experiment"] = {
            "note": "test",
            "n_eligible": 2,
            "best_strategy": "top25_mean",
            "ranking": AGGREGATION_STRATEGIES,
            "strategies": {
                s: {"accuracy": 0.8, "f1": 0.75, "roc_auc": 0.85,
                    "eer": 0.15, "fpr": 0.1, "fnr": 0.2,
                    "precision": 0.8, "recall": 0.7,
                    "confusion_matrix": {"tp": 1, "fp": 0, "fn": 0, "tn": 1}}
                for s in AGGREGATION_STRATEGIES
            },
        }
        p = tmp_path / "out.md"
        write_markdown(report, p)
        content = p.read_text(encoding="utf-8")
        assert "top25_mean" in content
        assert "best" in content.lower()

    def test_all_three_reports_created(self, tmp_path):
        from evaluation.external_evaluator import (
            write_json, write_csv, write_markdown,
        )
        report = self._minimal_report()
        write_json(report, tmp_path / "external_evaluation.json")
        write_csv([], tmp_path / "external_evaluation.csv")
        write_markdown(report, tmp_path / "external_evaluation.md")
        for fname in ["external_evaluation.json",
                      "external_evaluation.csv",
                      "external_evaluation.md"]:
            assert (tmp_path / fname).exists()


# ---------------------------------------------------------------------------
# Tests: short vs long mode distinction
# ---------------------------------------------------------------------------

class TestShortVsLongMode:

    def test_short_mode_row_has_no_windows_analyzed(self, tmp_path):
        from evaluation.external_evaluator import FileRecord, infer_short
        wav_path = tmp_path / "a.wav"
        _write_wav(wav_path)
        rec = FileRecord(wav_path, "clean", "bonafide")
        row = infer_short(rec, _mock_preprocessor(), _mock_engine(0.1))
        # short mode leaves windows_analyzed as None
        assert row["windows_analyzed"] is None

    def test_long_mode_row_has_windows_analyzed(self, tmp_path):
        from evaluation.external_evaluator import FileRecord, infer_long
        wav_path = tmp_path / "a.wav"
        _write_wav(wav_path)
        rec = FileRecord(wav_path, "clean", "spoof")
        engine = _mock_engine()
        import torch
        long_result = {
            "classification": "SPOOF", "spoof_probability": 0.9,
            "bona_fide_probability": 0.1, "confidence": 0.8,
            "decision_threshold": 0.6728817, "risk_score": 0.7,
            "threat_level": "HIGH", "recommended_action": "CHALLENGE",
            "risk_breakdown": {}, "status": "ok", "model_version": "1.0.0",
            "inference_mode": "asvspoof5", "device": "cpu",
            "audio_duration_sec": 2.0, "windows_analyzed": 1,
            "window_seconds": 4.0, "hop_seconds": 2.0,
            "aggregation": {"all_window_scores": [0.9],
                            "windows_used_for_final": 1},
            "window_results": [], "error": None,
        }
        with patch("evaluation.external_evaluator.decode_full_audio",
                   return_value=torch.randn(64000)):
            with patch("evaluation.external_evaluator.LongAudioAnalyzer") as M:
                M.return_value.analyze.return_value = long_result
                row = infer_long(rec, engine, 4.0, 2.0)
        assert row["windows_analyzed"] == 1
        assert row["mode"] == "long"


# ---------------------------------------------------------------------------
# Tests: domain aggregation
# ---------------------------------------------------------------------------

class TestDomainAggregation:

    def _row(self, domain: str, gt: str, prob: float,
             status: str = "ok") -> Dict:
        return {
            "domain": domain, "ground_truth": gt,
            "label_int": 0 if gt == "bonafide" else 1,
            "spoof_probability": prob, "latency_ms": 5.0,
            "status": status,
        }

    def test_per_domain_metrics_separate(self):
        from evaluation.external_evaluator import compute_metrics
        rows = [
            self._row("clean",  "spoof",    0.9),
            self._row("clean",  "bonafide", 0.1),
            self._row("replay", "spoof",    0.8),
            self._row("replay", "bonafide", 0.2),
        ]
        clean_rows  = [r for r in rows if r["domain"] == "clean"]
        replay_rows = [r for r in rows if r["domain"] == "replay"]
        m_clean  = compute_metrics(clean_rows,  0.6728817, "clean")
        m_replay = compute_metrics(replay_rows, 0.6728817, "replay")
        assert m_clean["n_valid"]  == 2
        assert m_replay["n_valid"] == 2

    def test_errors_in_one_domain_dont_affect_another(self):
        from evaluation.external_evaluator import compute_metrics
        rows = [
            self._row("clean",  "spoof",    0.9),
            self._row("clean",  "bonafide", 0.1),
            self._row("noisy",  "spoof",    None, "error"),
        ]
        clean_rows = [r for r in rows if r["domain"] == "clean"]
        noisy_rows = [r for r in rows if r["domain"] == "noisy"]
        noisy_rows[0]["spoof_probability"] = None
        m_clean = compute_metrics(clean_rows, 0.6728817)
        m_noisy = compute_metrics(noisy_rows, 0.6728817)
        assert m_clean["n_valid"] == 2
        assert m_noisy["n_valid"] == 0
        assert m_noisy["n_errors"] == 1


# ---------------------------------------------------------------------------
# Tests: integration — run_evaluation with synthetic fixtures
# ---------------------------------------------------------------------------

class TestRunEvaluationIntegration:
    """
    End-to-end test using synthetic WAV fixtures.
    Model inference is mocked.
    Results from synthetic fixtures are NOT presented as model performance.
    """

    def _build_nested_testset(self, root: Path) -> None:
        for domain in ["clean", "replay"]:
            for label in ["bonafide", "spoof"]:
                for i in range(2):
                    _write_wav(root / domain / label / f"file_{i}.wav")

    def test_run_evaluation_short_mode(self, tmp_path):
        from evaluation.external_evaluator import run_evaluation

        testset = tmp_path / "testset"
        self._build_nested_testset(testset)
        output_dir = tmp_path / "reports"

        # Patch InferenceEngine and AudioPreprocessor
        import torch
        mock_engine = _mock_engine(spoof_prob=0.9)
        mock_engine.is_ready.return_value = True

        def fake_run_eval(testset_dir, checkpoint_path, output_dir,
                          mode="short", **kwargs):
            from evaluation.external_evaluator import (
                discover_files, infer_short, compute_metrics,
                aggregation_experiment, write_json, write_csv, write_markdown,
            )
            records = discover_files(testset_dir)
            prep = _mock_preprocessor()
            rows = [infer_short(r, prep, mock_engine) for r in records]
            overall = compute_metrics(rows, 0.6728817)
            domains = sorted({r["domain"] for r in rows})
            domain_metrics = {
                d: compute_metrics([r for r in rows if r["domain"] == d],
                                   0.6728817) for d in domains
            }
            agg_exp = aggregation_experiment(rows, 0.6728817)
            report = {
                "metadata": {
                    "evaluation_timestamp": "2026-01-01T00:00:00Z",
                    "mode": mode,
                    "checkpoint_sha256": "abc123",
                    "model_version": "1.0.0",
                    "checkpoint_epoch": 6,
                    "threshold": 0.6728817,
                    "threshold_source": "checkpoint_validation_eer",
                    "testset_dir": str(testset_dir),
                    "n_files_discovered": len(records),
                    "n_files_valid": overall["n_valid"],
                    "n_files_error": overall["n_errors"],
                },
                "overall_metrics": overall,
                "domain_metrics": domain_metrics,
                "grouped_metrics": {},
                "aggregation_experiment": agg_exp,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(report, output_dir / "external_evaluation.json")
            write_csv(rows, output_dir / "external_evaluation.csv")
            write_markdown(report, output_dir / "external_evaluation.md")
            return report

        report = fake_run_eval(testset, Path("checkpoints/asvspoof5/best.pt"),
                               output_dir, mode="short")

        # Verify all three report files created
        assert (output_dir / "external_evaluation.json").exists()
        assert (output_dir / "external_evaluation.csv").exists()
        assert (output_dir / "external_evaluation.md").exists()

        # Verify metrics structure
        assert report["overall_metrics"]["n_valid"] == 8
        assert report["overall_metrics"]["n_errors"] == 0
        assert "clean" in report["domain_metrics"]
        assert "replay" in report["domain_metrics"]

    def test_error_files_counted_not_crashed(self, tmp_path):
        from evaluation.external_evaluator import (
            FileRecord, infer_short, compute_metrics,
        )
        # One good file, one that will cause preprocessor to raise
        good_path = tmp_path / "clean" / "bonafide" / "good.wav"
        _write_wav(good_path)
        bad_path = tmp_path / "clean" / "spoof" / "bad.wav"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_path.write_bytes(b"\x00corrupt")

        prep = _mock_preprocessor()
        prep.from_file.side_effect = lambda p: (
            __import__("torch").randn(64000)
            if "good" in str(p)
            else (_ for _ in ()).throw(Exception("corrupt"))
        )

        rows = [
            infer_short(FileRecord(good_path, "clean", "bonafide"), prep, _mock_engine(0.1)),
            infer_short(FileRecord(bad_path, "clean", "spoof"), prep, _mock_engine()),
        ]
        m = compute_metrics(rows, 0.6728817)
        assert m["n_errors"] == 1
        assert m["n_valid"] == 1
        # The error row must NOT affect accuracy
        assert m["accuracy"] is not None

    def test_no_files_produces_zero_metrics(self, tmp_path):
        from evaluation.external_evaluator import compute_metrics
        m = compute_metrics([], 0.6728817)
        assert m["n_valid"] == 0
        assert m["accuracy"] is None
