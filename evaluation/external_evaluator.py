"""
VoxShield External Evaluation Harness
======================================

Evaluates the production VoxShieldNet checkpoint on labeled audio
directories that are NOT part of the ASVspoof5 training / validation splits.

This is designed to answer the deployment question:
  "How does the current model perform on real-world audio we collected?"

IMPORTANT PRINCIPLES (non-negotiable):

1.  The production threshold (0.6728817… from the checkpoint) is used as-is.
    It must NOT be re-tuned on this evaluation set.

2.  ERROR files are excluded from ALL classification metrics.
    They are counted and reported separately.

3.  Continuous metrics (ROC-AUC, EER) are threshold-independent.

4.  Threshold-dependent metrics use the checkpoint threshold only.

5.  Aggregation policy comparison is EVALUATION-ONLY.
    It does not change the deployed /predict-long behaviour.

6.  Synthetic test fixtures (created by the test suite) must NOT be
    presented as evidence of production model performance.

DIRECTORY LAYOUT (new canonical format):

    evaluation/external_testset/
        clean/
            bonafide/
            spoof/
        replay/
            bonafide/
            spoof/
        noisy/
            bonafide/
            spoof/
        codec/
            bonafide/
            spoof/

BACKWARD-COMPATIBLE flat layout (also supported):

    evaluation/external_testset/
        bonafide/          → inferred as clean/bonafide
        spoof/             → inferred as clean/spoof
        replay_bonafide/   → inferred as replay/bonafide
        replay_spoof/      → inferred as replay/spoof

Usage:

    python -m evaluation.external_evaluator \\
        --testset evaluation/external_testset \\
        --checkpoint checkpoints/asvspoof5/best.pt \\
        --output-dir evaluation/reports \\
        --mode short

    python -m evaluation.external_evaluator \\
        --testset evaluation/external_testset \\
        --checkpoint checkpoints/asvspoof5/best.pt \\
        --output-dir evaluation/reports \\
        --mode long \\
        --hop-sec 2.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import long-audio helpers at module level so tests can patch them here.
from inference.long_audio import (  # noqa: E402
    decode_full_audio,
    LongAudioAnalyzer,
    LongAudioError,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mp4"})

# Supported test domains
DOMAIN_CLEAN          = "clean"
DOMAIN_REPLAY         = "replay"
DOMAIN_NOISY          = "noisy"
DOMAIN_CODEC          = "codec"
DOMAIN_UNKNOWN        = "unknown"

VALID_DOMAINS = {DOMAIN_CLEAN, DOMAIN_REPLAY, DOMAIN_NOISY, DOMAIN_CODEC}

# Label strings
LABEL_BONAFIDE = "bonafide"
LABEL_SPOOF    = "spoof"

# Numeric labels
LABEL_INT = {LABEL_BONAFIDE: 0, LABEL_SPOOF: 1}

# Old flat-folder → domain mapping
FLAT_FOLDER_DOMAIN_MAP: Dict[str, Tuple[str, str]] = {
    "bonafide":         (DOMAIN_CLEAN, LABEL_BONAFIDE),
    "spoof":            (DOMAIN_CLEAN, LABEL_SPOOF),
    "replay_bonafide":  (DOMAIN_REPLAY, LABEL_BONAFIDE),
    "replay_spoof":     (DOMAIN_REPLAY, LABEL_SPOOF),
    "noisy_bonafide":   (DOMAIN_NOISY,  LABEL_BONAFIDE),
    "noisy_spoof":      (DOMAIN_NOISY,  LABEL_SPOOF),
    "codec_bonafide":   (DOMAIN_CODEC,  LABEL_BONAFIDE),
    "codec_spoof":      (DOMAIN_CODEC,  LABEL_SPOOF),
}


# ──────────────────────────────────────────────────────────────────────────────
# Directory scanning
# ──────────────────────────────────────────────────────────────────────────────

class FileRecord:
    """One audio file with its ground-truth metadata."""

    __slots__ = ("path", "domain", "ground_truth", "label_int")

    def __init__(self, path: Path, domain: str, ground_truth: str):
        self.path        = path
        self.domain      = domain
        self.ground_truth = ground_truth
        self.label_int   = LABEL_INT[ground_truth]

    def __repr__(self) -> str:
        return (
            f"FileRecord(path={self.path.name!r}, "
            f"domain={self.domain!r}, gt={self.ground_truth!r})"
        )


def _discover_nested(root: Path) -> List[FileRecord]:
    """
    Discover files from the new nested layout:
        root/<domain>/<label>/<files>

    <domain>: clean | replay | noisy | codec
    <label>:  bonafide | spoof
    """
    records: List[FileRecord] = []
    for domain_dir in sorted(root.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name.lower()
        if domain not in VALID_DOMAINS:
            continue
        for label_dir in sorted(domain_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name.lower()
            if label not in LABEL_INT:
                continue
            for f in sorted(label_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
                    records.append(FileRecord(f, domain, label))
    return records


def _discover_flat(root: Path) -> List[FileRecord]:
    """
    Discover files from the old flat layout:
        root/<folder_name>/<files>

    Folder names are mapped via FLAT_FOLDER_DOMAIN_MAP.
    """
    records: List[FileRecord] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        mapping = FLAT_FOLDER_DOMAIN_MAP.get(folder.name.lower())
        if mapping is None:
            continue
        domain, label = mapping
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
                records.append(FileRecord(f, domain, label))
    return records


def discover_files(root: Path) -> List[FileRecord]:
    """
    Auto-detect layout (nested vs flat) and return sorted FileRecord list.

    Detection logic:
    - If any first-level subdirectory name is in VALID_DOMAINS → nested layout.
    - Otherwise fall back to flat layout.
    """
    if not root.exists():
        raise FileNotFoundError(f"Test set root not found: {root}")

    first_level = [d.name.lower() for d in root.iterdir() if d.is_dir()]
    if any(n in VALID_DOMAINS for n in first_level):
        records = _discover_nested(root)
        layout = "nested"
    else:
        records = _discover_flat(root)
        layout = "flat"

    logger.info(
        f"Discovered {len(records)} audio files "
        f"(layout={layout}) in {root}"
    )
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Per-file inference
# ──────────────────────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def infer_short(
    rec: FileRecord,
    preprocessor,
    engine,
) -> Dict[str, Any]:
    """
    Run the standard fixed-4s inference path on one file.
    Returns a flat result dict; never raises (errors captured in 'status').
    """
    t0 = time.perf_counter()
    row: Dict[str, Any] = {
        "path":               str(rec.path),
        "domain":             rec.domain,
        "ground_truth":       rec.ground_truth,
        "label_int":          rec.label_int,
        "predicted_class":    None,
        "spoof_probability":  None,
        "decision_threshold": None,
        "confidence":         None,
        "risk_score":         None,
        "threat_level":       None,
        "action":             None,
        "latency_ms":         None,
        "duration_sec":       None,
        "model_version":      None,
        "windows_analyzed":   None,
        "window_scores":      None,
        "status":             "error",
        "error":              None,
        "mode":               "short",
    }

    try:
        # duration via soundfile (fast, no full decode)
        try:
            import soundfile as sf
            info = sf.info(str(rec.path))
            row["duration_sec"] = round(info.duration, 3)
        except Exception:
            pass

        wav = preprocessor.from_file(rec.path)
        result = engine.predict(wav)

        wall_ms = round((time.perf_counter() - t0) * 1000, 2)

        row["predicted_class"]    = result.get("classification")
        row["spoof_probability"]  = _safe_float(result.get("spoof_probability"))
        row["decision_threshold"] = _safe_float(result.get("decision_threshold"))
        row["confidence"]         = _safe_float(result.get("confidence"))
        row["risk_score"]         = _safe_float(result.get("risk_score"))
        row["threat_level"]       = result.get("threat_level")
        row["action"]             = result.get("recommended_action")
        row["latency_ms"]         = _safe_float(result.get("latency_ms")) or wall_ms
        row["model_version"]      = result.get("model_version")
        row["status"]             = result.get("status", "ok")
        row["error"]              = result.get("error")

    except Exception as exc:
        row["status"] = "error"
        row["error"]  = str(exc)
        logger.debug(f"Error on {rec.path}: {exc}")

    return row


def infer_long(
    rec: FileRecord,
    engine,
    window_sec: float,
    hop_sec: float,
) -> Dict[str, Any]:
    """
    Run the full-audio sliding-window inference path on one file.
    Returns a flat result dict; never raises.
    """

    t0 = time.perf_counter()
    row: Dict[str, Any] = {
        "path":               str(rec.path),
        "domain":             rec.domain,
        "ground_truth":       rec.ground_truth,
        "label_int":          rec.label_int,
        "predicted_class":    None,
        "spoof_probability":  None,
        "decision_threshold": None,
        "confidence":         None,
        "risk_score":         None,
        "threat_level":       None,
        "action":             None,
        "latency_ms":         None,
        "duration_sec":       None,
        "model_version":      None,
        "windows_analyzed":   None,
        "window_scores":      None,
        "status":             "error",
        "error":              None,
        "mode":               "long",
    }

    try:
        file_bytes = rec.path.read_bytes()
        wav = decode_full_audio(file_bytes, rec.path.suffix)

        analyzer = LongAudioAnalyzer(engine, window_sec=window_sec, hop_sec=hop_sec)
        result = analyzer.analyze(wav)

        wall_ms = round((time.perf_counter() - t0) * 1000, 2)

        row["predicted_class"]    = result.get("classification")
        row["spoof_probability"]  = _safe_float(result.get("spoof_probability"))
        row["decision_threshold"] = _safe_float(result.get("decision_threshold"))
        row["confidence"]         = _safe_float(result.get("confidence"))
        row["risk_score"]         = _safe_float(result.get("risk_score"))
        row["threat_level"]       = result.get("threat_level")
        row["action"]             = result.get("recommended_action")
        row["latency_ms"]         = wall_ms
        row["duration_sec"]       = _safe_float(result.get("audio_duration_sec"))
        row["model_version"]      = result.get("model_version")
        row["windows_analyzed"]   = result.get("windows_analyzed")
        row["status"]             = result.get("status", "ok")
        row["error"]              = result.get("error")

        # Store per-window spoof scores as JSON string (compact)
        agg = result.get("aggregation", {})
        scores_list = agg.get("all_window_scores")
        if scores_list is not None:
            row["window_scores"] = json.dumps(scores_list)

    except Exception as exc:
        row["status"] = "error"
        row["error"]  = str(exc)
        logger.debug(f"Error on {rec.path}: {exc}")

    return row


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Return (eer, eer_threshold). Requires at least one sample of each class."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def compute_metrics(
    rows: List[Dict[str, Any]],
    threshold: float,
    note: str = "",
) -> Dict[str, Any]:
    """
    Compute the full metric suite for a set of rows.

    Only rows with status='ok' and numeric spoof_probability are used.
    ERROR rows are counted separately and excluded from all metrics.
    """
    valid = [
        r for r in rows
        if r.get("status") == "ok"
        and r.get("spoof_probability") is not None
        and r.get("ground_truth") in LABEL_INT
    ]
    errors = [r for r in rows if r.get("status") != "ok"]

    n_total  = len(rows)
    n_valid  = len(valid)
    n_errors = len(errors)

    base: Dict[str, Any] = {
        "note":                note,
        "n_total":             n_total,
        "n_valid":             n_valid,
        "n_errors":            n_errors,
        "threshold_used":      round(threshold, 7),
        "threshold_source":    "checkpoint_validation_eer",
    }

    if n_valid == 0:
        base.update({
            "n_bonafide": 0, "n_spoof": 0,
            "roc_auc": None, "eer": None, "eer_threshold": None,
            "accuracy": None, "precision": None, "recall": None, "f1": None,
            "fpr": None, "fnr": None,
            "confusion_matrix": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
            "mean_latency_ms": None, "p50_latency_ms": None, "p95_latency_ms": None,
        })
        return base

    labels = np.array([r["label_int"] for r in valid], dtype=int)
    scores = np.array([r["spoof_probability"] for r in valid], dtype=float)
    preds  = (scores >= threshold).astype(int)

    n_bonafide = int((labels == 0).sum())
    n_spoof    = int((labels == 1).sum())

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    accuracy  = (tp + tn) / n_valid
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fpr_val   = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val   = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # ROC-AUC and EER require both classes
    roc_auc = eer = eer_threshold = None
    if n_bonafide > 0 and n_spoof > 0:
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = round(float(roc_auc_score(labels, scores)), 6)
            eer_val, eer_thr = _eer(labels, scores)
            eer          = round(eer_val, 6)
            eer_threshold = round(eer_thr, 6)
        except Exception as e:
            logger.warning(f"ROC/EER computation failed: {e}")

    # Latency
    lats = [r["latency_ms"] for r in valid if r.get("latency_ms") is not None]
    mean_lat = p50_lat = p95_lat = None
    if lats:
        lats_arr = np.array(lats, dtype=float)
        mean_lat = round(float(lats_arr.mean()), 2)
        p50_lat  = round(float(np.percentile(lats_arr, 50)), 2)
        p95_lat  = round(float(np.percentile(lats_arr, 95)), 2)

    base.update({
        "n_bonafide":     n_bonafide,
        "n_spoof":        n_spoof,
        "roc_auc":        roc_auc,
        "eer":            eer,
        "eer_threshold":  eer_threshold,
        "accuracy":       round(accuracy, 6),
        "precision":      round(precision, 6),
        "recall":         round(recall, 6),
        "f1":             round(f1, 6),
        "fpr":            round(fpr_val, 6),
        "fnr":            round(fnr_val, 6),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "mean_latency_ms": mean_lat,
        "p50_latency_ms":  p50_lat,
        "p95_latency_ms":  p95_lat,
    })
    return base


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation strategy comparison (evaluation-only, does not affect deployment)
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_window_scores(
    scores: List[float],
    strategy: str,
) -> float:
    """Apply one aggregation strategy to a list of window-level spoof scores."""
    if not scores:
        return 0.0
    arr = sorted(scores, reverse=True)  # descending

    if strategy == "mean":
        return float(np.mean(arr))
    elif strategy == "median":
        return float(np.median(arr))
    elif strategy == "max":
        return float(arr[0])
    elif strategy == "top10_mean":
        k = max(1, math.ceil(len(arr) * 0.10))
        return float(np.mean(arr[:k]))
    elif strategy == "top25_mean":
        k = max(1, math.ceil(len(arr) * 0.25))
        return float(np.mean(arr[:k]))
    elif strategy == "top50_mean":
        k = max(1, math.ceil(len(arr) * 0.50))
        return float(np.mean(arr[:k]))
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")


AGGREGATION_STRATEGIES = [
    "mean", "median", "max",
    "top10_mean", "top25_mean", "top50_mean",
]


def aggregation_experiment(
    long_rows: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    """
    Compare aggregation strategies on the long-mode results.

    Each row must have 'window_scores' (JSON-encoded list of floats).
    Rows without window_scores are skipped.

    Returns a dict with per-strategy metrics and a 'best_strategy' key.

    NOTE: This is evaluation-only analysis.
          It does NOT change the deployed /predict-long behaviour.
    """
    eligible = [
        r for r in long_rows
        if r.get("status") == "ok"
        and r.get("window_scores") is not None
        and r.get("ground_truth") in LABEL_INT
    ]

    if not eligible:
        return {
            "note": "No eligible rows (need long mode + valid window_scores)",
            "strategies": {},
            "best_strategy": None,
        }

    labels = np.array([LABEL_INT[r["ground_truth"]] for r in eligible])
    results: Dict[str, Any] = {}

    for strategy in AGGREGATION_STRATEGIES:
        agg_scores = []
        for r in eligible:
            try:
                ws = json.loads(r["window_scores"])
            except Exception:
                ws = []
            agg_scores.append(_aggregate_window_scores(ws, strategy))

        agg_arr = np.array(agg_scores, dtype=float)
        preds   = (agg_arr >= threshold).astype(int)

        n = len(labels)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())

        accuracy  = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        fpr_v     = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr_v     = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        roc_auc = eer = None
        n_bf = int((labels == 0).sum())
        n_sp = int((labels == 1).sum())
        if n_bf > 0 and n_sp > 0:
            try:
                from sklearn.metrics import roc_auc_score
                roc_auc = round(float(roc_auc_score(labels, agg_arr)), 6)
                eer_v, _ = _eer(labels, agg_arr)
                eer = round(eer_v, 6)
            except Exception:
                pass

        results[strategy] = {
            "accuracy":  round(accuracy, 6),
            "f1":        round(f1, 6),
            "precision": round(precision, 6),
            "recall":    round(recall, 6),
            "roc_auc":   roc_auc,
            "eer":       eer,
            "fpr":       round(fpr_v, 6),
            "fnr":       round(fnr_v, 6),
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }

    # Best strategy = highest F1 (most balanced); tie-break by lowest EER
    def _sort_key(item: Tuple[str, Dict]) -> Tuple[float, float]:
        s = item[1]
        f1_v  = s["f1"] if s["f1"] is not None else -1.0
        eer_v = s["eer"] if s["eer"] is not None else 1.0
        return (-f1_v, eer_v)

    ranked = sorted(results.items(), key=_sort_key)
    best_strategy = ranked[0][0] if ranked else None

    return {
        "note": (
            "Evaluation-only comparison. Does NOT change deployed inference. "
            "n_eligible_files=" + str(len(eligible))
        ),
        "n_eligible": len(eligible),
        "strategies": results,
        "best_strategy": best_strategy,
        "ranking": [s for s, _ in ranked],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report writers
# ──────────────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "path", "domain", "ground_truth", "label_int",
    "predicted_class", "spoof_probability", "decision_threshold",
    "confidence", "risk_score", "threat_level", "action",
    "latency_ms", "duration_sec", "model_version",
    "windows_analyzed", "window_scores",
    "status", "error", "mode",
]


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(report: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


def _fmt(v: Any, decimals: int = 4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    """Generate a human-readable Markdown evaluation report."""
    meta = report.get("metadata", {})
    overall = report.get("overall_metrics", {})
    domain_metrics = report.get("domain_metrics", {})
    agg_exp = report.get("aggregation_experiment", {})

    lines: List[str] = [
        "# VoxShield External Evaluation Report",
        "",
        f"**Generated**: {meta.get('evaluation_timestamp', 'N/A')}  ",
        f"**Mode**: {meta.get('mode', 'N/A')}  ",
        f"**Checkpoint SHA-256** (first 16): `{meta.get('checkpoint_sha256', 'N/A')}`  ",
        f"**Model version**: {meta.get('model_version', 'N/A')}  ",
        f"**Epoch**: {meta.get('checkpoint_epoch', 'N/A')}  ",
        f"**Threshold** (from checkpoint, not tuned here): `{meta.get('threshold', 'N/A')}`  ",
        f"**Threshold source**: {meta.get('threshold_source', 'N/A')}  ",
        f"**Test set directory**: `{meta.get('testset_dir', 'N/A')}`  ",
        f"**Files discovered**: {meta.get('n_files_discovered', 'N/A')}  ",
        f"**Files valid**: {meta.get('n_files_valid', 'N/A')}  ",
        f"**Files with errors**: {meta.get('n_files_error', 'N/A')}  ",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Files total | {overall.get('n_total', 'N/A')} |",
        f"| Files valid | {overall.get('n_valid', 'N/A')} |",
        f"| Files error | {overall.get('n_errors', 'N/A')} |",
        f"| Bonafide samples | {overall.get('n_bonafide', 'N/A')} |",
        f"| Spoof samples | {overall.get('n_spoof', 'N/A')} |",
        f"| **ROC-AUC** | **{_fmt(overall.get('roc_auc'))}** |",
        f"| **EER** | **{_fmt(overall.get('eer'))}** |",
        f"| EER threshold | {_fmt(overall.get('eer_threshold'))} |",
        f"| Accuracy | {_fmt(overall.get('accuracy'))} |",
        f"| Precision | {_fmt(overall.get('precision'))} |",
        f"| Recall | {_fmt(overall.get('recall'))} |",
        f"| F1 Score | {_fmt(overall.get('f1'))} |",
        f"| FPR (False Positive Rate) | {_fmt(overall.get('fpr'))} |",
        f"| FNR (False Negative Rate) | {_fmt(overall.get('fnr'))} |",
        f"| Mean latency (ms) | {_fmt(overall.get('mean_latency_ms'), 2)} |",
        f"| p50 latency (ms) | {_fmt(overall.get('p50_latency_ms'), 2)} |",
        f"| p95 latency (ms) | {_fmt(overall.get('p95_latency_ms'), 2)} |",
        "",
        "**Confusion matrix**",
        "",
        "```",
    ]
    cm = overall.get("confusion_matrix", {})
    lines += [
        f"  TP (spoof correctly detected):    {cm.get('tp', 'N/A')}",
        f"  FP (bonafide misclassified):      {cm.get('fp', 'N/A')}",
        f"  FN (spoof missed):                {cm.get('fn', 'N/A')}",
        f"  TN (bonafide correctly accepted): {cm.get('tn', 'N/A')}",
        "```",
        "",
        "---",
        "",
        "## Per-Domain Metrics",
        "",
        "| Domain | N | Valid | Errors | Accuracy | F1 | ROC-AUC | EER | FPR | FNR |",
        "|--------|---|-------|--------|----------|----|---------|-----|-----|-----|",
    ]

    for domain in sorted(domain_metrics):
        m = domain_metrics[domain]
        lines.append(
            f"| {domain} "
            f"| {m.get('n_total','N/A')} "
            f"| {m.get('n_valid','N/A')} "
            f"| {m.get('n_errors','N/A')} "
            f"| {_fmt(m.get('accuracy'))} "
            f"| {_fmt(m.get('f1'))} "
            f"| {_fmt(m.get('roc_auc'))} "
            f"| {_fmt(m.get('eer'))} "
            f"| {_fmt(m.get('fpr'))} "
            f"| {_fmt(m.get('fnr'))} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Aggregation Strategy Comparison",
        "",
        "> **Note**: This comparison is evaluation-only.",
        "> It does NOT change the deployed `/predict-long` behaviour.",
        "",
    ]

    strats = agg_exp.get("strategies", {})
    best   = agg_exp.get("best_strategy")
    n_elig = agg_exp.get("n_eligible", 0)
    lines.append(f"Eligible files (long mode with per-window scores): **{n_elig}**")
    lines.append("")

    if strats:
        lines += [
            "| Strategy | Accuracy | F1 | ROC-AUC | EER | FPR | FNR |",
            "|----------|----------|----|---------|-----|-----|-----|",
        ]
        for s in AGGREGATION_STRATEGIES:
            m = strats.get(s, {})
            marker = " ← best" if s == best else ""
            lines.append(
                f"| {s}{marker} "
                f"| {_fmt(m.get('accuracy'))} "
                f"| {_fmt(m.get('f1'))} "
                f"| {_fmt(m.get('roc_auc'))} "
                f"| {_fmt(m.get('eer'))} "
                f"| {_fmt(m.get('fpr'))} "
                f"| {_fmt(m.get('fnr'))} |"
            )
        lines.append(f"\n**Best strategy** (by F1, tie-break EER): `{best}`")
    else:
        lines.append(
            "_No aggregation comparison available "
            "(run with --mode long to populate)._"
        )

    lines += [
        "",
        "---",
        "",
        "## Disclaimer",
        "",
        "> These results reflect the model's performance on the external test set.",
        "> The decision threshold comes from the training checkpoint (validated on",
        "> ASVspoof5 speaker-disjoint validation data) and was NOT re-tuned here.",
        "> Results on a small external set are indicative only — not production guarantees.",
        "> VoxShieldNet is trained from random initialization on ASVspoof5.",
        "> Performance may differ significantly on out-of-distribution audio.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def _checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _runtime_versions() -> Dict[str, str]:
    return {
        "python":   platform.python_version(),
        "torch":    torch.__version__,
        "platform": platform.platform(),
        "numpy":    np.__version__,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    testset_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    mode: str = "short",
    window_sec: float = 4.0,
    hop_sec: float = 2.0,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full evaluation pipeline.

    Args:
        testset_dir:     Root directory of labeled test audio.
        checkpoint_path: Path to trained .pt checkpoint.
        output_dir:      Where to write reports.
        mode:            "short" or "long".
        window_sec:      Window length for long mode.
        hop_sec:         Hop for long mode.
        device_str:      Device override ("cpu", "cuda", etc.).

    Returns:
        The full report dict (also written to disk).
    """
    from inference.engine import InferenceEngine
    from inference.preprocessor import AudioPreprocessor

    ts = datetime.now(timezone.utc).isoformat()
    logger.info(f"External evaluation started at {ts}")
    logger.info(f"Mode: {mode}  | Testset: {testset_dir}")

    # ── Load model ────────────────────────────────────────────────────────────
    engine = InferenceEngine(
        project_root=PROJECT_ROOT,
        device=device_str,
    )
    if not engine.is_ready():
        raise RuntimeError(
            f"Model not ready — checkpoint not found or failed to load. "
            f"Expected: {checkpoint_path}"
        )

    threshold = float(engine.threshold)
    logger.info(
        f"Threshold={threshold:.7f} (from checkpoint, NOT tuned on this data)"
    )

    preprocessor = AudioPreprocessor() if mode == "short" else None

    # ── Checkpoint metadata ───────────────────────────────────────────────────
    ckpt_sha = _checkpoint_sha256(checkpoint_path)
    ckpt_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_version = (
        ckpt_raw.get("metadata", {}).get("version")
        or ckpt_raw.get("model_config", {}).get("version", "1.0.0")
    )
    checkpoint_epoch = ckpt_raw.get("epoch", "?")

    metadata = {
        "evaluation_timestamp":   ts,
        "mode":                   mode,
        "window_sec":             window_sec if mode == "long" else None,
        "hop_sec":                hop_sec    if mode == "long" else None,
        "testset_dir":            str(testset_dir),
        "checkpoint_path":        str(checkpoint_path),
        "checkpoint_sha256":      ckpt_sha[:16],
        "checkpoint_sha256_full": ckpt_sha,
        "checkpoint_epoch":       checkpoint_epoch,
        "model_version":          model_version,
        "threshold":              threshold,
        "threshold_source":       "checkpoint_validation_eer",
        "runtime_versions":       _runtime_versions(),
    }

    # ── Discover files ────────────────────────────────────────────────────────
    records = discover_files(testset_dir)
    metadata["n_files_discovered"] = len(records)

    if not records:
        logger.warning("No audio files found in the test set directory.")

    # ── Run inference ─────────────────────────────────────────────────────────
    all_rows: List[Dict[str, Any]] = []
    n_done = 0

    for rec in records:
        if mode == "short":
            row = infer_short(rec, preprocessor, engine)
        else:
            row = infer_long(rec, engine, window_sec=window_sec, hop_sec=hop_sec)

        all_rows.append(row)
        n_done += 1
        if n_done % 10 == 0 or n_done == len(records):
            logger.info(f"  Processed {n_done}/{len(records)}")

    n_valid = sum(1 for r in all_rows if r.get("status") == "ok")
    n_error = sum(1 for r in all_rows if r.get("status") != "ok")
    metadata["n_files_valid"] = n_valid
    metadata["n_files_error"] = n_error

    logger.info(
        f"Inference complete: {n_valid} valid, {n_error} errors "
        f"(out of {len(records)})"
    )

    # ── Overall metrics ───────────────────────────────────────────────────────
    overall = compute_metrics(all_rows, threshold, note="overall")

    # ── Per-domain metrics ────────────────────────────────────────────────────
    domains = sorted({r["domain"] for r in all_rows})
    domain_metrics: Dict[str, Any] = {}
    for domain in domains:
        d_rows = [r for r in all_rows if r["domain"] == domain]
        domain_metrics[domain] = compute_metrics(
            d_rows, threshold, note=domain
        )

    # Convenience groupings
    clean_rows  = [r for r in all_rows if r["domain"] == DOMAIN_CLEAN]
    replay_rows = [r for r in all_rows if r["domain"] == DOMAIN_REPLAY]
    noisy_rows  = [r for r in all_rows if r["domain"] == DOMAIN_NOISY]
    codec_rows  = [r for r in all_rows if r["domain"] == DOMAIN_CODEC]

    grouped_metrics = {
        "clean":  compute_metrics(clean_rows,  threshold, "clean"),
        "replay": compute_metrics(replay_rows, threshold, "replay"),
        "noisy":  compute_metrics(noisy_rows,  threshold, "noisy"),
        "codec":  compute_metrics(codec_rows,  threshold, "codec"),
    }

    # ── Aggregation experiment (long mode only) ───────────────────────────────
    agg_experiment: Dict[str, Any]
    if mode == "long":
        agg_experiment = aggregation_experiment(all_rows, threshold)
    else:
        agg_experiment = {
            "note": "Run with --mode long to enable aggregation comparison.",
            "strategies": {},
            "best_strategy": None,
        }

    # ── Assemble report ───────────────────────────────────────────────────────
    report: Dict[str, Any] = {
        "metadata":              metadata,
        "overall_metrics":       overall,
        "domain_metrics":        domain_metrics,
        "grouped_metrics":       grouped_metrics,
        "aggregation_experiment": agg_experiment,
        "disclaimer": (
            "Results from external evaluation on user-collected audio. "
            "Threshold from checkpoint (validation EER) was NOT re-tuned. "
            "This is NOT a production accuracy guarantee. "
            "VoxShieldNet trained from random initialization on ASVspoof5."
        ),
    }

    # ── Write reports ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "external_evaluation.json"
    csv_path  = output_dir / "external_evaluation.csv"
    md_path   = output_dir / "external_evaluation.md"

    write_json(report, json_path)
    write_csv(all_rows, csv_path)
    write_markdown(report, md_path)

    logger.info(f"Reports written to {output_dir}:")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  CSV:  {csv_path}")
    logger.info(f"  MD:   {md_path}")

    return report


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(report: Dict[str, Any]) -> None:
    """Print a concise summary to stdout."""
    meta  = report.get("metadata", {})
    ov    = report.get("overall_metrics", {})
    dm    = report.get("domain_metrics", {})
    agg   = report.get("aggregation_experiment", {})
    best  = agg.get("best_strategy")

    print()
    print("=" * 68)
    print("VoxShield External Evaluation Summary")
    print("=" * 68)
    print(f"  Timestamp     : {meta.get('evaluation_timestamp','?')}")
    print(f"  Mode          : {meta.get('mode','?')}")
    print(f"  Checkpoint    : sha256={meta.get('checkpoint_sha256','?')}...")
    print(f"  Model version : {meta.get('model_version','?')}, epoch={meta.get('checkpoint_epoch','?')}")
    print(f"  Threshold     : {meta.get('threshold','?')} (checkpoint val EER — NOT tuned here)")
    print(f"  Files         : {meta.get('n_files_discovered','?')} discovered, "
          f"{meta.get('n_files_valid','?')} valid, "
          f"{meta.get('n_files_error','?')} errors")
    print()
    print("  OVERALL METRICS (valid files only, errors excluded)")
    print(f"    n_valid   : {ov.get('n_valid','N/A')}")
    print(f"    n_bonafide: {ov.get('n_bonafide','N/A')}")
    print(f"    n_spoof   : {ov.get('n_spoof','N/A')}")
    print(f"    ROC-AUC   : {_fmt(ov.get('roc_auc'))}")
    print(f"    EER       : {_fmt(ov.get('eer'))}")
    print(f"    Accuracy  : {_fmt(ov.get('accuracy'))}")
    print(f"    F1        : {_fmt(ov.get('f1'))}")
    print(f"    FPR       : {_fmt(ov.get('fpr'))}")
    print(f"    FNR       : {_fmt(ov.get('fnr'))}")
    print(f"    p95 lat   : {_fmt(ov.get('p95_latency_ms'), 1)} ms")
    print()
    if dm:
        print("  PER-DOMAIN")
        print(f"    {'domain':10s}  {'n':>5s}  {'ok':>4s}  {'err':>4s}  "
              f"{'acc':>7s}  {'F1':>7s}  {'AUC':>7s}  {'EER':>7s}")
        for domain in sorted(dm):
            m = dm[domain]
            print(
                f"    {domain:10s}  "
                f"{m.get('n_total','?'):>5}  "
                f"{m.get('n_valid','?'):>4}  "
                f"{m.get('n_errors','?'):>4}  "
                f"{_fmt(m.get('accuracy')):>7}  "
                f"{_fmt(m.get('f1')):>7}  "
                f"{_fmt(m.get('roc_auc')):>7}  "
                f"{_fmt(m.get('eer')):>7}"
            )
    print()
    if best:
        strat_m = agg.get("strategies", {}).get(best, {})
        print(f"  BEST AGGREGATION STRATEGY: {best}")
        print(f"    F1={_fmt(strat_m.get('f1'))}  "
              f"AUC={_fmt(strat_m.get('roc_auc'))}  "
              f"EER={_fmt(strat_m.get('eer'))}")
    print("=" * 68)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "VoxShield external evaluation harness. "
            "Evaluates the production checkpoint on labeled audio directories."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--testset",
        default="evaluation/external_testset",
        help="Root directory of labeled test audio.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/asvspoof5/best.pt",
        help="Path to the trained checkpoint (.pt file).",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/reports",
        help="Directory for evaluation reports.",
    )
    parser.add_argument(
        "--mode",
        choices=["short", "long"],
        default="short",
        help="short: fixed 4s window (standard /predict path). "
             "long: sliding-window /predict-long path.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=4.0,
        help="Window length in seconds (long mode only).",
    )
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=2.0,
        help="Hop between windows in seconds (long mode only).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override: cpu, cuda, cuda:0, etc.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    report = run_evaluation(
        testset_dir=Path(args.testset).resolve(),
        checkpoint_path=Path(args.checkpoint).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        mode=args.mode,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        device_str=args.device,
    )
    _print_summary(report)
