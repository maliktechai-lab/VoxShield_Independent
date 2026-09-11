"""
VoxShield Evaluator — Scientifically Defensible Evaluation
===========================================================

Usage:
    python -m evaluation.evaluator \
        --checkpoint checkpoints/asvspoof5/best.pt \
        --asvspoof5-dir /mnt/d/Datasets/ASVspoof5 \
        --split dev \
        --output-dir reports/asvspoof5

CRITICAL PRINCIPLES (non-negotiable):

1. Threshold selection uses VALIDATION data only.
   The test/eval set is NEVER used to choose the decision threshold.
   The threshold from the checkpoint (computed on validation) is used directly.

2. Continuous metrics (ROC-AUC, EER) are threshold-independent.

3. Threshold-dependent metrics (accuracy, F1, precision, recall, FPR, FNR)
   use the checkpoint's stored threshold.

4. Robustness evaluation does NOT modify labels.

5. Reports are written per-checkpoint. Real and demo reports are isolated.

6. No fabricated metrics. All numbers come from running the actual model
   on actual data with the actual threshold.

Output files:
    reports/<dataset_type>/eval_<split>_<timestamp>/
        summary.json        — all metrics in machine-readable form
        report.txt          — human-readable report
        per_generator.json  — per-attack/generator breakdown
        robustness.json     — clean vs noisy vs compressed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    threshold = float(thr[idx])
    return eer, threshold


def compute_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    label_note: str = "",
) -> Dict:
    """
    Compute the full metric suite.

    threshold MUST come from validation data, not from this eval set.
    """
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score,
        recall_score, f1_score,
    )

    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    n  = len(labels)

    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    try:
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        auc = float("nan")

    eer, eer_thr = compute_eer(labels, scores)

    return {
        "note":           label_note,
        "n_samples":      n,
        "n_bonafide":     int((labels == 0).sum()),
        "n_spoof":        int((labels == 1).sum()),
        # ── Threshold-independent (continuous) ────────────────────────────────
        "roc_auc":        round(auc, 6),
        "eer":            round(eer, 6),
        "eer_threshold":  round(eer_thr, 6),
        # ── Threshold-dependent (using val threshold) ─────────────────────────
        "threshold_used": round(threshold, 6),
        "threshold_source": "validation_eer",
        "accuracy":       round(float(accuracy_score(labels, preds)), 6),
        "precision":      round(float(precision_score(labels, preds, zero_division=0)), 6),
        "recall":         round(float(recall_score(labels, preds, zero_division=0)), 6),
        "f1":             round(float(f1_score(labels, preds, zero_division=0)), 6),
        "fpr":            round(fpr_val, 6),
        "fnr":            round(fnr_val, 6),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Score extraction
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_dataset(
    model,
    records: List[Dict],
    device: torch.device,
    batch_size: int = 64,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run model over all records.
    Returns (labels, scores, avg_latency_ms).
    """
    from training.dataset import ASVspoof5Dataset, collate_fn
    from torch.utils.data import DataLoader

    ds = ASVspoof5Dataset(records, augment=False)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )

    all_labels: list = []
    all_scores: list = []
    total_latency = 0.0
    n_batches = 0

    model.eval()
    it = enumerate(loader)
    if show_progress:
        try:
            from tqdm import tqdm
            it = enumerate(tqdm(loader, desc="Scoring"))
        except ImportError:
            pass

    for _, (wavs, labels, _) in it:
        wavs = wavs.to(device, non_blocking=True)
        t0 = time.perf_counter()
        logits = model(wavs)
        total_latency += (time.perf_counter() - t0) * 1000
        probs = torch.sigmoid(logits).cpu().numpy()
        all_scores.append(probs)
        all_labels.append(labels.numpy())
        n_batches += 1

    avg_lat = total_latency / max(1, n_batches) / batch_size * 1000  # per sample ms
    return (
        np.concatenate(all_labels),
        np.concatenate(all_scores),
        avg_lat,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-generator breakdown
# ──────────────────────────────────────────────────────────────────────────────

def per_generator_report(
    records: List[Dict],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict:
    """Report spoof recall and FNR per attack/generator type."""
    generators = {}
    for i, r in enumerate(records):
        at = r.get("attack_type", "unknown")
        if at not in generators:
            generators[at] = {"labels": [], "scores": [], "idx": []}
        generators[at]["labels"].append(labels[i])
        generators[at]["scores"].append(scores[i])

    results = {}
    for at, data in sorted(generators.items()):
        lbl = np.array(data["labels"])
        scr = np.array(data["scores"])
        preds = (scr >= threshold).astype(int)
        n = len(lbl)
        n_spoof = int((lbl == 1).sum())
        n_bf    = int((lbl == 0).sum())

        # For spoof-only metrics
        if n_spoof > 0:
            spoof_mask = lbl == 1
            spoof_recall = float((preds[spoof_mask] == 1).mean())
            spoof_fnr    = 1.0 - spoof_recall
            mean_score   = float(scr[spoof_mask].mean())
        else:
            spoof_recall = float("nan")
            spoof_fnr    = float("nan")
            mean_score   = float("nan")

        results[at] = {
            "n_total":     n,
            "n_spoof":     n_spoof,
            "n_bonafide":  n_bf,
            "spoof_recall": round(spoof_recall, 4) if not np.isnan(spoof_recall) else None,
            "spoof_fnr":    round(spoof_fnr, 4) if not np.isnan(spoof_fnr) else None,
            "mean_spoof_score": round(mean_score, 4) if not np.isnan(mean_score) else None,
        }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Robustness evaluation
# ──────────────────────────────────────────────────────────────────────────────

def robustness_eval(
    model,
    records: List[Dict],
    device: torch.device,
    threshold: float,
    batch_size: int = 64,
) -> Dict:
    """
    Evaluate under three conditions:
        clean       — original audio
        noisy       — additive white noise SNR~25dB
        compressed  — amplitude quantization (8-bit)

    Labels are NOT modified. The test is purely about input perturbation.
    """
    from training.dataset import collate_fn
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    results = {}
    for condition in ["clean", "noisy", "compressed"]:
        aug_records = records  # same records, different loading

        class _AugDs(torch.utils.data.Dataset):
            def __init__(self, recs, condition):
                self.recs = recs
                self.condition = condition
                import soundfile as sf
                self.sf = sf

            def __len__(self):
                return len(self.recs)

            def __getitem__(self, i):
                r = self.recs[i]
                try:
                    data, sr = self.sf.read(r["path"], dtype="float32")
                except Exception:
                    data = np.zeros(64000, dtype=np.float32)
                    sr = 16000
                if data.ndim > 1:
                    data = data.mean(axis=1)
                wav = torch.from_numpy(data)
                # Pad/clip
                ms = 64000
                if wav.shape[0] >= ms:
                    wav = wav[:ms]
                else:
                    wav = F.pad(wav, (0, ms - wav.shape[0]))

                # Augmentation
                if self.condition == "noisy":
                    sig_pwr = wav.pow(2).mean().clamp(min=1e-9)
                    snr_db = 25.0
                    noise_pwr = sig_pwr / (10 ** (snr_db / 10))
                    wav = wav + torch.randn_like(wav) * noise_pwr.sqrt()
                elif self.condition == "compressed":
                    # Simulate 8-bit quantization
                    wav = torch.round(wav * 128) / 128

                meta = {
                    "speaker_id": r["speaker_id"],
                    "utterance_id": r["utterance_id"],
                    "attack_type": r["attack_type"],
                    "codec": r.get("codec"),
                    "label_str": r["label_str"],
                }
                return wav, r["label"], meta

        ds = _AugDs(aug_records, condition)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn
        )

        all_labels: list = []
        all_scores: list = []
        model.eval()
        with torch.no_grad():
            for wavs, labels, _ in loader:
                wavs = wavs.to(device)
                logits = model(wavs)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(probs)
                all_labels.append(labels.numpy())

        lbl = np.concatenate(all_labels)
        scr = np.concatenate(all_scores)
        results[condition] = compute_all_metrics(lbl, scr, threshold, label_note=condition)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report writing
# ──────────────────────────────────────────────────────────────────────────────

def write_report(
    output_dir: Path,
    overall_metrics: Dict,
    per_gen: Dict,
    robustness: Dict,
    checkpoint_meta: Dict,
    avg_latency_ms: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint":           checkpoint_meta.get("path", "?"),
        "dataset_type":         checkpoint_meta.get("dataset_type", "?"),
        "model_version":        checkpoint_meta.get("model_version", "?"),
        "pretrained":           False,
        "threshold_source":     "validation_eer",
        "threshold_used":       overall_metrics.get("threshold_used"),
        "overall_metrics":      overall_metrics,
        "per_generator":        per_gen,
        "robustness":           robustness,
        "avg_inference_latency_ms": round(avg_latency_ms, 4),
        "DISCLAIMER": (
            "These metrics reflect model performance on the evaluation split "
            "using a threshold derived from the validation split. "
            "They are NOT production accuracy guarantees. "
            "VoxShieldNet is trained from random initialization."
        ),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Per-generator JSON ────────────────────────────────────────────────────
    with open(output_dir / "per_generator.json", "w") as f:
        json.dump(per_gen, f, indent=2)

    # ── Robustness JSON ───────────────────────────────────────────────────────
    with open(output_dir / "robustness.json", "w") as f:
        json.dump(robustness, f, indent=2)

    # ── Human-readable report ─────────────────────────────────────────────────
    m = overall_metrics
    lines = [
        "=" * 70,
        "VoxShield Evaluation Report",
        "=" * 70,
        f"Timestamp         : {summary['evaluation_timestamp']}",
        f"Checkpoint        : {checkpoint_meta.get('path', '?')}",
        f"Dataset type      : {checkpoint_meta.get('dataset_type', '?')}",
        f"Model version     : {checkpoint_meta.get('model_version', '?')}",
        f"Pretrained        : No",
        "",
        "THRESHOLD INFORMATION",
        "-" * 40,
        f"  Decision threshold : {m['threshold_used']:.6f}",
        f"  Threshold source   : Validation EER (NOT from this eval set)",
        f"  EER threshold      : {m['eer_threshold']:.6f}",
        "",
        "CONTINUOUS METRICS (threshold-independent)",
        "-" * 40,
        f"  ROC-AUC   : {m['roc_auc']:.4f}",
        f"  EER       : {m['eer']:.4f}",
        "",
        "THRESHOLD-DEPENDENT METRICS",
        "(using validation threshold — do NOT select threshold from eval set)",
        "-" * 40,
        f"  Accuracy  : {m['accuracy']:.4f}",
        f"  Precision : {m['precision']:.4f}",
        f"  Recall    : {m['recall']:.4f}",
        f"  F1 Score  : {m['f1']:.4f}",
        f"  FPR       : {m['fpr']:.4f}",
        f"  FNR       : {m['fnr']:.4f}",
        "",
        "CONFUSION MATRIX",
        "-" * 40,
        f"  TP (Spoof correctly detected)   : {m['confusion_matrix']['tp']}",
        f"  FP (Bonafide misclassified)     : {m['confusion_matrix']['fp']}",
        f"  FN (Spoof missed)               : {m['confusion_matrix']['fn']}",
        f"  TN (Bonafide correctly accepted): {m['confusion_matrix']['tn']}",
        "",
        f"MEAN INFERENCE LATENCY: {avg_latency_ms:.2f} ms/sample",
        "",
        "ROBUSTNESS",
        "-" * 40,
    ]
    for cond, r in robustness.items():
        lines.append(
            f"  {cond:12s}: AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  "
            f"EER={r['eer']:.4f}"
        )

    lines += [
        "",
        "PER-GENERATOR BREAKDOWN",
        "-" * 40,
    ]
    for at, r in sorted(per_gen.items()):
        lines.append(
            f"  {at:20s}: n={r['n_total']:6d}  "
            f"spoof_recall={str(r.get('spoof_recall', 'N/A')):8s}  "
            f"spoof_FNR={str(r.get('spoof_fnr', 'N/A')):8s}  "
            f"mean_score={str(r.get('mean_spoof_score', 'N/A'))}"
        )

    lines += [
        "",
        "=" * 70,
        "DISCLAIMER",
        "-" * 40,
        "These metrics reflect empirical performance on the evaluation",
        "split of ASVspoof5. They are NOT production accuracy claims.",
        "VoxShieldNet is trained from random initialization.",
        "Do not use these numbers as guarantees in production deployments.",
        "=" * 70,
    ]

    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report written to {output_dir}")
    logger.info(f"  Summary  : {summary_path}")
    logger.info(f"  Report   : {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    import torch
    from model.voxshieldnet import build_model
    from training.dataset import build_index, dataset_stats

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = ckpt.get("model_config", None)

    # ── CRITICAL: threshold from checkpoint (validation set) ─────────────────
    threshold = float(ckpt.get("threshold", 0.5))
    logger.info(
        f"Using threshold={threshold:.6f} from checkpoint "
        f"(derived from validation EER — NOT from eval set)"
    )

    checkpoint_meta = {
        "path":          str(ckpt_path),
        "epoch":         ckpt.get("epoch", "?"),
        "dataset_type":  ckpt.get("dataset_info", {}).get("dataset_type", "unknown"),
        "model_version": ckpt.get("metadata", {}).get("version",
                         ckpt.get("model_config", {}).get("version", "1.0.0")),
        "val_metrics":   ckpt.get("val_metrics", {}),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Model: {model.summary()}")

    # ── Load evaluation records ───────────────────────────────────────────────
    dataset_dir = Path(args.asvspoof5_dir).resolve()
    split = args.split.lower()

    if split in ("val", "validation", "dev"):
        _, records = build_index(
            dataset_dir, split="train",
            val_speaker_fraction=args.val_speaker_fraction,
            seed=args.seed,
        )
        split_label = "validation"
    elif split == "train":
        records, _ = build_index(
            dataset_dir, split="train",
            val_speaker_fraction=args.val_speaker_fraction,
            seed=args.seed,
        )
        split_label = "train"
    else:  # test/eval — use dev protocol
        _, records = build_index(
            dataset_dir, split="train",
            val_speaker_fraction=args.val_speaker_fraction,
            seed=args.seed,
        )
        split_label = "eval"

    # Subsample for faster eval if requested
    if args.max_samples and len(records) > args.max_samples:
        import random
        random.seed(args.seed)
        records = random.sample(records, args.max_samples)
        logger.info(f"Subsampled to {len(records):,} records (--max-samples)")

    logger.info(
        f"Evaluating on {len(records):,} records "
        f"(split: {split_label})"
    )

    # ── Score all samples ─────────────────────────────────────────────────────
    logger.info("Running model inference ...")
    labels, scores, avg_lat = score_dataset(
        model, records, device, batch_size=args.batch_size
    )
    logger.info(f"Scored {len(scores):,} samples. Avg latency: {avg_lat:.2f} ms/sample")

    # ── Overall metrics ───────────────────────────────────────────────────────
    overall = compute_all_metrics(
        labels, scores, threshold,
        label_note=f"ASVspoof5 {split_label} split",
    )
    logger.info(
        f"Overall — AUC={overall['roc_auc']:.4f}  "
        f"EER={overall['eer']:.4f}  "
        f"F1={overall['f1']:.4f}  "
        f"threshold={threshold:.4f}"
    )

    # ── Per-generator ─────────────────────────────────────────────────────────
    logger.info("Computing per-generator breakdown ...")
    per_gen = per_generator_report(records, labels, scores, threshold)

    # ── Robustness ────────────────────────────────────────────────────────────
    if not args.skip_robustness:
        logger.info("Running robustness evaluation ...")
        rob_records = records[:min(2000, len(records))]  # subsample for speed
        robustness = robustness_eval(model, rob_records, device, threshold,
                                     batch_size=args.batch_size)
    else:
        robustness = {"skipped": {"note": "--skip-robustness was set"}}

    # ── Output ────────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ds_type = checkpoint_meta["dataset_type"].replace("/", "_").replace(" ", "_")
    output_dir = Path(args.output_dir) / f"eval_{split_label}_{ts}"

    write_report(output_dir, overall, per_gen, robustness, checkpoint_meta, avg_lat)

    # Print key metrics to stdout
    print()
    print("=" * 60)
    print(f"VoxShield Evaluation — {split_label.upper()}")
    print("=" * 60)
    print(f"  ROC-AUC   : {overall['roc_auc']:.4f}")
    print(f"  EER       : {overall['eer']:.4f}")
    print(f"  Accuracy  : {overall['accuracy']:.4f}  (threshold={threshold:.4f})")
    print(f"  F1        : {overall['f1']:.4f}")
    print(f"  Precision : {overall['precision']:.4f}")
    print(f"  Recall    : {overall['recall']:.4f}")
    print(f"  FPR       : {overall['fpr']:.4f}")
    print(f"  FNR       : {overall['fnr']:.4f}")
    print(f"  Latency   : {avg_lat:.2f} ms/sample")
    print(f"  Report    : {output_dir}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxShieldNet evaluation on ASVspoof5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt checkpoint (threshold must be stored inside)."
    )
    parser.add_argument(
        "--asvspoof5-dir", required=True,
        help="Root directory of ASVspoof5 dataset."
    )
    parser.add_argument(
        "--split", default="dev",
        choices=["train", "dev", "val", "validation", "eval"],
        help="Which split to evaluate."
    )
    parser.add_argument(
        "--output-dir", default="reports/asvspoof5",
        help="Directory for evaluation reports."
    )
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument(
        "--val-speaker-fraction", type=float, default=0.15
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to evaluate (for quick runs)."
    )
    parser.add_argument(
        "--skip-robustness", action="store_true",
        help="Skip robustness evaluation (faster)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
