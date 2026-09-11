"""
VoxShield Training Pipeline
============================

Usage:
    # Full ASVspoof5 training
    python -m training.train \
        --asvspoof5-dir /mnt/d/Datasets/ASVspoof5 \
        --epochs 30 \
        --checkpoint-dir checkpoints/asvspoof5

    # Resume from last checkpoint
    python -m training.train \
        --asvspoof5-dir /mnt/d/Datasets/ASVspoof5 \
        --checkpoint-dir checkpoints/asvspoof5 \
        --resume

Checkpoint behaviour:
    checkpoints/asvspoof5/best.pt  — best validation AUC
    checkpoints/asvspoof5/last.pt  — last completed epoch (always saved)

    Checkpoints include:
        model_state_dict, model_config, epoch, optimizer_state_dict,
        scheduler_state_dict, scaler_state_dict (if AMP), metadata,
        dataset_info, threshold, val_metrics

    Writes are atomic (write to .tmp then rename).
    ASVspoof5 checkpoints are isolated under --checkpoint-dir.
    Demo checkpoints remain under checkpoints/ (root).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# ── Add project root to path ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.voxshieldnet import VoxShieldNet, build_model, MODEL_CONFIG
from training.dataset import (
    ASVspoof5Dataset, build_index, collate_fn, dataset_stats
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# EER computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate and the corresponding threshold."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    abs_diff = np.abs(fpr - fnr)
    eer_idx = np.argmin(abs_diff)
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    threshold = float(thresholds[eer_idx])
    return eer, threshold


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict:
    """Compute full set of classification metrics at a given threshold."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    try:
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        auc = 0.5
    eer, _ = compute_eer(labels, scores)

    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "roc_auc":   auc,
        "eer":       eer,
        "fpr":       fpr_val,
        "fnr":       fnr_val,
        "threshold": threshold,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def _atomic_save(obj: dict, path: Path) -> None:
    """Save checkpoint atomically: write to .tmp then rename."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    shutil.move(str(tmp), str(path))


def load_checkpoint(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt
    except Exception as e:
        logger.warning(f"Could not load checkpoint {path}: {e}")
        return None


def save_checkpoint(
    path: Path,
    model: VoxShieldNet,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    val_metrics: Dict,
    dataset_info: Dict,
    threshold: float,
    best_auc: float,
    training_history: list,
) -> None:
    ckpt = {
        "model_state_dict":     model.state_dict(),
        "model_config":         model.config,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict":    scaler.state_dict() if scaler else None,
        "epoch":                epoch,
        "val_metrics":          val_metrics,
        "threshold":            threshold,
        "best_val_auc":         best_auc,
        "dataset_info":         dataset_info,
        "training_history":     training_history,
        "metadata": {
            "model_name":    "VoxShieldNet",
            "version":       MODEL_CONFIG["version"],
            "dataset_type":  dataset_info.get("dataset_type", "unknown"),
            "pretrained":    False,
            "training_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(ckpt, path)


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: VoxShieldNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run validation. Returns (avg_loss, all_labels, all_scores)."""
    model.eval()
    total_loss = 0.0
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for wavs, labels, _ in loader:
        wavs   = wavs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(wavs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_scores.append(probs)
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(1, sum(len(x) for x in all_labels))
    return (
        avg_loss,
        np.concatenate(all_labels),
        np.concatenate(all_scores),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── Device setup ─────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    use_amp = args.amp and device.type == "cuda"
    if args.amp and not use_amp:
        logger.info("AMP requested but no CUDA available — running in FP32.")
    if use_amp:
        logger.info("Mixed precision (AMP) enabled.")

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_dir = Path(args.asvspoof5_dir).resolve()
    logger.info(f"ASVspoof5 dataset: {dataset_dir}")

    train_records, val_records = build_index(
        dataset_dir,
        split="train",
        val_speaker_fraction=args.val_speaker_fraction,
        seed=args.seed,
        force_rebuild=args.force_rebuild_index,
    )

    # ── Safety guards — fail early with a clear message ───────────────────────
    if len(train_records) == 0:
        logger.error(
            "Training dataset is empty — 0 records resolved.\n"
            f"  Dataset dir : {dataset_dir}\n"
            "  Expected to find .flac files under flac_T/.\n"
            "  Run: python -m training.prepare_asvspoof5 "
            f"--dataset-dir \"{dataset_dir}\" --force-rebuild"
        )
        sys.exit(1)

    if len(val_records) == 0:
        logger.error(
            "Validation dataset is empty — 0 records resolved.\n"
            "  Cannot train without a validation set.\n"
            "  Check dev audio in flac_D/ or increase --val-speaker-fraction."
        )
        sys.exit(1)

    logger.info(
        f"Dataset: {len(train_records):,} train records, "
        f"{len(val_records):,} val records"
    )

    train_ds = ASVspoof5Dataset(
        train_records,
        max_samples=int(MODEL_CONFIG["max_length_sec"] * MODEL_CONFIG["sample_rate"]),
        augment=True,
        seed=args.seed,
    )
    val_ds = ASVspoof5Dataset(
        val_records,
        max_samples=int(MODEL_CONFIG["max_length_sec"] * MODEL_CONFIG["sample_rate"]),
        augment=False,
        seed=args.seed,
    )

    dataset_info = {
        "dataset_type":   "ASVspoof5",
        "dataset_dir":    str(dataset_dir),
        "train_stats":    dataset_stats(train_records),
        "val_stats":      dataset_stats(val_records),
        "val_speaker_fraction": args.val_speaker_fraction,
        "seed":           args.seed,
    }

    # ── Sampler for class balance ─────────────────────────────────────────────
    sample_w = train_ds.sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_w,
        num_samples=len(train_ds),
        replacement=True,
    )

    pin_mem = (device.type == "cuda")
    nw = args.num_workers
    persistent = (nw > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        collate_fn=collate_fn,
        drop_last=False,
    )

    logger.info(
        f"Train: {len(train_ds):,} samples / {len(train_loader):,} batches  |  "
        f"Val: {len(val_ds):,} samples / {len(val_loader):,} batches"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(device)
    logger.info(model.summary())

    # ── Loss (class-weighted BCE) ─────────────────────────────────────────────
    cw = train_ds.class_weights()
    pos_weight = torch.tensor([cw[1] / cw[0]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_auc = 0.0
    training_history: list = []
    best_threshold = 0.5

    if args.resume:
        resume_ckpt = load_checkpoint(last_path)
        if resume_ckpt is None:
            resume_ckpt = load_checkpoint(best_path)
        if resume_ckpt is not None:
            # Validate dataset matches
            ckpt_ds = resume_ckpt.get("dataset_info", {}).get("dataset_type", "")
            cur_ds  = dataset_info["dataset_type"]
            if ckpt_ds and ckpt_ds != cur_ds:
                logger.error(
                    f"Dataset mismatch: checkpoint is '{ckpt_ds}', "
                    f"current is '{cur_ds}'. "
                    "Use --checkpoint-dir to isolate datasets."
                )
                sys.exit(1)

            model.load_state_dict(resume_ckpt["model_state_dict"])
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            if resume_ckpt.get("scheduler_state_dict") and scheduler:
                try:
                    scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
                except Exception as e:
                    logger.warning(f"Could not restore scheduler state: {e}")
            if scaler and resume_ckpt.get("scaler_state_dict"):
                scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
            start_epoch = resume_ckpt.get("epoch", 0) + 1
            best_val_auc = resume_ckpt.get("best_val_auc", 0.0)
            best_threshold = resume_ckpt.get("threshold", 0.5)
            training_history = resume_ckpt.get("training_history", [])
            logger.info(
                f"Resumed from epoch {start_epoch} "
                f"(best AUC so far: {best_val_auc:.4f})"
            )
        else:
            logger.warning("--resume specified but no checkpoint found. Starting fresh.")

    # ── Early stopping ────────────────────────────────────────────────────────
    patience_counter = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"Starting training from epoch {start_epoch + 1}/{args.epochs}")
    logger.info(
        f"Batch size: {args.batch_size}  |  Workers: {nw}  |  "
        f"LR: {args.lr}  |  Patience: {args.patience}"
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        t0 = time.time()

        for batch_idx, (wavs, labels, _) in enumerate(train_loader):
            wavs   = wavs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(wavs)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(wavs)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            n_batches  += 1

            if batch_idx % 200 == 0:
                lr_now = scheduler.get_last_lr()[0]
                logger.info(
                    f"  Epoch {epoch+1}/{args.epochs} "
                    f"[{batch_idx}/{len(train_loader)}]  "
                    f"loss={epoch_loss/n_batches:.4f}  lr={lr_now:.2e}"
                )

        avg_train_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t0

        # ── Validation ────────────────────────────────────────────────────────
        val_loss, val_labels, val_scores = validate(model, val_loader, criterion, device)
        eer, eer_threshold = compute_eer(val_labels, val_scores)
        val_metrics = compute_metrics(val_labels, val_scores, eer_threshold)
        val_metrics["val_loss"] = val_loss
        val_metrics["train_loss"] = avg_train_loss

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"AUC={val_metrics['roc_auc']:.4f} | "
            f"EER={eer:.4f} | "
            f"F1={val_metrics['f1']:.4f} | "
            f"thr={eer_threshold:.4f} | "
            f"elapsed={elapsed:.0f}s"
        )

        training_history.append({
            "epoch":       epoch + 1,
            "train_loss":  avg_train_loss,
            "val_loss":    val_loss,
            "val_auc":     val_metrics["roc_auc"],
            "val_eer":     eer,
            "val_f1":      val_metrics["f1"],
            "threshold":   eer_threshold,
        })

        # ── Best checkpoint ───────────────────────────────────────────────────
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc    = val_metrics["roc_auc"]
            best_threshold  = eer_threshold
            patience_counter = 0
            save_checkpoint(
                best_path, model, optimizer, scheduler, scaler,
                epoch, val_metrics, dataset_info,
                best_threshold, best_val_auc, training_history,
            )
            logger.info(
                f"  ✓ New best AUC={best_val_auc:.4f} saved → {best_path}"
            )
        else:
            patience_counter += 1
            logger.info(
                f"  No improvement (patience {patience_counter}/{args.patience})"
            )

        # ── Last checkpoint (always) ──────────────────────────────────────────
        save_checkpoint(
            last_path, model, optimizer, scheduler, scaler,
            epoch, val_metrics, dataset_info,
            best_threshold, best_val_auc, training_history,
        )

        # ── Save training history JSON ────────────────────────────────────────
        history_path = ckpt_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_counter >= args.patience:
            logger.info(
                f"Early stopping at epoch {epoch+1} "
                f"(no AUC improvement for {args.patience} epochs)"
            )
            break

    logger.info("═" * 60)
    logger.info(f"Training complete.")
    logger.info(f"Best validation AUC : {best_val_auc:.4f}")
    logger.info(f"Best threshold (EER): {best_threshold:.4f}")
    logger.info(f"Best checkpoint     : {best_path}")
    logger.info(f"Last checkpoint     : {last_path}")
    logger.info("═" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxShieldNet training on ASVspoof5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--asvspoof5-dir", required=True,
        help="Root directory of ASVspoof5 dataset."
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/asvspoof5",
        help="Directory for saving checkpoints (isolated per dataset)."
    )
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--patience",    type=int,   default=8)
    parser.add_argument(
        "--val-speaker-fraction", type=float, default=0.15,
        help="Fraction of speakers for validation if dev audio is absent."
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="Enable AMP mixed precision (CUDA only)."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint in --checkpoint-dir."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device: cpu | cuda | cuda:0 etc."
    )
    parser.add_argument(
        "--force-rebuild-index", action="store_true",
        help="Rebuild dataset index cache."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
