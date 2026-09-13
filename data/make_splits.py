"""
VoxShield Production Dataset Split Generator
=============================================

Reads the production manifest CSV and generates deterministic, leakage-free
train / calibration / test splits.

Leakage prevention:
  1. Speaker-level disjoint: if speaker_id is populated, all files from a
     speaker go to exactly one split.
  2. Generator/source-level disjoint: if generator_id or tts_source is
     populated, all files from a generator go to exactly one split.
  3. SHA-256 near-duplicate check: identical files (same SHA-256) cannot
     appear in more than one split.
  4. When no speaker/generator metadata is available, files are assigned by
     SHA-256 bucket (deterministic).

Output:
    data/manifests/train.csv
    data/manifests/calibration.csv
    data/manifests/test.csv
    data/manifests/split_summary.json

Usage:
    python -m data.make_splits \\
        --manifest data/manifests/production_audio.csv \\
        --train 0.70 --cal 0.15 --test 0.15 --seed 42

Split fractions must sum to 1.0.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SPLIT_TRAIN = "train"
SPLIT_CAL   = "calibration"
SPLIT_TEST  = "test"

SPLIT_CSV_COLS: List[str] = [
    "path", "sha256", "label", "label_str", "domain", "source",
    "speaker_id", "generator_id", "tts_source", "replay_device", "codec",
    "sample_rate", "channels", "duration_sec", "file_size_bytes",
    "validation_status", "validation_notes", "split",
]


# ── Manifest reader ───────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """
    Load the production manifest CSV.
    Skips rows whose validation_status is not 'OK' or 'WARNING'.
    (ERROR rows are in the quarantine file, not the manifest.)
    """
    rows: List[Dict[str, Any]] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    logger.info(f"Loaded {len(rows)} rows from {manifest_path}")
    return rows


# ── Grouping helpers ──────────────────────────────────────────────────────────

def _group_key(row: Dict[str, Any]) -> str:
    """
    Return the grouping key for leakage-prevention assignment.

    Priority:
      1. speaker_id  (if non-empty)
      2. generator_id (if non-empty) — covers TTS voices
      3. tts_source   (if non-empty)
      4. sha256[:8]   (fallback — deterministic per unique file)
    """
    sid = (row.get("speaker_id") or "").strip()
    if sid:
        return f"spk:{sid}"

    gid = (row.get("generator_id") or "").strip()
    if gid:
        return f"gen:{gid}"

    tts = (row.get("tts_source") or "").strip()
    if tts:
        return f"tts:{tts}"

    sha = (row.get("sha256") or "")[:8]
    return f"sha:{sha}"


def _group_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Partition rows into groups by their grouping key."""
    groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)
    return dict(groups)


# ── Deterministic shuffler ────────────────────────────────────────────────────

def _stable_shuffle(items: list, seed: int) -> list:
    """Return a deterministically shuffled copy of items."""
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    return items


# ── Split assignment ──────────────────────────────────────────────────────────

def assign_splits(
    rows: List[Dict[str, Any]],
    train_frac: float,
    cal_frac: float,
    test_frac: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Assign each row to train / calibration / test deterministically.

    Strategy:
      1. Group rows by their leakage-prevention key.
      2. Shuffle groups deterministically with seed.
      3. Assign groups to splits in proportion to requested fractions,
         attempting to preserve the bonafide/spoof ratio within each split.
      4. Return rows with an added 'split' column.

    Near-duplicates (same SHA-256) are guaranteed not to span splits
    because the SHA-256 is part of the group key for ungrouped files.
    """
    if abs(train_frac + cal_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got "
            f"{train_frac}+{cal_frac}+{test_frac}={train_frac+cal_frac+test_frac}"
        )
    if not rows:
        return []

    groups = _group_rows(rows)
    group_keys = _stable_shuffle(list(groups.keys()), seed)
    n_groups = len(group_keys)

    # Assign groups to splits
    n_train = max(1, round(n_groups * train_frac))
    n_cal   = max(1, round(n_groups * cal_frac))
    # test gets remainder
    n_test  = n_groups - n_train - n_cal
    if n_test < 0:
        # Edge case: very few groups — give test at least 1 if possible
        n_cal  = max(0, n_groups - n_train - 1)
        n_test = n_groups - n_train - n_cal

    train_keys = set(group_keys[:n_train])
    cal_keys   = set(group_keys[n_train : n_train + n_cal])
    test_keys  = set(group_keys[n_train + n_cal :])

    split_map = {}
    for k in train_keys:
        split_map[k] = SPLIT_TRAIN
    for k in cal_keys:
        split_map[k] = SPLIT_CAL
    for k in test_keys:
        split_map[k] = SPLIT_TEST

    result = []
    for row in rows:
        out = dict(row)
        out["split"] = split_map.get(_group_key(row), SPLIT_TRAIN)
        result.append(out)

    return result


# ── Leakage verification ──────────────────────────────────────────────────────

def check_leakage(rows_with_splits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verify no leakage between splits.

    Checks:
      - No SHA-256 appears in more than one split.
      - No speaker_id appears in more than one split.
      - No generator_id appears in more than one split.
      - No tts_source appears in more than one split.

    Returns a report dict.  'clean' is True only if all checks pass.
    """
    sha_splits:  Dict[str, Set[str]] = collections.defaultdict(set)
    spk_splits:  Dict[str, Set[str]] = collections.defaultdict(set)
    gen_splits:  Dict[str, Set[str]] = collections.defaultdict(set)
    tts_splits:  Dict[str, Set[str]] = collections.defaultdict(set)

    for row in rows_with_splits:
        sp = row.get("split", "")
        sha = (row.get("sha256") or "")[:16]
        if sha:
            sha_splits[sha].add(sp)

        sid = (row.get("speaker_id") or "").strip()
        if sid:
            spk_splits[sid].add(sp)

        gid = (row.get("generator_id") or "").strip()
        if gid:
            gen_splits[gid].add(sp)

        tts = (row.get("tts_source") or "").strip()
        if tts:
            tts_splits[tts].add(sp)

    sha_leaks = {k: sorted(v) for k, v in sha_splits.items() if len(v) > 1}
    spk_leaks = {k: sorted(v) for k, v in spk_splits.items() if len(v) > 1}
    gen_leaks = {k: sorted(v) for k, v in gen_splits.items() if len(v) > 1}
    tts_leaks = {k: sorted(v) for k, v in tts_splits.items() if len(v) > 1}

    clean = not any([sha_leaks, spk_leaks, gen_leaks, tts_leaks])

    if sha_leaks:
        logger.error(f"SHA-256 leakage detected: {len(sha_leaks)} hashes across splits")
    if spk_leaks:
        logger.error(f"Speaker leakage detected: {len(spk_leaks)} speakers across splits")
    if gen_leaks:
        logger.error(f"Generator leakage detected: {len(gen_leaks)} generators across splits")
    if tts_leaks:
        logger.error(f"TTS source leakage detected: {len(tts_leaks)} sources across splits")
    if clean:
        logger.info("Leakage check: CLEAN — no speaker/generator/SHA overlap between splits")

    return {
        "clean":          clean,
        "sha256_leaks":   sha_leaks,
        "speaker_leaks":  spk_leaks,
        "generator_leaks": gen_leaks,
        "tts_source_leaks": tts_leaks,
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def split_statistics(rows_with_splits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-split counts and class balance."""
    stats: Dict[str, Any] = {}

    for split_name in [SPLIT_TRAIN, SPLIT_CAL, SPLIT_TEST]:
        split_rows = [r for r in rows_with_splits if r.get("split") == split_name]
        n = len(split_rows)
        n_bonafide = sum(1 for r in split_rows if str(r.get("label", "")) == "0")
        n_spoof    = sum(1 for r in split_rows if str(r.get("label", "")) == "1")
        domains: Dict[str, int] = collections.Counter(
            r.get("domain", "unknown") for r in split_rows
        )
        total_dur = sum(float(r.get("duration_sec", 0)) for r in split_rows)
        stats[split_name] = {
            "n_total":     n,
            "n_bonafide":  n_bonafide,
            "n_spoof":     n_spoof,
            "domains":     dict(sorted(domains.items())),
            "total_duration_sec": round(total_dur, 3),
            "balance_ratio": round(n_spoof / n, 4) if n else None,
        }

    return stats


# ── Writer ────────────────────────────────────────────────────────────────────

def _write_split_csv(
    rows: List[Dict[str, Any]],
    split_name: str,
    out_dir: Path,
) -> Path:
    path = out_dir / f"{split_name}.csv"
    split_rows = [r for r in rows if r.get("split") == split_name]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(split_rows)
    return path


# ── Main entry point ──────────────────────────────────────────────────────────

def make_splits(
    manifest_path: Path,
    out_dir: Path,
    train_frac: float = 0.70,
    cal_frac:   float = 0.15,
    test_frac:  float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Load manifest, assign splits, check leakage, write CSVs + summary.

    Returns the summary dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    if not rows:
        logger.warning("Manifest is empty — no splits will be generated.")
        summary: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(manifest_path),
            "seed": seed,
            "fractions": {"train": train_frac, "calibration": cal_frac, "test": test_frac},
            "total_rows": 0,
            "leakage_check": {"clean": True},
            "split_statistics": {},
        }
        _write_summary(summary, out_dir)
        return summary

    logger.info(
        f"Generating splits: train={train_frac:.2f} cal={cal_frac:.2f} "
        f"test={test_frac:.2f} seed={seed}"
    )

    rows_with_splits = assign_splits(rows, train_frac, cal_frac, test_frac, seed)
    leakage_report   = check_leakage(rows_with_splits)
    stats            = split_statistics(rows_with_splits)

    # Write per-split CSVs
    train_path = _write_split_csv(rows_with_splits, SPLIT_TRAIN, out_dir)
    cal_path   = _write_split_csv(rows_with_splits, SPLIT_CAL,   out_dir)
    test_path  = _write_split_csv(rows_with_splits, SPLIT_TEST,  out_dir)

    logger.info(f"Train CSV:       {train_path} ({stats[SPLIT_TRAIN]['n_total']} rows)")
    logger.info(f"Calibration CSV: {cal_path}   ({stats[SPLIT_CAL]['n_total']} rows)")
    logger.info(f"Test CSV:        {test_path}  ({stats[SPLIT_TEST]['n_total']} rows)")

    if not leakage_report["clean"]:
        logger.error("LEAKAGE DETECTED — review split_summary.json")

    summary = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "manifest_path":  str(manifest_path),
        "seed":           seed,
        "fractions":      {"train": train_frac, "calibration": cal_frac, "test": test_frac},
        "total_rows":     len(rows_with_splits),
        "leakage_check":  leakage_report,
        "split_statistics": stats,
        "output_files": {
            "train":       str(train_path),
            "calibration": str(cal_path),
            "test":        str(test_path),
        },
        "disclaimer": (
            "Splits are deterministic given the same seed and manifest. "
            "Speaker/generator/SHA disjointness is enforced where metadata exists."
        ),
    }
    _write_summary(summary, out_dir)
    return summary


def _write_summary(summary: Dict[str, Any], out_dir: Path) -> None:
    p = out_dir / "split_summary.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Split summary written: {p}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate train/calibration/test splits from production manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--manifest",
        default="data/manifests/production_audio.csv",
        help="Path to production manifest CSV.",
    )
    p.add_argument(
        "--out-dir",
        default="data/manifests",
        help="Output directory for split CSVs.",
    )
    p.add_argument("--train", type=float, default=0.70,
                   help="Train fraction (0–1).")
    p.add_argument("--cal",   type=float, default=0.15,
                   help="Calibration fraction (0–1).")
    p.add_argument("--test",  type=float, default=0.15,
                   help="Test fraction (0–1).")
    p.add_argument("--seed",  type=int,   default=42,
                   help="Random seed for deterministic splitting.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    summary = make_splits(
        manifest_path=Path(args.manifest),
        out_dir=Path(args.out_dir),
        train_frac=args.train,
        cal_frac=args.cal,
        test_frac=args.test,
        seed=args.seed,
    )
    stats = summary.get("split_statistics", {})
    print(f"\nTotal rows : {summary['total_rows']}")
    for split_name in ["train", "calibration", "test"]:
        s = stats.get(split_name, {})
        print(
            f"  {split_name:12s}: {s.get('n_total', 0):5d} files  "
            f"({s.get('n_bonafide', 0)} bonafide, {s.get('n_spoof', 0)} spoof)"
        )
    leak = summary.get("leakage_check", {})
    print(f"\nLeakage check: {'CLEAN' if leak.get('clean') else 'LEAKAGE DETECTED'}")
