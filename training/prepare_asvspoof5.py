"""
ASVspoof5 Dataset Preparation Command
======================================

Usage:
    python -m training.prepare_asvspoof5 --dataset-dir "D:\\Datasets\\ASVspoof5"

What it does:
    1. Verifies required protocol files exist (at root OR under protocols/).
    2. Verifies audio directory exists and has .flac files.
    3. Builds and caches a full index (path existence check, no audio decode).
    4. Reports class counts, speaker counts, generator/attack type distribution.
    5. Reports missing audio files.
    6. Reports speaker and generator overlap between train and dev splits.
    7. Optionally verifies all files can be opened (--verify-audio flag).

Does NOT:
    - Decode every file unless --verify-audio is passed.
    - Modify, copy, or move any dataset files.
    - Put dataset audio into the project repository.
    - Require artificial train/ or dev/ subdirectories.

Official ASVspoof5 native layout expected:
    <dataset-dir>/
        ASVspoof5.train.tsv          (or protocols/ASVspoof5.train.tsv)
        ASVspoof5.dev.track_1.tsv    (optional, or protocols/...)
        ASVspoof5.eval.track_1.tsv   (optional)
        flac_T/
            T_*.flac
        flac_D/                      (optional)
            D_*.flac
        flac_E_eval/                 (optional)
            E_*.flac
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _check_protocol(proto_path: Path) -> bool:
    ok = proto_path.exists()
    status = "OK" if ok else "MISSING"
    logger.info(f"  Protocol [{status}]: {proto_path}")
    return ok


def _find_protocol_path(dataset_dir: Path, filename: str) -> Path:
    """Return the path where a protocol file was found, or the root-level path."""
    candidate = dataset_dir / filename
    if candidate.exists():
        return candidate
    candidate2 = dataset_dir / "protocols" / filename
    if candidate2.exists():
        return candidate2
    return dataset_dir / filename  # return the root-level path for error messages


def _count_flac(audio_dir: Path) -> int:
    if not audio_dir.exists():
        return 0
    return sum(1 for _ in audio_dir.glob("*.flac"))


def _check_overlap(set_a: set, set_b: set, kind: str) -> None:
    overlap = set_a & set_b
    if overlap:
        logger.warning(
            f"  {kind} overlap (train ∩ dev): {len(overlap)} items"
            " — this is expected for attack types but NOT for speakers."
        )
    else:
        logger.info(f"  {kind}: NO overlap between train and dev ✓")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and verify ASVspoof5 dataset index for VoxShield."
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Root directory of ASVspoof5 dataset."
    )
    parser.add_argument(
        "--val-speaker-fraction", type=float, default=0.15,
        help="Fraction of speakers to use as validation if dev audio is absent."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for speaker split."
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Rebuild index cache even if it already exists."
    )
    parser.add_argument(
        "--verify-audio", action="store_true",
        help="Open every file to verify it is readable (slow for large datasets)."
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Optional path to write a JSON summary of the dataset."
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    logger.info(f"Dataset directory: {dataset_dir}")

    if not dataset_dir.exists():
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    # ── Protocol checks ───────────────────────────────────────────────────────
    logger.info("Checking protocols ...")

    # Required protocols — checked at both root and protocols/ subdirectory
    required_filenames = [
        "ASVspoof5.train.tsv",
    ]
    optional_filenames = [
        "ASVspoof5.dev.track_1.tsv",
        "ASVspoof5.eval.track_1.tsv",
        "ASVspoof5.codec.config.csv",
    ]

    all_ok = True
    for name in required_filenames:
        path = _find_protocol_path(dataset_dir, name)
        ok = _check_protocol(path)
        all_ok = all_ok and ok
    for name in optional_filenames:
        path = _find_protocol_path(dataset_dir, name)
        _check_protocol(path)

    if not all_ok:
        logger.error("One or more required protocol files are missing.")
        logger.error(
            "Protocol files should be at the dataset root "
            "(e.g. D:\\Datasets\\ASVspoof5\\ASVspoof5.train.tsv) "
            "or under a protocols/ subdirectory."
        )
        sys.exit(1)

    # ── Audio directory checks ────────────────────────────────────────────────
    logger.info("Checking audio directories ...")

    # Official native ASVspoof5 layout
    train_audio_dir = dataset_dir / "flac_T"
    dev_audio_dir   = dataset_dir / "flac_D"
    eval_audio_dir  = dataset_dir / "flac_E_eval"

    n_train_flac = _count_flac(train_audio_dir)
    n_dev_flac   = _count_flac(dev_audio_dir)
    n_eval_flac  = _count_flac(eval_audio_dir)

    logger.info(f"  Train audio [flac_T/]    : {n_train_flac:,} .flac files — {train_audio_dir}")
    logger.info(f"  Dev audio   [flac_D/]    : {n_dev_flac:,} .flac files — {dev_audio_dir}")
    logger.info(f"  Eval audio  [flac_E_eval/]: {n_eval_flac:,} .flac files — {eval_audio_dir}")

    if n_train_flac == 0:
        logger.error(
            "No training audio files found in flac_T/.\n"
            "  Expected: " + str(train_audio_dir) + "\\T_*.flac\n"
            "  Extract the training archives (flac_T_aa.tar…flac_T_ae.tar) "
            "directly into the dataset root."
        )
        sys.exit(1)

    # ── Build index ───────────────────────────────────────────────────────────
    logger.info("Building/loading dataset index ...")
    try:
        from training.dataset import build_index, dataset_stats, _parse_protocol
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from training.dataset import build_index, dataset_stats, _parse_protocol

    try:
        train_records, val_records = build_index(
            dataset_dir,
            split="train",
            val_speaker_fraction=args.val_speaker_fraction,
            seed=args.seed,
            force_rebuild=args.force_rebuild,
        )
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Dataset statistics ────────────────────────────────────────────────────
    logger.info("\n══════════════════════════════════════")
    logger.info("TRAIN SPLIT")
    logger.info("══════════════════════════════════════")
    t_stats = dataset_stats(train_records)
    _print_stats(t_stats)

    logger.info("\n══════════════════════════════════════")
    logger.info("VALIDATION SPLIT")
    logger.info("══════════════════════════════════════")
    v_stats = dataset_stats(val_records)
    _print_stats(v_stats)

    # ── Overlap checks ────────────────────────────────────────────────────────
    logger.info("\nOverlap analysis ...")
    train_speakers = {r["speaker_id"] for r in train_records}
    val_speakers   = {r["speaker_id"] for r in val_records}
    _check_overlap(train_speakers, val_speakers, "Speaker")

    train_attacks = {r["attack_type"] for r in train_records}
    val_attacks   = {r["attack_type"] for r in val_records}
    _check_overlap(train_attacks, val_attacks, "Attack type (overlap expected)")

    # ── Optional audio verification ───────────────────────────────────────────
    if args.verify_audio:
        _verify_audio(train_records + val_records)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "dataset_dir": str(dataset_dir),
        "train": t_stats,
        "val": v_stats,
        "audio_dirs": {
            "flac_T":     n_train_flac,
            "flac_D":     n_dev_flac,
            "flac_E_eval": n_eval_flac,
        },
        "speaker_overlap": len(train_speakers & val_speakers),
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSummary written to {out_path}")

    logger.info("\n✓ Dataset preparation complete.")
    logger.info(
        f"  Ready: {t_stats['total']:,} train + {v_stats['total']:,} val samples"
    )
    logger.info(
        "  Next: python -m training.train "
        f"--asvspoof5-dir \"{dataset_dir}\""
    )


def _print_stats(stats: dict) -> None:
    logger.info(f"  Total samples  : {stats['total']:,}")
    logger.info(f"  Bonafide (0)   : {stats['bonafide']:,} ({stats['bonafide_pct']:.1f}%)")
    logger.info(f"  Spoof    (1)   : {stats['spoof']:,} ({stats['spoof_pct']:.1f}%)")
    logger.info(f"  Unique speakers: {stats['unique_speakers']:,}")
    logger.info("  Attack type distribution:")
    for at, cnt in sorted(stats["attack_types"].items(), key=lambda x: -x[1]):
        logger.info(f"    {at:20s}: {cnt:,}")


def _verify_audio(records: list) -> None:
    import soundfile as sf
    from tqdm import tqdm
    logger.info(f"\nVerifying {len(records):,} audio files ...")
    errors = 0
    for r in tqdm(records, desc="Verifying audio"):
        try:
            info = sf.info(r["path"])
            if info.samplerate != 16000:
                logger.warning(
                    f"  Sample rate {info.samplerate} (expected 16000): {r['path']}"
                )
        except Exception as e:
            logger.error(f"  Cannot read {r['path']}: {e}")
            errors += 1
    if errors:
        logger.error(f"{errors} files could not be opened.")
    else:
        logger.info("All files verified OK.")


if __name__ == "__main__":
    main()
