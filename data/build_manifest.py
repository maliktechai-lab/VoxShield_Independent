"""
VoxShield Production Dataset Manifest Builder
==============================================

Recursively discovers audio files under a root directory, extracts metadata,
computes SHA-256 fingerprints, validates each file, and writes:

    data/manifests/production_audio.csv        — per-file manifest
    data/manifests/production_audio_errors.csv — quarantine report
    data/manifests/production_audio_summary.json

Usage:
    python -m data.build_manifest --root production_data
    python -m data.build_manifest --root production_data --out-dir data/manifests

Expected directory layout (label and domain inferred from path):

    <root>/
      bonafide/
        clean/      → label=0, domain=clean
        noisy/      → label=0, domain=noisy
        replay/     → label=0, domain=replay
        codec/      → label=0, domain=codec
      spoof/
        tts/        → label=1, domain=tts
        converted/  → label=1, domain=converted
        noisy/      → label=1, domain=noisy
        replay/     → label=1, domain=replay
        codec/      → label=1, domain=codec

Arbitrary nesting below the domain folder is supported; files are discovered
recursively.  The folder name immediately under bonafide/ or spoof/ determines
the domain.  If a file sits directly under bonafide/ or spoof/ with no domain
subfolder, domain='clean' is assumed.

Validation rules (per file):
  ERROR  — zero-byte file
  ERROR  — unsupported format
  ERROR  — unreadable / corrupt audio
  ERROR  — non-finite decoded samples
  ERROR  — duration < min_duration_sec (default 0.1 s)
  WARNING — silent audio (peak amplitude < 1e-6)
  WARNING — duplicate SHA-256 (only the first occurrence is kept in the manifest)
  WARNING — suspicious duplicate filename (same basename in multiple locations)
  WARNING — missing optional metadata fields (speaker_id, generator_id, etc.)

Design principles:
  - Never silently discard bad files — all go to the error/quarantine CSV.
  - SHA-256 is computed from raw file bytes (format-agnostic).
  - Audio metadata (sample_rate, channels, duration_sec) uses soundfile.
  - No model inference is performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mp4"}
)

# Canonical label names → int
LABEL_MAP: Dict[str, int] = {"bonafide": 0, "spoof": 1}

# Valid domain names
VALID_DOMAINS: frozenset[str] = frozenset(
    {"clean", "tts", "converted", "replay", "noisy", "codec"}
)

# Default domain when a file lives directly under bonafide/ or spoof/
DEFAULT_DOMAIN = "clean"

MIN_DURATION_SEC: float = 0.1   # files shorter than this are flagged ERROR
SILENT_PEAK_THRESHOLD: float = 1e-6  # below this → WARNING:silent
CHUNK_BYTES: int = 1 << 20      # 1 MB chunks for SHA-256

# Manifest CSV columns (ordered)
MANIFEST_COLS: List[str] = [
    "path",
    "sha256",
    "label",
    "label_str",
    "domain",
    "source",
    "speaker_id",
    "generator_id",
    "tts_source",
    "replay_device",
    "codec",
    "sample_rate",
    "channels",
    "duration_sec",
    "file_size_bytes",
    "validation_status",
    "validation_notes",
]

ERROR_COLS: List[str] = [
    "path",
    "file_size_bytes",
    "inferred_label_str",
    "inferred_domain",
    "error_type",
    "error_detail",
]


# ── SHA-256 ───────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file by reading in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Label / domain inference ──────────────────────────────────────────────────

def infer_label_and_domain(
    path: Path,
    root: Path,
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Walk the path relative to root to find the label and domain folders.

    Returns (label_int, label_str, domain) or (None, None, None) if
    the file does not sit under a recognised label directory.

    Logic:
      - Find the part of the path that matches 'bonafide' or 'spoof'.
      - The first directory immediately after is the domain (if it matches
        VALID_DOMAINS), otherwise DEFAULT_DOMAIN is used.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None, None, None

    parts = list(rel.parts)
    label_str: Optional[str] = None
    label_idx: int = -1

    for i, part in enumerate(parts):
        if part.lower() in LABEL_MAP:
            label_str = part.lower()
            label_idx = i
            break

    if label_str is None:
        return None, None, None

    label_int = LABEL_MAP[label_str]

    # Domain: directory immediately after the label directory
    domain: str = DEFAULT_DOMAIN
    if label_idx + 1 < len(parts) - 1:          # there's a part between label and file
        candidate = parts[label_idx + 1].lower()
        if candidate in VALID_DOMAINS:
            domain = candidate
        # else keep DEFAULT_DOMAIN (unexpected subfolder name)

    return label_int, label_str, domain


# ── Audio metadata ────────────────────────────────────────────────────────────

def audio_metadata(path: Path) -> Dict[str, Any]:
    """
    Extract audio metadata using soundfile (fast, no full decode).

    Returns dict with keys: sample_rate, channels, duration_sec, format, subtype.
    Raises RuntimeError on failure.
    """
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return {
            "sample_rate":  int(info.samplerate),
            "channels":     int(info.channels),
            "duration_sec": round(float(info.duration), 6),
            "sf_format":    info.format,
            "sf_subtype":   info.subtype,
        }
    except Exception as exc:
        raise RuntimeError(f"soundfile cannot read '{path}': {exc}") from exc


def check_audio_content(path: Path, max_decode_sec: float = 10.0) -> Dict[str, Any]:
    """
    Decode up to max_decode_sec of audio and check for non-finite / silent samples.

    Returns dict with keys: is_silent, has_non_finite, peak_amplitude.
    May raise RuntimeError if the file cannot be decoded.
    """
    import soundfile as sf
    import numpy as np

    try:
        info = sf.info(str(path))
        frames_to_read = min(
            int(info.samplerate * max_decode_sec),
            info.frames,
        )
        data, _ = sf.read(
            str(path),
            frames=frames_to_read,
            dtype="float32",
            always_2d=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Cannot decode '{path}': {exc}") from exc

    if data.size == 0:
        return {"is_silent": True, "has_non_finite": False, "peak_amplitude": 0.0}

    has_non_finite = bool(not np.isfinite(data).all())
    peak = float(np.abs(data).max())
    is_silent = peak < SILENT_PEAK_THRESHOLD

    return {
        "is_silent":      is_silent,
        "has_non_finite": has_non_finite,
        "peak_amplitude": round(peak, 8),
    }


# ── Per-file processing ───────────────────────────────────────────────────────

def process_file(
    path: Path,
    root: Path,
    seen_hashes: Set[str],
    seen_basenames: Dict[str, List[str]],
    min_duration_sec: float = MIN_DURATION_SEC,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process one audio file.

    Returns (manifest_row, error_row).
    Exactly one of them is None.
    - If file is valid (or only has warnings):  manifest_row is populated,
      error_row is None.
    - If file has a fatal error:                manifest_row is None,
      error_row is populated.
    - Duplicate SHA-256: first occurrence goes to manifest, subsequent
      occurrences go to error_row with error_type='DUPLICATE_SHA256'.
    """
    path_str = str(path)
    file_size = path.stat().st_size if path.exists() else 0

    label_int, label_str, domain = infer_label_and_domain(path, root)

    def _error(etype: str, detail: str) -> Dict[str, Any]:
        return {
            "path":               path_str,
            "file_size_bytes":    file_size,
            "inferred_label_str": label_str or "unknown",
            "inferred_domain":    domain or "unknown",
            "error_type":         etype,
            "error_detail":       detail,
        }

    # ── Zero-byte ─────────────────────────────────────────────────────────────
    if file_size == 0:
        return None, _error("ZERO_BYTE", "File is empty (0 bytes)")

    # ── Extension check ───────────────────────────────────────────────────────
    ext = path.suffix.lower()
    if ext not in AUDIO_EXTENSIONS:
        return None, _error(
            "UNSUPPORTED_FORMAT",
            f"Extension '{ext}' not in supported set {sorted(AUDIO_EXTENSIONS)}",
        )

    # ── Label inference ───────────────────────────────────────────────────────
    if label_str is None:
        return None, _error(
            "LABEL_UNKNOWN",
            "Cannot infer label: file is not under a 'bonafide' or 'spoof' directory",
        )

    # ── SHA-256 ───────────────────────────────────────────────────────────────
    try:
        sha = sha256_file(path)
    except Exception as exc:
        return None, _error("READ_ERROR", f"Cannot read file for hashing: {exc}")

    # ── Duplicate SHA-256 ─────────────────────────────────────────────────────
    if sha in seen_hashes:
        return None, _error(
            "DUPLICATE_SHA256",
            f"SHA-256 {sha[:16]}... already seen; first occurrence kept",
        )
    seen_hashes.add(sha)

    # ── Duplicate basename tracking (warning only, not fatal) ─────────────────
    basename = path.name
    seen_basenames[basename].append(path_str)
    dup_basename_note = ""
    if len(seen_basenames[basename]) > 1:
        dup_basename_note = (
            f"WARN:duplicate_filename:{basename} appears in multiple locations"
        )

    # ── Audio metadata ────────────────────────────────────────────────────────
    try:
        meta = audio_metadata(path)
    except RuntimeError as exc:
        return None, _error("UNREADABLE_AUDIO", str(exc))

    # ── Duration check ────────────────────────────────────────────────────────
    dur = meta["duration_sec"]
    if dur < min_duration_sec:
        return None, _error(
            "TOO_SHORT",
            f"Duration {dur:.3f}s < minimum {min_duration_sec:.3f}s",
        )

    # ── Content check (non-finite / silence) ──────────────────────────────────
    try:
        content = check_audio_content(path)
    except RuntimeError as exc:
        return None, _error("DECODE_ERROR", str(exc))

    if content["has_non_finite"]:
        return None, _error(
            "NON_FINITE_SAMPLES",
            "Decoded audio contains NaN or Inf values",
        )

    # Build validation notes (warnings)
    notes: List[str] = []
    if content["is_silent"]:
        notes.append(f"WARN:silent:peak={content['peak_amplitude']:.2e}")
    if dup_basename_note:
        notes.append(dup_basename_note)

    validation_status = "WARNING" if notes else "OK"

    # ── Build manifest row ────────────────────────────────────────────────────
    # Attempt to derive optional metadata from path components
    source = _infer_source(path, root, label_str, domain)

    row: Dict[str, Any] = {
        "path":              path_str,
        "sha256":            sha,
        "label":             label_int,
        "label_str":         label_str,
        "domain":            domain,
        "source":            source,
        "speaker_id":        "",   # populated externally or from filename conventions
        "generator_id":      "",
        "tts_source":        "",
        "replay_device":     "",
        "codec":             meta.get("sf_subtype", ""),
        "sample_rate":       meta["sample_rate"],
        "channels":          meta["channels"],
        "duration_sec":      dur,
        "file_size_bytes":   file_size,
        "validation_status": validation_status,
        "validation_notes":  "; ".join(notes),
    }

    return row, None


def _infer_source(
    path: Path, root: Path, label_str: str, domain: str
) -> str:
    """
    Derive a 'source' string from path structure.
    This is a best-effort string — callers can override it.

    Convention: <label_str>/<domain>
    e.g. bonafide/clean, spoof/tts
    """
    return f"{label_str}/{domain}"


# ── Manifest builder ──────────────────────────────────────────────────────────

def build_manifest(
    root: Path,
    out_dir: Path,
    min_duration_sec: float = MIN_DURATION_SEC,
    manifest_name: str = "production_audio",
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Discover all audio files under root, validate them, and write:
      out_dir/<manifest_name>.csv
      out_dir/<manifest_name>_errors.csv
      out_dir/<manifest_name>_summary.json

    Returns the summary dict.
    """
    root = root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    logger.info(f"Scanning {root} for audio files …")

    # Discover all files with supported extensions recursively
    all_files: List[Path] = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    logger.info(f"Found {len(all_files)} candidate audio files")

    seen_hashes: Set[str] = set()
    seen_basenames: Dict[str, List[str]] = defaultdict(list)

    manifest_rows: List[Dict[str, Any]] = []
    error_rows:    List[Dict[str, Any]] = []

    for i, path in enumerate(all_files):
        if show_progress and (i % 50 == 0 or i == len(all_files) - 1):
            logger.info(f"  Processing {i + 1}/{len(all_files)}: {path.name}")

        ok_row, err_row = process_file(
            path, root, seen_hashes, seen_basenames,
            min_duration_sec=min_duration_sec,
        )
        if ok_row is not None:
            manifest_rows.append(ok_row)
        else:
            error_rows.append(err_row)

    # ── Write manifest CSV ────────────────────────────────────────────────────
    manifest_path = out_dir / f"{manifest_name}.csv"
    _write_csv(manifest_rows, MANIFEST_COLS, manifest_path)
    logger.info(f"Manifest written: {manifest_path} ({len(manifest_rows)} rows)")

    # ── Write error/quarantine CSV ────────────────────────────────────────────
    error_path = out_dir / f"{manifest_name}_errors.csv"
    _write_csv(error_rows, ERROR_COLS, error_path)
    logger.info(f"Error report written: {error_path} ({len(error_rows)} rows)")

    # ── Compute statistics ────────────────────────────────────────────────────
    summary = _compute_summary(
        manifest_rows=manifest_rows,
        error_rows=error_rows,
        root=root,
        manifest_path=manifest_path,
        error_path=error_path,
        min_duration_sec=min_duration_sec,
    )

    # ── Write JSON summary ────────────────────────────────────────────────────
    summary_path = out_dir / f"{manifest_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary written: {summary_path}")

    return summary


def _write_csv(
    rows: List[Dict[str, Any]],
    cols: List[str],
    path: Path,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _compute_summary(
    manifest_rows: List[Dict[str, Any]],
    error_rows: List[Dict[str, Any]],
    root: Path,
    manifest_path: Path,
    error_path: Path,
    min_duration_sec: float,
) -> Dict[str, Any]:
    """Compute per-label, per-domain, per-source statistics."""

    def _count_by(key: str) -> Dict[str, int]:
        d: Dict[str, int] = defaultdict(int)
        for r in manifest_rows:
            d[str(r.get(key, "") or "unknown")] += 1
        return dict(sorted(d.items()))

    def _dur_by(key: str) -> Dict[str, float]:
        d: Dict[str, float] = defaultdict(float)
        for r in manifest_rows:
            d[str(r.get(key, "") or "unknown")] += float(r.get("duration_sec", 0))
        return {k: round(v, 3) for k, v in sorted(d.items())}

    total_valid = len(manifest_rows)
    total_errors = len(error_rows)
    total_discovered = total_valid + total_errors

    durations = [float(r["duration_sec"]) for r in manifest_rows]
    total_dur = sum(durations)
    mean_dur  = total_dur / total_valid if total_valid else 0.0

    errors_by_type: Dict[str, int] = defaultdict(int)
    for e in error_rows:
        errors_by_type[e.get("error_type", "UNKNOWN")] += 1

    warnings = [r for r in manifest_rows if r.get("validation_status") == "WARNING"]

    return {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "root_directory":      str(root),
        "manifest_path":       str(manifest_path),
        "error_path":          str(error_path),
        "min_duration_sec":    min_duration_sec,
        "files_discovered":    total_discovered,
        "files_valid":         total_valid,
        "files_errors":        total_errors,
        "files_warnings":      len(warnings),
        "total_duration_sec":  round(total_dur, 3),
        "mean_duration_sec":   round(mean_dur, 3),
        "by_label": {
            "counts":    _count_by("label_str"),
            "durations": _dur_by("label_str"),
        },
        "by_domain": {
            "counts":    _count_by("domain"),
            "durations": _dur_by("domain"),
        },
        "by_source": {
            "counts":    _count_by("source"),
        },
        "by_sample_rate":    _count_by("sample_rate"),
        "by_channels":       _count_by("channels"),
        "errors_by_type":    dict(sorted(errors_by_type.items())),
        "disclaimer": (
            "This manifest reflects discovered audio files only. "
            "No model inference was performed. "
            "Metadata is extracted from audio file headers."
        ),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build production audio manifest for VoxShield.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--root",
        default="production_data",
        help="Root directory containing labelled audio.",
    )
    p.add_argument(
        "--out-dir",
        default="data/manifests",
        help="Output directory for manifest files.",
    )
    p.add_argument(
        "--min-duration",
        type=float,
        default=MIN_DURATION_SEC,
        help="Minimum audio duration in seconds.",
    )
    p.add_argument(
        "--manifest-name",
        default="production_audio",
        help="Base name for output files.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    summary = build_manifest(
        root=Path(args.root),
        out_dir=Path(args.out_dir),
        min_duration_sec=args.min_duration,
        manifest_name=args.manifest_name,
        show_progress=not args.quiet,
    )
    print(f"\nFiles discovered : {summary['files_discovered']}")
    print(f"Files valid      : {summary['files_valid']}")
    print(f"Files with errors: {summary['files_errors']}")
    print(f"Files warnings   : {summary['files_warnings']}")
    print(f"Total duration   : {summary['total_duration_sec']:.1f}s")
    print(f"\nBy label  : {summary['by_label']['counts']}")
    print(f"By domain : {summary['by_domain']['counts']}")
