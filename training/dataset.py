"""
ASVspoof5 Dataset — Training/Validation/Evaluation Split
=========================================================

Label convention (non-negotiable):
    0 = bonafide  (genuine human speech)
    1 = spoof     (synthetic / converted speech)

Official ASVspoof5 protocol fields (0-indexed, space-separated, 10 columns):
    0: SPEAKER_ID
    1: FLAC_FILE_NAME
    2: SPEAKER_GENDER
    3: CODEC
    4: CODEC_Q
    5: CODEC_SEED
    6: ATTACK_TAG
    7: ATTACK_LABEL   ("bonafide" for genuine, e.g. "A01"–"An" for spoof)
    8: KEY            ("bonafide" | "spoof")
    9: TMP

Official native ASVspoof5 directory layout (after extracting official archives):

    <root>/
        ASVspoof5.train.tsv               (or protocols/ subdirectory — both supported)
        flac_T/
            T_*.flac
        ASVspoof5.dev.track_1.tsv
        flac_D/
            D_*.flac
        ASVspoof5.eval.track_1.tsv
        flac_E_eval/
            E_*.flac

Audio resolution:
    T_* utterances -> <root>/flac_T/<utterance_id>.flac
    D_* utterances -> <root>/flac_D/<utterance_id>.flac
    E_* utterances -> <root>/flac_E_eval/<utterance_id>.flac

Caching:
    An index pickle is written to <root>/.voxshield_cache/ so subsequent
    launches skip file-existence scanning.

    Cache is automatically invalidated and rebuilt if:
      - the protocol file content changes (SHA-256 hash mismatch)
      - the expected audio directories change
      - the record count in the cache is zero
      - sampled cached paths no longer exist on disk
      - the cache format version changes
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
MAX_SEC = 4.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_SEC)

# Increment this string whenever the cache schema or index logic changes.
# A different value forces every existing cache file to be rebuilt.
CACHE_VERSION = "v4"


# ──────────────────────────────────────────────────────────────────────────────
# Protocol discovery — supports both flat root and protocols/ subdirectory
# ──────────────────────────────────────────────────────────────────────────────

def _find_protocol(dataset_dir: Path, filename: str) -> Optional[Path]:
    """
    Locate an ASVspoof5 protocol file.

    Checks the official native flat layout first (file at dataset root),
    then falls back to a protocols/ subdirectory for backward compatibility.

    Returns the Path if found, or None.
    """
    # Native/official: TSV at root level
    candidate = dataset_dir / filename
    if candidate.exists():
        return candidate
    # Legacy / alternate layout: protocols/ subdirectory
    candidate = dataset_dir / "protocols" / filename
    if candidate.exists():
        return candidate
    return None


def _require_protocol(dataset_dir: Path, filename: str) -> Path:
    """Like _find_protocol but raises FileNotFoundError if not found."""
    p = _find_protocol(dataset_dir, filename)
    if p is None:
        raise FileNotFoundError(
            f"Protocol file '{filename}' not found in:\n"
            f"  {dataset_dir / filename}\n"
            f"  {dataset_dir / 'protocols' / filename}\n"
            "Check that the ASVspoof5 dataset is correctly extracted."
        )
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Protocol parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_protocol(tsv_path: Path) -> List[Dict]:
    """
    Parse a single ASVspoof5 .tsv protocol file into a list of records.

    Official field layout (10 space-separated columns):
        0  SPEAKER_ID
        1  FLAC_FILE_NAME
        2  SPEAKER_GENDER
        3  CODEC          ("-" for bonafide)
        4  CODEC_Q
        5  CODEC_SEED
        6  ATTACK_TAG
        7  ATTACK_LABEL   ("bonafide" or e.g. "A05")
        8  KEY            ("bonafide" | "spoof")
        9  TMP
    """
    records: List[Dict] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                logger.warning(
                    f"{tsv_path}:{lineno}: expected ≥9 fields, got {len(parts)} — skipping"
                )
                continue

            speaker_id   = parts[0]
            utterance_id = parts[1]
            gender       = parts[2]
            # Field 3 is CODEC (not field 6 — fields 4 and 5 are codec_quality
            # and codec_seed, which are "-" for bonafide utterances)
            codec        = parts[3] if parts[3] != "-" else None
            # Field 7 is the attack label ("bonafide" or "A01" etc.)
            attack_type  = parts[7]
            # Field 8 is the key ("bonafide" | "spoof")
            label_str    = parts[8]
            label = 0 if label_str == "bonafide" else 1

            records.append({
                "speaker_id":   speaker_id,
                "utterance_id": utterance_id,
                "gender":       gender,
                "codec":        codec,
                "attack_type":  attack_type,
                "label":        label,
                "label_str":    label_str,
            })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Audio path resolution
# ──────────────────────────────────────────────────────────────────────────────

def _find_audio(root: Path, utterance_id: str) -> Optional[Path]:
    """
    Locate the .flac file for a given utterance_id using the native
    official ASVspoof5 directory layout.

    Layout:
        T_* -> <root>/flac_T/<utterance_id>.flac   (training set)
        D_* -> <root>/flac_D/<utterance_id>.flac   (development set)
        E_* -> <root>/flac_E_eval/<utterance_id>.flac  (evaluation set)

    Does NOT require artificial train/ or dev/ subdirectories.
    """
    if utterance_id.startswith("T_"):
        candidate = root / "flac_T" / f"{utterance_id}.flac"
    elif utterance_id.startswith("D_"):
        candidate = root / "flac_D" / f"{utterance_id}.flac"
    elif utterance_id.startswith("E_"):
        candidate = root / "flac_E_eval" / f"{utterance_id}.flac"
    else:
        # Fallback: try at root level
        candidate = root / f"{utterance_id}.flac"
    return candidate if candidate.exists() else None


# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def _protocol_hash(proto_path: Path) -> str:
    """Return the SHA-256 hex digest of a protocol file's contents."""
    h = hashlib.sha256()
    with open(proto_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _audio_dir_fingerprint(root: Path) -> str:
    """
    Return a lightweight fingerprint of the audio directories present.
    Encodes which flac_* directories exist under root and their file counts.
    """
    dirs = ["flac_T", "flac_D", "flac_E_eval"]
    parts = []
    for d in dirs:
        p = root / d
        if p.exists():
            count = sum(1 for _ in p.glob("*.flac"))
            parts.append(f"{d}={count}")
        else:
            parts.append(f"{d}=absent")
    return "|".join(parts)


def _cache_path(dataset_dir: Path, split: str) -> Path:
    """Return the path for the dataset index cache file."""
    key = hashlib.md5(str(dataset_dir.resolve()).encode()).hexdigest()[:8]
    cache_dir = dataset_dir / ".voxshield_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"index_{CACHE_VERSION}_{split}_{key}.pkl"


def _validate_cache(
    data: dict,
    dataset_dir: Path,
    proto_hash: str,
    audio_fingerprint: str,
    n_probe: int = 20,
) -> Tuple[bool, str]:
    """
    Validate a loaded cache dict.

    Returns (is_valid, reason_if_invalid).

    Checks:
      1. cache format version matches
      2. protocol content hash matches (detects changed protocol)
      3. audio directory fingerprint matches (detects moved/added audio)
      4. train + val record counts are non-zero
      5. a random sample of cached paths still exist on disk
    """
    # 1. Version
    if data.get("cache_version") != CACHE_VERSION:
        return False, f"cache_version mismatch: {data.get('cache_version')} != {CACHE_VERSION}"

    # 2. Protocol hash
    if data.get("proto_hash") != proto_hash:
        return False, "protocol file content changed"

    # 3. Audio fingerprint
    if data.get("audio_fingerprint") != audio_fingerprint:
        return False, f"audio directory fingerprint changed: {data.get('audio_fingerprint')} -> {audio_fingerprint}"

    # 4. Non-empty
    train_recs = data.get("train", [])
    val_recs   = data.get("val",   [])
    if not train_recs:
        return False, "cached train records is empty"

    # 5. Path existence probe
    all_paths = [r["path"] for r in train_recs] + [r["path"] for r in val_recs]
    if all_paths:
        rng = random.Random(42)
        sample = rng.sample(all_paths, min(n_probe, len(all_paths)))
        missing = [p for p in sample if not Path(p).exists()]
        if missing:
            return False, (
                f"{len(missing)} of {len(sample)} probed cached paths no longer exist "
                f"(e.g. {missing[0]})"
            )

    return True, ""


# ──────────────────────────────────────────────────────────────────────────────
# Index building
# ──────────────────────────────────────────────────────────────────────────────

def build_index(
    dataset_dir: Path,
    split: str,               # "train" | "dev" | "eval"
    val_speaker_fraction: float = 0.15,
    seed: int = 42,
    force_rebuild: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build (train_records, val_records) index.

    If dev protocol + audio exist, use them as validation.
    Otherwise carve a speaker-disjoint validation split from training data.

    Returns lists of dicts with keys:
        path, speaker_id, utterance_id, gender, codec, attack_type, label, label_str

    Raises RuntimeError if train_records is empty after indexing.
    """
    train_proto_path = _require_protocol(dataset_dir, "ASVspoof5.train.tsv")
    dev_proto_path   = _find_protocol(dataset_dir, "ASVspoof5.dev.track_1.tsv")

    # Compute metadata for cache validation
    proto_hash       = _protocol_hash(train_proto_path)
    audio_fingerprint = _audio_dir_fingerprint(dataset_dir)

    cache_file = _cache_path(dataset_dir, split)

    if not force_rebuild and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            valid, reason = _validate_cache(
                data, dataset_dir, proto_hash, audio_fingerprint
            )
            if valid:
                logger.info(f"Cache valid — loaded from {cache_file}")
                logger.info(
                    f"  train={len(data['train']):,}  val={len(data['val']):,}"
                )
                return data["train"], data["val"]
            else:
                logger.warning(f"Cache invalid ({reason}) — rebuilding ...")
        except Exception as e:
            logger.warning(f"Cache unreadable ({e}) — rebuilding ...")

    logger.info(f"Building dataset index from {dataset_dir} ...")
    logger.info(f"  Protocol : {train_proto_path}")

    # ── Parse training records ────────────────────────────────────────────────
    train_raw = _parse_protocol(train_proto_path)
    logger.info(f"  Protocol entries: {len(train_raw):,}")

    train_records: List[Dict] = []
    missing = 0
    for r in train_raw:
        audio_path = _find_audio(dataset_dir, r["utterance_id"])
        if audio_path is None:
            missing += 1
            continue
        train_records.append({**r, "path": str(audio_path)})

    if missing:
        logger.warning(
            f"  {missing:,} training protocol entries had no matching audio file"
        )
    logger.info(f"  Resolved {len(train_records):,} training audio files")

    if not train_records:
        raise RuntimeError(
            f"No training audio files resolved from {dataset_dir}.\n"
            f"Expected to find .flac files under {dataset_dir / 'flac_T'}.\n"
            f"Protocol: {train_proto_path} ({len(train_raw):,} entries).\n"
            "Check that the ASVspoof5 training archives (flac_T_aa.tar … flac_T_ae.tar) "
            "have been extracted into the dataset root directory."
        )

    # ── Validation split ──────────────────────────────────────────────────────
    val_records: List[Dict] = []

    if dev_proto_path is not None:
        dev_raw = _parse_protocol(dev_proto_path)
        # Probe a handful of entries to see if dev audio is present
        dev_has_audio = False
        for r in dev_raw[:20]:
            if _find_audio(dataset_dir, r["utterance_id"]) is not None:
                dev_has_audio = True
                break

        if dev_has_audio:
            logger.info(
                f"Dev protocol found ({dev_proto_path.name}) with audio "
                f"— using as validation set"
            )
            dev_missing = 0
            for r in dev_raw:
                audio_path = _find_audio(dataset_dir, r["utterance_id"])
                if audio_path is not None:
                    val_records.append({**r, "path": str(audio_path)})
                else:
                    dev_missing += 1
            if dev_missing:
                logger.warning(
                    f"  {dev_missing:,} dev protocol entries had no matching audio"
                )
            logger.info(f"  Resolved {len(val_records):,} validation (dev) audio files")
        else:
            logger.info(
                f"Dev protocol found ({dev_proto_path.name}) but no dev audio "
                "detected in flac_D/ — falling back to speaker-disjoint split"
            )

    if not val_records:
        logger.info(
            f"Carving {val_speaker_fraction:.0%} speaker-disjoint validation "
            "split from training data."
        )
        train_records, val_records = _speaker_disjoint_split(
            train_records, val_speaker_fraction, seed
        )

    if not val_records:
        raise RuntimeError(
            "Validation set is empty after attempting both dev audio and "
            "speaker-disjoint fallback. Cannot train without validation data."
        )

    # ── Save cache with metadata ──────────────────────────────────────────────
    cache_data = {
        "cache_version":    CACHE_VERSION,
        "proto_hash":       proto_hash,
        "audio_fingerprint": audio_fingerprint,
        "train":            train_records,
        "val":              val_records,
    }
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(
            f"  Index cached: {len(train_records):,} train, "
            f"{len(val_records):,} val → {cache_file}"
        )
    except Exception as e:
        logger.warning(f"Could not write cache ({e}) — continuing without cache")

    return train_records, val_records


def _speaker_disjoint_split(
    records: List[Dict],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Split records by speaker to ensure no speaker overlap."""
    rng = random.Random(seed)
    speakers = sorted({r["speaker_id"] for r in records})
    n_val_speakers = max(1, int(len(speakers) * val_fraction))
    rng.shuffle(speakers)
    val_speakers = set(speakers[:n_val_speakers])
    train = [r for r in records if r["speaker_id"] not in val_speakers]
    val   = [r for r in records if r["speaker_id"] in val_speakers]
    logger.info(
        f"Speaker split: {len(train):,} train / {len(val):,} val "
        f"({len(speakers) - n_val_speakers} / {n_val_speakers} speakers)"
    )
    return train, val


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────

class ASVspoof5Dataset(Dataset):
    """
    PyTorch Dataset for ASVspoof5.

    Loads .flac audio via soundfile.
    Returns fixed-length 16 kHz mono waveforms.

    Label: 0 = bonafide, 1 = spoof.

    Audio loading behaviour:
      - If audio is shorter than max_samples, it is zero-padded.
      - If audio is longer than max_samples, it is clipped (random crop
        during augmentation, deterministic head-crop otherwise).
      - If audio is genuinely unreadable/corrupt, a RuntimeError is raised
        with the file path so the problem is visible and actionable.
        Training infrastructure must decide whether to skip or abort.
    """

    def __init__(
        self,
        records: List[Dict],
        sample_rate: int = SAMPLE_RATE,
        max_samples: int = MAX_SAMPLES,
        augment: bool = False,
        seed: int = 42,
    ):
        self.records = records
        self.sample_rate = int(sample_rate)
        # Coerce to int: config values such as max_length_sec * sample_rate
        # produce a float (e.g. 4.0 * 16000 = 64000.0) which would cause
        # F.pad to receive a float tuple and raise TypeError.
        self.max_samples = int(max_samples)
        self.augment = augment
        self._rng = np.random.default_rng(seed)

        # Class balance info
        labels = [r["label"] for r in records]
        n_bf = sum(1 for l in labels if l == 0)
        n_sp = sum(1 for l in labels if l == 1)
        if len(records) > 0:
            logger.info(
                f"Dataset: {len(records):,} samples | "
                f"bonafide={n_bf:,} ({n_bf/len(records):.1%}) | "
                f"spoof={n_sp:,} ({n_sp/len(records):.1%})"
            )
        else:
            logger.info("Dataset: 0 samples")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        rec = self.records[idx]
        wav = self._load_audio(rec["path"])
        if self.augment:
            wav = self._augment(wav)
        meta = {
            "speaker_id":   rec["speaker_id"],
            "utterance_id": rec["utterance_id"],
            "attack_type":  rec["attack_type"],
            "codec":        rec.get("codec"),
            "label_str":    rec["label_str"],
        }
        return wav, rec["label"], meta

    def _load_audio(self, path: str) -> torch.Tensor:
        """
        Load .flac using soundfile, resample if needed, fix length.

        Raises RuntimeError for corrupt/unreadable files.
        Padding is always zero-padding for short files (normal behaviour).
        """
        if not Path(path).exists():
            raise RuntimeError(
                f"Audio file not found: {path}\n"
                "This path was resolved during indexing but no longer exists. "
                "Re-run with --force-rebuild-index to refresh the index."
            )

        try:
            data, sr = sf.read(path, dtype="float32")
        except Exception as e:
            raise RuntimeError(
                f"Cannot read audio file: {path}\n"
                f"Error: {e}\n"
                "The file may be corrupt or truncated. "
                "Do not silently use zeros — this hides dataset problems."
            ) from e

        # Mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        wav = torch.from_numpy(data)

        # Resample if needed (shouldn't happen with ASVspoof5 but defensive)
        if sr != self.sample_rate:
            import torchaudio.functional as AF
            wav = AF.resample(wav.unsqueeze(0), sr, self.sample_rate).squeeze(0)

        # Clip or pad
        n = wav.shape[0]
        if n >= self.max_samples:
            # Random crop during training augmentation
            if self.augment and n > self.max_samples:
                start = int(self._rng.integers(0, n - self.max_samples))
                wav = wav[start: start + self.max_samples]
            else:
                wav = wav[: self.max_samples]
        else:
            # Zero-pad (normal — many utterances are shorter than 4 s)
            wav = torch.nn.functional.pad(wav, (0, self.max_samples - n))

        return wav  # shape: (max_samples,)

    def _augment(self, wav: torch.Tensor) -> torch.Tensor:
        """Light augmentation: additive Gaussian noise + amplitude scale."""
        # Amplitude scale ±20 %
        scale = float(self._rng.uniform(0.8, 1.2))
        wav = wav * scale
        # Additive noise SNR ~ 30–50 dB
        if self._rng.random() < 0.5:
            sig_power = wav.pow(2).mean().clamp(min=1e-9)
            snr_db = float(self._rng.uniform(30, 50))
            noise_power = sig_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(wav) * noise_power.sqrt()
            wav = wav + noise
        return wav

    # ── Class weights for imbalanced datasets ─────────────────────────────────

    def class_weights(self) -> torch.Tensor:
        """Return [w_bonafide, w_spoof] for use with WeightedRandomSampler."""
        labels = [r["label"] for r in self.records]
        n_total = len(labels)
        n_bf = sum(1 for l in labels if l == 0)
        n_sp = n_total - n_bf
        w_bf = n_total / (2 * n_bf) if n_bf > 0 else 1.0
        w_sp = n_total / (2 * n_sp) if n_sp > 0 else 1.0
        return torch.tensor([w_bf, w_sp], dtype=torch.float32)

    def sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        cw = self.class_weights()
        return torch.tensor(
            [float(cw[r["label"]]) for r in self.records], dtype=torch.float32
        )


# ──────────────────────────────────────────────────────────────────────────────
# Collate function
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Stack waveforms and labels; gather metadata as list."""
    wavs, labels, metas = zip(*batch)
    return (
        torch.stack(wavs),
        torch.tensor(labels, dtype=torch.float32),
        list(metas),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset statistics helper
# ──────────────────────────────────────────────────────────────────────────────

def dataset_stats(records: List[Dict]) -> Dict:
    """Return summary statistics for a record list."""
    labels = [r["label"] for r in records]
    attack_types = {}
    speakers = set()
    for r in records:
        speakers.add(r["speaker_id"])
        at = r["attack_type"]
        attack_types[at] = attack_types.get(at, 0) + 1

    n = len(records)
    n_bf = sum(1 for l in labels if l == 0)
    n_sp = n - n_bf
    return {
        "total": n,
        "bonafide": n_bf,
        "spoof": n_sp,
        "bonafide_pct": round(n_bf / n * 100, 1) if n > 0 else 0.0,
        "spoof_pct": round(n_sp / n * 100, 1) if n > 0 else 0.0,
        "unique_speakers": len(speakers),
        "attack_types": dict(sorted(attack_types.items())),
    }
