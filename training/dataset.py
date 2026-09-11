"""
ASVspoof5 Dataset — Training/Validation/Evaluation Split
=========================================================

Label convention (non-negotiable):
    0 = bonafide  (genuine human speech)
    1 = spoof     (synthetic / converted speech)

Protocol columns (0-indexed):
    0: speaker_id
    1: utterance_id
    2: gender
    3-5: unused (-)
    6: codec
    7: attack_type  ("bonafide" for genuine samples, e.g. "A01"–"An" for spoof)
    8: label        ("bonafide" | "spoof")
    9: unused

Audio directory:
    <root>/train/flac_T/<utterance_id>.flac   (train set)
    <root>/dev/flac_D/<utterance_id>.flac     (dev/val set — not always present)

    When dev audio is absent, a speaker-disjoint validation split is carved
    from the training set (default 15 % of speakers).

Caching:
    An index pickle is written to <root>/.voxshield_cache/ so subsequent
    launches skip file-existence scanning.
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
CACHE_VERSION = "v3"


# ──────────────────────────────────────────────────────────────────────────────
# Index building
# ──────────────────────────────────────────────────────────────────────────────

def _parse_protocol(tsv_path: Path) -> List[Dict]:
    """Parse a single ASVspoof5 .tsv protocol file into a list of records."""
    records: List[Dict] = []
    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            speaker_id   = parts[0]
            utterance_id = parts[1]
            gender       = parts[2]
            codec        = parts[6] if parts[6] != "-" else None
            attack_type  = parts[7]  # "bonafide" or e.g. "A05"
            label_str    = parts[8]  # "bonafide" or "spoof"
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


def _find_audio(root: Path, utterance_id: str) -> Optional[Path]:
    """Locate the audio file for a given utterance_id."""
    # Train prefix T_, dev prefix D_
    if utterance_id.startswith("T_"):
        candidate = root / "train" / "flac_T" / f"{utterance_id}.flac"
    elif utterance_id.startswith("D_"):
        # dev audio may not always be present
        candidate = root / "dev" / "flac_D" / f"{utterance_id}.flac"
    else:
        candidate = root / f"{utterance_id}.flac"
    return candidate if candidate.exists() else None


def _cache_path(dataset_dir: Path, split: str) -> Path:
    """Return path for the dataset index cache."""
    key = hashlib.md5(str(dataset_dir.resolve()).encode()).hexdigest()[:8]
    cache_dir = dataset_dir / ".voxshield_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"index_{CACHE_VERSION}_{split}_{key}.pkl"


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
    """
    cache_file = _cache_path(dataset_dir, split)
    if not force_rebuild and cache_file.exists():
        logger.info(f"Loading cached index from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["train"], data["val"]

    logger.info(f"Building dataset index from {dataset_dir} ...")

    train_proto = dataset_dir / "protocols" / "ASVspoof5.train.tsv"
    dev_proto   = dataset_dir / "protocols" / "ASVspoof5.dev.track_1.tsv"

    if not train_proto.exists():
        raise FileNotFoundError(f"Training protocol not found: {train_proto}")

    # ── Parse training records ────────────────────────────────────────────────
    train_raw = _parse_protocol(train_proto)
    logger.info(f"Protocol: {len(train_raw):,} raw training entries")

    train_records: List[Dict] = []
    missing = 0
    for r in train_raw:
        audio_path = _find_audio(dataset_dir, r["utterance_id"])
        if audio_path is None:
            missing += 1
            continue
        train_records.append({**r, "path": str(audio_path)})

    if missing:
        logger.warning(f"Skipped {missing:,} training entries (audio not found)")
    logger.info(f"Resolved {len(train_records):,} training audio files")

    # ── Validation split ──────────────────────────────────────────────────────
    val_records: List[Dict] = []

    if dev_proto.exists():
        dev_raw = _parse_protocol(dev_proto)
        dev_has_audio = False
        for r in dev_raw[:20]:  # probe
            if _find_audio(dataset_dir, r["utterance_id"]) is not None:
                dev_has_audio = True
                break

        if dev_has_audio:
            logger.info("Dev protocol + audio found — using as validation set")
            for r in dev_raw:
                audio_path = _find_audio(dataset_dir, r["utterance_id"])
                if audio_path is not None:
                    val_records.append({**r, "path": str(audio_path)})
            logger.info(f"Resolved {len(val_records):,} validation (dev) audio files")

    if not val_records:
        logger.info(
            f"Dev audio not found. Carving {val_speaker_fraction:.0%} "
            "speaker-disjoint validation from training data."
        )
        train_records, val_records = _speaker_disjoint_split(
            train_records, val_speaker_fraction, seed
        )

    # ── Save cache ────────────────────────────────────────────────────────────
    with open(cache_file, "wb") as f:
        pickle.dump({"train": train_records, "val": val_records}, f)
    logger.info(
        f"Index cached: {len(train_records):,} train, "
        f"{len(val_records):,} val → {cache_file}"
    )

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

    Loads .flac audio via soundfile (torchaudio 2.11 requires this backend).
    Returns fixed-length 16 kHz mono waveforms.

    Label: 0 = bonafide, 1 = spoof.
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
        """Load .flac using soundfile, resample if needed, fix length."""
        try:
            data, sr = sf.read(path, dtype="float32")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}. Returning silence.")
            return torch.zeros(self.max_samples)

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
            # Random crop during training for augmentation
            if self.augment and n > self.max_samples:
                start = int(self._rng.integers(0, n - self.max_samples))
                wav = wav[start: start + self.max_samples]
            else:
                wav = wav[: self.max_samples]
        else:
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
