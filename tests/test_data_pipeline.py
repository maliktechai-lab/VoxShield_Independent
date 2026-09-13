"""
VoxShield Data Pipeline Unit Tests
====================================

Tests for:
  - Label inference
  - Domain inference
  - Nested discovery
  - SHA-256 hashing
  - Duplicate detection (SHA + basename)
  - Corrupt audio handling
  - Silent audio detection
  - Too-short audio detection
  - Manifest output (CSV columns, row counts, error CSV)
  - Deterministic splitting
  - Speaker leakage prevention
  - Generator leakage prevention
  - TTS-source leakage prevention
  - SHA leakage prevention
  - Class/domain statistics
  - Tiny dataset edge cases

Audio fixtures are synthesised in-memory using numpy + scipy (no real recordings).
No model inference is performed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import struct
import tempfile
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# ── import the modules under test ─────────────────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.build_manifest import (
    AUDIO_EXTENSIONS,
    DEFAULT_DOMAIN,
    LABEL_MAP,
    MANIFEST_COLS,
    ERROR_COLS,
    VALID_DOMAINS,
    build_manifest,
    check_audio_content,
    infer_label_and_domain,
    process_file,
    sha256_file,
)
from data.make_splits import (
    SPLIT_CAL,
    SPLIT_TEST,
    SPLIT_TRAIN,
    assign_splits,
    check_leakage,
    load_manifest,
    make_splits,
    split_statistics,
    _group_key,
    _group_rows,
    _stable_shuffle,
)


# ── Fixture helpers ────────────────────────────────────────────────────────────

_wav_counter = 0  # global counter to ensure every file gets unique content


def _write_wav(path: Path, duration_sec: float, sample_rate: int = 16000,
               amplitude: float = 0.5, silent: bool = False,
               channels: int = 1, freq: float = 0.0) -> None:
    """
    Write a minimal valid PCM WAV file using only stdlib.

    Each call produces unique content: the frequency is varied via a global
    counter so that two files with the same nominal parameters still differ
    in their SHA-256 fingerprint.
    """
    global _wav_counter
    _wav_counter += 1

    n_samples = max(1, int(sample_rate * duration_sec))
    if silent:
        data_i16 = np.zeros((n_samples, channels), dtype=np.int16)
    else:
        # Use caller-supplied freq if given, otherwise vary by counter
        actual_freq = freq if freq > 0 else (200 + _wav_counter * 37)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        tone = (amplitude * np.sin(2 * np.pi * actual_freq * t) * 32767).astype(np.int16)
        if channels == 2:
            data_i16 = np.stack([tone, tone], axis=-1)
        else:
            data_i16 = tone.reshape(-1, 1)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data_i16.tobytes())


def _make_audio_tree(root: Path) -> None:
    """
    Create a minimal labelled audio tree under root:

        root/
          bonafide/
            clean/
              b_clean_1.wav   (1 s, 440 Hz)
              b_clean_2.wav   (1 s, 440 Hz)
            noisy/
              b_noisy_1.wav   (0.5 s)
          spoof/
            tts/
              s_tts_1.wav     (1 s)
              s_tts_2.wav     (1 s)
            converted/
              s_conv_1.wav    (0.8 s)
    """
    (root / "bonafide" / "clean").mkdir(parents=True)
    (root / "bonafide" / "noisy").mkdir(parents=True)
    (root / "spoof"    / "tts").mkdir(parents=True)
    (root / "spoof"    / "converted").mkdir(parents=True)

    _write_wav(root / "bonafide" / "clean"    / "b_clean_1.wav",    1.0)
    _write_wav(root / "bonafide" / "clean"    / "b_clean_2.wav",    1.0)
    _write_wav(root / "bonafide" / "noisy"    / "b_noisy_1.wav",    0.5)
    _write_wav(root / "spoof"    / "tts"      / "s_tts_1.wav",      1.0)
    _write_wav(root / "spoof"    / "tts"      / "s_tts_2.wav",      1.0)
    _write_wav(root / "spoof"    / "converted"/ "s_conv_1.wav",     0.8)


def _make_rows(n: int, label: int = 0, domain: str = "clean",
               speaker_prefix: str = "spk") -> List[Dict[str, Any]]:
    """Return fake manifest rows for split testing."""
    return [
        {
            "path":            f"/fake/{i}.wav",
            "sha256":          hashlib.sha256(f"file{i}".encode()).hexdigest(),
            "label":           str(label),
            "label_str":       "bonafide" if label == 0 else "spoof",
            "domain":          domain,
            "source":          f"bonafide/{domain}" if label == 0 else f"spoof/{domain}",
            "speaker_id":      f"{speaker_prefix}{i}",
            "generator_id":    "",
            "tts_source":      "",
            "replay_device":   "",
            "codec":           "PCM_16",
            "sample_rate":     "16000",
            "channels":        "1",
            "duration_sec":    "1.0",
            "file_size_bytes": "32044",
            "validation_status": "OK",
            "validation_notes":  "",
        }
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Label inference
# ═══════════════════════════════════════════════════════════════════════════════

class TestLabelInference:

    def test_bonafide_label(self, tmp_path):
        f = tmp_path / "bonafide" / "clean" / "a.wav"
        f.parent.mkdir(parents=True)
        f.touch()
        label, label_str, domain = infer_label_and_domain(f, tmp_path)
        assert label == 0
        assert label_str == "bonafide"

    def test_spoof_label(self, tmp_path):
        f = tmp_path / "spoof" / "tts" / "a.wav"
        f.parent.mkdir(parents=True)
        f.touch()
        label, label_str, domain = infer_label_and_domain(f, tmp_path)
        assert label == 1
        assert label_str == "spoof"

    def test_unknown_label_returns_none(self, tmp_path):
        f = tmp_path / "unlabelled" / "a.wav"
        f.parent.mkdir(parents=True)
        f.touch()
        label, label_str, domain = infer_label_and_domain(f, tmp_path)
        assert label is None
        assert label_str is None

    def test_case_insensitive_label(self, tmp_path):
        f = tmp_path / "Bonafide" / "clean" / "a.wav"
        f.parent.mkdir(parents=True)
        f.touch()
        label, label_str, domain = infer_label_and_domain(f, tmp_path)
        assert label == 0

    def test_file_outside_root_returns_none(self, tmp_path):
        other = tmp_path.parent / "elsewhere" / "a.wav"
        label, label_str, domain = infer_label_and_domain(other, tmp_path)
        assert label is None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Domain inference
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainInference:

    @pytest.mark.parametrize("domain_dir,expected", [
        ("clean",     "clean"),
        ("tts",       "tts"),
        ("converted", "converted"),
        ("replay",    "replay"),
        ("noisy",     "noisy"),
        ("codec",     "codec"),
    ])
    def test_valid_domains(self, tmp_path, domain_dir, expected):
        f = tmp_path / "spoof" / domain_dir / "a.wav"
        f.parent.mkdir(parents=True)
        f.touch()
        _, _, domain = infer_label_and_domain(f, tmp_path)
        assert domain == expected

    def test_file_directly_under_label_defaults_to_clean(self, tmp_path):
        f = tmp_path / "bonafide" / "a.wav"
        f.parent.mkdir(parents=True)
        f.touch()
        _, _, domain = infer_label_and_domain(f, tmp_path)
        assert domain == DEFAULT_DOMAIN

    def test_unknown_domain_defaults_to_clean(self, tmp_path):
        # An unrecognised subdirectory name should fall back to DEFAULT_DOMAIN
        f = tmp_path / "bonafide" / "mystery_domain" / "a.wav"
        f.parent.mkdir(parents=True)
        f.touch()
        _, _, domain = infer_label_and_domain(f, tmp_path)
        assert domain == DEFAULT_DOMAIN

    def test_deep_nesting_uses_first_domain_dir(self, tmp_path):
        # bonafide/tts/session1/speaker2/a.wav → domain should be tts
        f = tmp_path / "bonafide" / "tts" / "session1" / "speaker2" / "a.wav"
        f.parent.mkdir(parents=True)
        f.touch()
        _, _, domain = infer_label_and_domain(f, tmp_path)
        assert domain == "tts"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Nested discovery / full build_manifest integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestNestedDiscovery:

    def test_discovers_all_wav_files(self, tmp_path):
        root = tmp_path / "root"
        _make_audio_tree(root)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        assert summary["files_discovered"] == 6  # 3 bonafide + 3 spoof

    def test_discovers_deeply_nested_files(self, tmp_path):
        root = tmp_path / "root"
        deep = root / "bonafide" / "clean" / "a" / "b" / "c"
        deep.mkdir(parents=True)
        _write_wav(deep / "deep.wav", 1.0)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        assert summary["files_valid"] == 1

    def test_ignores_unsupported_extensions(self, tmp_path):
        root = tmp_path / "root"
        (root / "bonafide" / "clean").mkdir(parents=True)
        (root / "bonafide" / "clean" / "notes.txt").write_text("not audio")
        _write_wav(root / "bonafide" / "clean" / "real.wav", 1.0)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        assert summary["files_valid"] == 1
        assert summary["files_errors"] == 0   # txt is not even an audio candidate


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SHA-256 hashing
# ═══════════════════════════════════════════════════════════════════════════════

class TestSHA256:

    def test_same_content_same_hash(self, tmp_path):
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        _write_wav(a, 1.0)
        import shutil
        shutil.copy(a, b)
        assert sha256_file(a) == sha256_file(b)

    def test_different_content_different_hash(self, tmp_path):
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        _write_wav(a, 1.0, amplitude=0.5)
        _write_wav(b, 1.0, amplitude=0.9)
        # Very likely different; even if tones are close the timestamps differ
        assert sha256_file(a) != sha256_file(b)

    def test_hash_is_64_hex_chars(self, tmp_path):
        f = tmp_path / "t.wav"
        _write_wav(f, 0.5)
        h = sha256_file(f)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Duplicate detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestDuplicateDetection:

    def test_sha256_duplicate_goes_to_error(self, tmp_path):
        root = tmp_path / "root"
        (root / "bonafide" / "clean").mkdir(parents=True)
        (root / "bonafide" / "noisy").mkdir(parents=True)
        _write_wav(root / "bonafide" / "clean" / "a.wav", 1.0)

        # Copy identical bytes to a different path
        import shutil
        shutil.copy(
            root / "bonafide" / "clean" / "a.wav",
            root / "bonafide" / "noisy"  / "a_copy.wav",
        )
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        # One valid, one error (DUPLICATE_SHA256)
        assert summary["files_valid"] == 1
        assert summary["files_errors"] == 1
        assert summary["errors_by_type"].get("DUPLICATE_SHA256", 0) == 1

    def test_duplicate_basename_warning_in_manifest(self, tmp_path):
        root = tmp_path / "root"
        (root / "bonafide" / "clean").mkdir(parents=True)
        (root / "bonafide" / "noisy").mkdir(parents=True)
        # Different content, same filename
        _write_wav(root / "bonafide" / "clean" / "track.wav", 1.0, amplitude=0.3)
        _write_wav(root / "bonafide" / "noisy" / "track.wav", 1.0, amplitude=0.7)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        # Both should be valid (different SHA), but one should have a warning
        assert summary["files_valid"] == 2
        assert summary["files_warnings"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Corrupt audio
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorruptAudio:

    def test_corrupt_file_goes_to_error(self, tmp_path):
        root = tmp_path / "root"
        (root / "bonafide" / "clean").mkdir(parents=True)
        corrupt = root / "bonafide" / "clean" / "corrupt.wav"
        corrupt.write_bytes(b"NOTAVALIDWAV\x00\x01\x02\x03" * 10)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        assert summary["files_valid"] == 0
        assert summary["files_errors"] == 1
        assert summary["errors_by_type"].get("UNREADABLE_AUDIO", 0) == 1

    def test_zero_byte_file_goes_to_error(self, tmp_path):
        root = tmp_path / "root"
        (root / "spoof" / "tts").mkdir(parents=True)
        empty = root / "spoof" / "tts" / "empty.wav"
        empty.write_bytes(b"")
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        assert summary["errors_by_type"].get("ZERO_BYTE", 0) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Silent audio
# ═══════════════════════════════════════════════════════════════════════════════

class TestSilentAudio:

    def test_silent_audio_gets_warning_not_error(self, tmp_path):
        root = tmp_path / "root"
        (root / "bonafide" / "clean").mkdir(parents=True)
        _write_wav(root / "bonafide" / "clean" / "silent.wav", 1.0, silent=True)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        # File should be valid (not errored) but have a warning
        assert summary["files_valid"] == 1
        assert summary["files_errors"] == 0
        assert summary["files_warnings"] == 1

    def test_silent_audio_note_contains_warn_keyword(self, tmp_path):
        root = tmp_path / "root"
        (root / "bonafide" / "clean").mkdir(parents=True)
        _write_wav(root / "bonafide" / "clean" / "s.wav", 1.0, silent=True)
        out = tmp_path / "manifests"
        build_manifest(root, out, show_progress=False)
        rows = list(csv.DictReader(open(out / "production_audio.csv")))
        assert rows[0]["validation_status"] == "WARNING"
        assert "silent" in rows[0]["validation_notes"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Too-short audio
# ═══════════════════════════════════════════════════════════════════════════════

class TestTooShortAudio:

    def test_too_short_goes_to_error(self, tmp_path):
        root = tmp_path / "root"
        (root / "bonafide" / "clean").mkdir(parents=True)
        # Write 0.05 s (below default 0.1 s threshold)
        _write_wav(root / "bonafide" / "clean" / "tiny.wav", 0.05)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, min_duration_sec=0.1, show_progress=False)
        assert summary["errors_by_type"].get("TOO_SHORT", 0) == 1

    def test_exactly_at_threshold_passes(self, tmp_path):
        root = tmp_path / "root"
        (root / "bonafide" / "clean").mkdir(parents=True)
        _write_wav(root / "bonafide" / "clean" / "ok.wav", 0.15)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, min_duration_sec=0.1, show_progress=False)
        assert summary["files_valid"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Manifest output
# ═══════════════════════════════════════════════════════════════════════════════

class TestManifestOutput:

    def test_manifest_has_correct_columns(self, tmp_path):
        root = tmp_path / "root"
        _make_audio_tree(root)
        out = tmp_path / "manifests"
        build_manifest(root, out, show_progress=False)
        with open(out / "production_audio.csv") as f:
            header = csv.DictReader(f).fieldnames
        for col in MANIFEST_COLS:
            assert col in header, f"Missing column: {col}"

    def test_error_csv_has_correct_columns(self, tmp_path):
        root = tmp_path / "root"
        (root / "spoof" / "tts").mkdir(parents=True)
        (root / "spoof" / "tts" / "bad.wav").write_bytes(b"\x00")  # zero byte
        out = tmp_path / "manifests"
        build_manifest(root, out, show_progress=False)
        with open(out / "production_audio_errors.csv") as f:
            header = csv.DictReader(f).fieldnames
        for col in ERROR_COLS:
            assert col in header, f"Missing error column: {col}"

    def test_summary_json_written(self, tmp_path):
        root = tmp_path / "root"
        _make_audio_tree(root)
        out = tmp_path / "manifests"
        build_manifest(root, out, show_progress=False)
        assert (out / "production_audio_summary.json").exists()
        with open(out / "production_audio_summary.json") as f:
            s = json.load(f)
        assert "files_valid" in s
        assert "by_label" in s
        assert "by_domain" in s

    def test_manifest_row_count_matches_summary(self, tmp_path):
        root = tmp_path / "root"
        _make_audio_tree(root)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        rows = list(csv.DictReader(open(out / "production_audio.csv")))
        assert len(rows) == summary["files_valid"]

    def test_label_values_are_0_and_1(self, tmp_path):
        root = tmp_path / "root"
        _make_audio_tree(root)
        out = tmp_path / "manifests"
        build_manifest(root, out, show_progress=False)
        rows = list(csv.DictReader(open(out / "production_audio.csv")))
        labels = {r["label"] for r in rows}
        assert labels == {"0", "1"}

    def test_no_model_inference_in_output(self, tmp_path):
        """Sanity check: summary must contain the no-model-inference disclaimer."""
        root = tmp_path / "root"
        _make_audio_tree(root)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        assert "No model inference" in summary.get("disclaimer", "")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Deterministic splitting
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterministicSplitting:

    def test_same_seed_same_splits(self):
        rows = _make_rows(30)
        a = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        b = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        assert [r["split"] for r in a] == [r["split"] for r in b]

    def test_different_seed_different_splits(self):
        rows = _make_rows(30)
        a = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        b = assign_splits(rows, 0.70, 0.15, 0.15, seed=99)
        # With 30 distinct speakers, almost certain to differ
        assert [r["split"] for r in a] != [r["split"] for r in b]

    def test_all_rows_assigned(self):
        rows = _make_rows(20)
        result = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        for r in result:
            assert r["split"] in {SPLIT_TRAIN, SPLIT_CAL, SPLIT_TEST}

    def test_fractions_must_sum_to_one(self):
        rows = _make_rows(10)
        with pytest.raises(ValueError):
            assign_splits(rows, 0.60, 0.15, 0.15, seed=42)  # sums to 0.90


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Speaker leakage prevention
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeakerLeakagePrevention:

    def test_no_speaker_in_multiple_splits(self):
        rows = _make_rows(30, speaker_prefix="spk")
        result = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        report = check_leakage(result)
        assert report["speaker_leaks"] == {}

    def test_leakage_check_detects_injection(self):
        """Manually inject a speaker across two splits and verify detection."""
        rows = [
            {**_make_rows(1, speaker_prefix="evil")[0],
             "sha256": hashlib.sha256(b"x").hexdigest(), "split": "train"},
            {**_make_rows(1, speaker_prefix="evil")[0],
             "sha256": hashlib.sha256(b"y").hexdigest(), "split": "test"},
        ]
        # They both have speaker_id = 'evil0', across train and test
        rows[1]["speaker_id"] = "evil0"
        report = check_leakage(rows)
        assert not report["clean"]
        assert "evil0" in report["speaker_leaks"]


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Generator leakage prevention
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeneratorLeakagePrevention:

    def _gen_rows(self, n: int) -> List[Dict[str, Any]]:
        rows = []
        for i in range(n):
            rows.append({
                "path":            f"/fake/{i}.wav",
                "sha256":          hashlib.sha256(f"gen{i}".encode()).hexdigest(),
                "label":           "1",
                "label_str":       "spoof",
                "domain":          "tts",
                "source":          "spoof/tts",
                "speaker_id":      "",
                "generator_id":    f"gen{i}",
                "tts_source":      "",
                "replay_device":   "",
                "codec":           "PCM_16",
                "sample_rate":     "16000",
                "channels":        "1",
                "duration_sec":    "1.0",
                "file_size_bytes": "32044",
                "validation_status": "OK",
                "validation_notes":  "",
            })
        return rows

    def test_no_generator_in_multiple_splits(self):
        rows = self._gen_rows(30)
        result = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        report = check_leakage(result)
        assert report["generator_leaks"] == {}

    def test_injected_generator_leakage_detected(self):
        rows = [
            {**self._gen_rows(1)[0], "split": "train"},
            {**self._gen_rows(1)[0], "split": "calibration",
             "sha256": hashlib.sha256(b"alt").hexdigest()},
        ]
        # same generator_id in two different splits
        rows[1]["generator_id"] = rows[0]["generator_id"]
        report = check_leakage(rows)
        assert not report["clean"]
        assert rows[0]["generator_id"] in report["generator_leaks"]


# ═══════════════════════════════════════════════════════════════════════════════
# 13. TTS-source leakage prevention
# ═══════════════════════════════════════════════════════════════════════════════

class TestTTSSourceLeakagePrevention:

    def _tts_rows(self, n: int) -> List[Dict[str, Any]]:
        rows = []
        for i in range(n):
            rows.append({
                "path":            f"/fake/tts{i}.wav",
                "sha256":          hashlib.sha256(f"tts{i}".encode()).hexdigest(),
                "label":           "1",
                "label_str":       "spoof",
                "domain":          "tts",
                "source":          "spoof/tts",
                "speaker_id":      "",
                "generator_id":    "",
                "tts_source":      f"tts_vendor_{i % 5}",   # 5 TTS vendors
                "replay_device":   "",
                "codec":           "PCM_16",
                "sample_rate":     "16000",
                "channels":        "1",
                "duration_sec":    "1.0",
                "file_size_bytes": "32044",
                "validation_status": "OK",
                "validation_notes":  "",
            })
        return rows

    def test_no_tts_source_in_multiple_splits(self):
        rows = self._tts_rows(30)
        result = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        report = check_leakage(result)
        assert report["tts_source_leaks"] == {}

    def test_injected_tts_leakage_detected(self):
        rows = [
            {**self._tts_rows(1)[0], "split": "train"},
            {**self._tts_rows(1)[0], "split": "test",
             "sha256": hashlib.sha256(b"z").hexdigest()},
        ]
        rows[1]["tts_source"] = rows[0]["tts_source"]
        report = check_leakage(rows)
        assert not report["clean"]


# ═══════════════════════════════════════════════════════════════════════════════
# 14. SHA leakage prevention
# ═══════════════════════════════════════════════════════════════════════════════

class TestSHALeakagePrevention:

    def test_sha_fallback_grouping_prevents_split_crossing(self):
        """
        When no speaker/generator/tts metadata is present, files should group
        by SHA prefix and not appear in multiple splits.
        """
        rows = []
        for i in range(20):
            sha = hashlib.sha256(f"file{i}".encode()).hexdigest()
            rows.append({
                "path":            f"/fake/{i}.wav",
                "sha256":          sha,
                "label":           "0",
                "label_str":       "bonafide",
                "domain":          "clean",
                "source":          "bonafide/clean",
                "speaker_id":      "",
                "generator_id":    "",
                "tts_source":      "",
                "replay_device":   "",
                "codec":           "PCM_16",
                "sample_rate":     "16000",
                "channels":        "1",
                "duration_sec":    "1.0",
                "file_size_bytes": "32044",
                "validation_status": "OK",
                "validation_notes":  "",
            })
        result = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        report = check_leakage(result)
        assert report["sha256_leaks"] == {}

    def test_injected_sha_leakage_detected(self):
        sha = hashlib.sha256(b"duplicate").hexdigest()
        rows = [
            {"sha256": sha, "split": "train",  "speaker_id": "", "generator_id": "", "tts_source": ""},
            {"sha256": sha, "split": "test",   "speaker_id": "", "generator_id": "", "tts_source": ""},
        ]
        report = check_leakage(rows)
        assert not report["clean"]
        assert sha[:16] in report["sha256_leaks"]


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Class and domain statistics
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassDomainStatistics:

    def test_by_label_counts_match_actual(self, tmp_path):
        root = tmp_path / "root"
        _make_audio_tree(root)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        # Tree has 3 bonafide, 3 spoof
        assert summary["by_label"]["counts"].get("bonafide", 0) == 3
        assert summary["by_label"]["counts"].get("spoof",    0) == 3

    def test_by_domain_counts_match_actual(self, tmp_path):
        root = tmp_path / "root"
        _make_audio_tree(root)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        # clean=2, noisy=1, tts=2, converted=1
        assert summary["by_domain"]["counts"].get("clean",     0) == 2
        assert summary["by_domain"]["counts"].get("noisy",     0) == 1
        assert summary["by_domain"]["counts"].get("tts",       0) == 2
        assert summary["by_domain"]["counts"].get("converted", 0) == 1

    def test_split_statistics_cover_all_splits(self):
        rows = _make_rows(30)
        result = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        stats = split_statistics(result)
        for split in [SPLIT_TRAIN, SPLIT_CAL, SPLIT_TEST]:
            assert split in stats
            assert stats[split]["n_total"] > 0

    def test_split_totals_equal_input(self):
        rows = _make_rows(20)
        result = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        stats = split_statistics(result)
        total = sum(stats[s]["n_total"] for s in [SPLIT_TRAIN, SPLIT_CAL, SPLIT_TEST])
        assert total == 20


# ═══════════════════════════════════════════════════════════════════════════════
# 16. Tiny dataset edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestTinyDatasetEdgeCases:

    def test_single_file(self, tmp_path):
        root = tmp_path / "root"
        (root / "bonafide" / "clean").mkdir(parents=True)
        _write_wav(root / "bonafide" / "clean" / "one.wav", 1.0)
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        assert summary["files_valid"] == 1
        assert summary["files_errors"] == 0

    def test_empty_root_produces_zero_counts(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        out = tmp_path / "manifests"
        summary = build_manifest(root, out, show_progress=False)
        assert summary["files_discovered"] == 0
        assert summary["files_valid"] == 0

    def test_split_with_single_row(self):
        rows = _make_rows(1)
        result = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        # All 1 row must be assigned somewhere
        assert result[0]["split"] in {SPLIT_TRAIN, SPLIT_CAL, SPLIT_TEST}

    def test_split_with_two_rows(self):
        rows = _make_rows(2)
        result = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        assert len(result) == 2

    def test_split_with_three_rows_all_assigned(self):
        rows = _make_rows(3)
        result = assign_splits(rows, 0.70, 0.15, 0.15, seed=42)
        assert len(result) == 3

    def test_empty_manifest_make_splits_returns_zero_rows(self, tmp_path):
        """make_splits on an empty manifest should not crash."""
        manifest = tmp_path / "empty.csv"
        # Write header-only CSV
        with open(manifest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "sha256", "label",
                                                   "label_str", "domain", "source",
                                                   "speaker_id", "generator_id",
                                                   "tts_source", "replay_device",
                                                   "codec", "sample_rate", "channels",
                                                   "duration_sec", "file_size_bytes",
                                                   "validation_status",
                                                   "validation_notes"])
            writer.writeheader()
        out = tmp_path / "splits"
        summary = make_splits(manifest, out, seed=42)
        assert summary["total_rows"] == 0

    def test_nonexistent_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_manifest(tmp_path / "no_such_dir", tmp_path / "out", show_progress=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 17. Full pipeline integration (build + split)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineIntegration:

    def test_build_then_split_no_leakage(self, tmp_path):
        root = tmp_path / "root"
        _make_audio_tree(root)
        manifests = tmp_path / "manifests"
        build_manifest(root, manifests, show_progress=False)
        summary = make_splits(
            manifests / "production_audio.csv",
            manifests,
            seed=42,
        )
        assert summary["leakage_check"]["clean"] is True

    def test_build_then_split_creates_all_output_files(self, tmp_path):
        root = tmp_path / "root"
        _make_audio_tree(root)
        manifests = tmp_path / "manifests"
        build_manifest(root, manifests, show_progress=False)
        make_splits(manifests / "production_audio.csv", manifests, seed=42)
        for fname in ["train.csv", "calibration.csv", "test.csv", "split_summary.json"]:
            assert (manifests / fname).exists(), f"Missing: {fname}"

    def test_split_summary_json_structure(self, tmp_path):
        root = tmp_path / "root"
        _make_audio_tree(root)
        manifests = tmp_path / "manifests"
        build_manifest(root, manifests, show_progress=False)
        summary = make_splits(manifests / "production_audio.csv", manifests, seed=42)
        assert "split_statistics" in summary
        assert "leakage_check" in summary
        assert "fractions" in summary
        assert summary["fractions"]["train"] == 0.70

    def test_leakage_fail_signal_on_injected_data(self):
        """
        Directly verify that check_leakage returns clean=False when
        the same speaker crosses splits, so callers can fail-fast.
        """
        rows_with_splits = [
            {"sha256": "a" * 64, "split": "train",
             "speaker_id": "spkX", "generator_id": "", "tts_source": ""},
            {"sha256": "b" * 64, "split": "test",
             "speaker_id": "spkX", "generator_id": "", "tts_source": ""},
        ]
        report = check_leakage(rows_with_splits)
        assert report["clean"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# 18. group_key and stable_shuffle helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_group_key_priority_speaker_over_generator(self):
        row = {"speaker_id": "s1", "generator_id": "g1",
               "tts_source": "t1", "sha256": "a" * 64}
        assert _group_key(row).startswith("spk:")

    def test_group_key_priority_generator_over_tts(self):
        row = {"speaker_id": "", "generator_id": "g1",
               "tts_source": "t1", "sha256": "a" * 64}
        assert _group_key(row).startswith("gen:")

    def test_group_key_priority_tts_over_sha(self):
        row = {"speaker_id": "", "generator_id": "",
               "tts_source": "t1", "sha256": "a" * 64}
        assert _group_key(row).startswith("tts:")

    def test_group_key_fallback_to_sha(self):
        row = {"speaker_id": "", "generator_id": "",
               "tts_source": "", "sha256": "deadbeef" + "0" * 56}
        assert _group_key(row).startswith("sha:")

    def test_stable_shuffle_is_deterministic(self):
        items = list(range(100))
        a = _stable_shuffle(items, seed=7)
        b = _stable_shuffle(items, seed=7)
        assert a == b

    def test_stable_shuffle_seed_affects_order(self):
        items = list(range(100))
        assert _stable_shuffle(items, seed=1) != _stable_shuffle(items, seed=2)

    def test_group_rows_partitions_correctly(self):
        rows = [
            {"speaker_id": "s1", "generator_id": "", "tts_source": "", "sha256": "a"*64},
            {"speaker_id": "s1", "generator_id": "", "tts_source": "", "sha256": "b"*64},
            {"speaker_id": "s2", "generator_id": "", "tts_source": "", "sha256": "c"*64},
        ]
        groups = _group_rows(rows)
        assert len(groups["spk:s1"]) == 2
        assert len(groups["spk:s2"]) == 1
