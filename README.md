# VoxShield Independent

**Voice Threat Intelligence — Local Anti-Spoofing Detection**

> VoxShieldNet is trained from random initialization and does not use pretrained speech models or external AI inference APIs.

---

## Overview

VoxShield Independent is a local, privacy-preserving system for detecting synthetic or AI-generated speech. It detects whether a given audio clip is genuine human speech (bonafide) or AI-generated/voice-converted audio (spoof).

All inference runs on your machine. No audio data is sent to external services. No cloud API is used.

---

## Problem

Voice authentication systems, call centers, and real-time communication platforms face a growing threat from AI-generated synthetic speech. Tools that can generate convincing synthetic voices are widely available. Detecting these voices requires a dedicated classification model trained on diverse spoofed audio.

---

## Solution

VoxShield provides:

- A custom neural network (VoxShieldNet) trained entirely from random initialization on ASVspoof5
- A FastAPI backend exposing classification via HTTP
- A React/Vite SOC-style dashboard for live analysis
- Scientifically defensible evaluation with EER-based thresholds from validation data
- A deterministic risk engine that translates model probability into operational threat levels

---

## Architecture

```
Audio Input (file / microphone)
    │
    ▼
AudioPreprocessor
    16 kHz mono, 4-second window, float32
    │
    ├──► Raw Waveform Branch (1D CNN)
    │         Conv1d × 4 with residual connections
    │         Temporal features
    │
    └──► Log-Mel Spectrogram Branch (2D CNN)
              Conv2d × 4 with residual connections
              Spectral features
                │
                ▼
         Feature Fusion (concatenate)
                │
                ▼
         Bidirectional GRU (2 layers)
                │
                ▼
         Temporal Attention
                │
                ▼
         Linear Classifier → RAW LOGIT
                │
                ▼
         sigmoid(logit) = spoof_probability
                │
                ▼
         Decision: spoof_prob ≥ threshold → SPOOF
                │
                ▼
         Risk Engine (deterministic policy)
         → threat_level, recommended_action
```

---

## VoxShieldNet Architecture

| Component | Details |
|-----------|---------|
| Raw waveform branch | 4-layer 1D CNN, kernels 15/9/7/5, strides 4/2/2/2, GELU, BN, residuals |
| Spectral branch | 4-layer 2D CNN on log-mel spectrogram (80 mel bins, 512-point FFT), GELU, BN, residuals |
| Temporal modeling | 2-layer BiGRU, hidden=256, bidirectional |
| Attention | Learnable temporal attention over GRU output |
| Classifier | 3-layer MLP → 1 logit (no sigmoid inside model) |
| Parameters | ~3.8 million |
| Training objective | BCEWithLogitsLoss with class weighting |
| Threshold selection | Equal Error Rate on validation set only |

---

## Why It Is Different

- **No pretrained components.** No Wav2Vec, HuBERT, Whisper, WavLM, AASIST, or any speech encoder. VoxShieldNet is trained entirely from random initialization.
- **No external APIs.** All inference runs locally. No audio data leaves the machine.
- **Transparent.** The risk engine is a deterministic policy, not a black-box model. You can read exactly how model probability maps to threat level.
- **Scientifically defensible evaluation.** The decision threshold is selected from validation data only. The test set is never used to choose a threshold.

---

## ASVspoof5

VoxShield uses the [ASVspoof5](https://www.asvspoof.org/index2024.html) dataset for training and evaluation.

**Label convention:**
- `0` = bonafide (genuine human speech)
- `1` = spoof (synthetic / voice-converted speech)

Expected dataset structure:
```
<DATASET>/
    protocols/
        ASVspoof5.train.tsv
        ASVspoof5.dev.track_1.tsv
        ASVspoof5.eval.track_1.tsv
    train/
        flac_T/
            T_0000000000.flac
            ...
```

---

## Dataset Preparation

```bash
python -m training.prepare_asvspoof5 --dataset-dir "D:\Datasets\ASVspoof5"
```

This command:
1. Verifies all required protocol files exist
2. Scans audio directories
3. Builds and caches an index (no full audio decode)
4. Reports class counts, speaker counts, attack type distribution
5. Reports speaker overlap between train and validation

---

## Training

**Full ASVspoof5 training:**
```bash
python -m training.train \
    --asvspoof5-dir "D:\Datasets\ASVspoof5" \
    --epochs 30 \
    --checkpoint-dir checkpoints/asvspoof5
```

**Windows PowerShell:**
```powershell
.\scripts\train_asvspoof5.ps1 -DatasetDir "D:\Datasets\ASVspoof5"
```

**With GPU (CUDA):**
```bash
python -m training.train \
    --asvspoof5-dir "D:\Datasets\ASVspoof5" \
    --epochs 30 \
    --batch-size 64 \
    --num-workers 8 \
    --amp \
    --checkpoint-dir checkpoints/asvspoof5
```

**Resume training:**
```bash
python -m training.train \
    --asvspoof5-dir "D:\Datasets\ASVspoof5" \
    --checkpoint-dir checkpoints/asvspoof5 \
    --resume
```

**Checkpoint location:**
```
checkpoints/asvspoof5/best.pt   ← best validation AUC
checkpoints/asvspoof5/last.pt   ← last epoch
```

---

## Evaluation

```bash
python -m evaluation.evaluator \
    --checkpoint checkpoints/asvspoof5/best.pt \
    --asvspoof5-dir "D:\Datasets\ASVspoof5" \
    --split dev
```

**Windows:**
```powershell
.\scripts\evaluate.ps1 -DatasetDir "D:\Datasets\ASVspoof5"
```

**Evaluation outputs:**
- `reports/asvspoof5/eval_dev_<timestamp>/report.txt`
- `reports/asvspoof5/eval_dev_<timestamp>/summary.json`
- `reports/asvspoof5/eval_dev_<timestamp>/per_generator.json`
- `reports/asvspoof5/eval_dev_<timestamp>/robustness.json`

**Metrics reported:**
- ROC-AUC, EER (threshold-independent)
- Accuracy, Precision, Recall, F1, FPR, FNR (at validation threshold)
- Confusion matrix
- Per attack-type breakdown
- Robustness: clean / noisy (SNR 25 dB) / compressed (8-bit quantization)

---

## Backend

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

**Windows:**
```powershell
.\scripts\run_backend.ps1
```

API documentation: `http://localhost:8000/docs`

---

## Frontend

```bash
cd frontend && npm run dev
```

**Windows:**
```powershell
.\scripts\run_frontend.ps1
```

Dashboard: `http://localhost:3000`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service liveness and model status |
| GET | `/model-info` | Detailed model metadata and validation metrics |
| POST | `/predict` | Upload audio, receive classification |
| GET | `/incidents` | List logged incidents |
| GET | `/incidents/stats` | Aggregate statistics |
| GET | `/incidents/timeline` | Recent events for live dashboard |
| POST | `/reload-model` | Reload model after training |

**POST /predict response:**
```json
{
  "classification": "SPOOF",
  "spoof_probability": 0.87,
  "bona_fide_probability": 0.13,
  "confidence": 0.74,
  "decision_threshold": 0.42,
  "risk_score": 0.82,
  "threat_level": "HIGH",
  "recommended_action": "CHALLENGE — ...",
  "risk_breakdown": { "ml_model": false, "policy_engine": "deterministic" },
  "latency_ms": 45.2,
  "device": "cpu",
  "model_version": "1.0.0",
  "inference_mode": "asvspoof5"
}
```

---

## Threat and Risk Engine

The risk engine is a **deterministic policy layer**, not a second ML model.

```
spoof_probability → risk policy → threat_level
```

| Threshold | Threat Level |
|-----------|-------------|
| ≥ 0.90    | CRITICAL    |
| ≥ 0.70    | HIGH        |
| ≥ 0.45    | MEDIUM      |
| < 0.45    | LOW         |

Risk score = `spoof_probability ^ 0.7` (monotone mapping, emphasises high-end risk).

Recommended actions are derived from threat level and margin from the decision boundary.

---

## Microphone / Audio Workflow

The current system analyzes **local audio files and uploaded recordings**. Use the dashboard to upload `.flac`, `.wav`, `.ogg`, or `.mp3` files.

For live microphone capture, record from your system microphone, save as WAV/FLAC, and upload to the dashboard.

**What this system is:**
- Local audio file analysis
- Controlled audio stream analysis
- Microphone recording analysis

**What this system is NOT:**
- Real-time phone call interception
- WhatsApp/cellular network interception
- Live VoIP capture (architecture is prepared for future WebRTC integration)

---

## Checkpoint Override

To use a specific checkpoint:
```bash
export VOXSHIELD_CHECKPOINT=/path/to/model.pt
uvicorn backend.app:app --port 8000
```

On Windows:
```powershell
$env:VOXSHIELD_CHECKPOINT = "D:\models\best.pt"
.\scripts\run_backend.ps1
```

---

## Limitations

- VoxShieldNet has not been validated on live phone networks or compressed codecs beyond those present in ASVspoof5.
- Performance depends on training completion and dataset coverage. Until training finishes, the dashboard correctly shows MODEL OFFLINE.
- ROC-AUC and EER are meaningful only once real training completes. Untrained model scores are random.
- The system does not claim to detect all forms of synthetic speech outside ASVspoof5 attack types.

---

## Security and Privacy

- All inference is local. No audio data leaves the machine.
- No model weights are downloaded from external services.
- Uploaded audio is processed in memory and temporary files are deleted immediately after inference.
- The incident log is a local SQLite database. It contains no audio, only metadata.
- No API keys, secrets, or credentials are required.

---

## SIH Demo Flow

1. Start backend: `.\scripts\run_backend.ps1`
2. Start frontend: `.\scripts\run_frontend.ps1`
3. Open dashboard: `http://localhost:3000`
4. Upload a `.flac` or `.wav` audio file
5. Click **Analyze Audio**
6. Observe:
   - BONA FIDE / SYNTHETIC verdict
   - Probability bars
   - Threat level and recommended action
   - Incident logged in history table

If no trained model is present, the dashboard clearly shows **MODEL OFFLINE — TRAINING REQUIRED** and does not display fabricated predictions.

For full demo with trained model:
```powershell
.\scripts\demo.ps1 -DatasetDir "D:\Datasets\ASVspoof5"
```

---

## Quick Reference

```bash
# First-time setup
.\scripts\setup.ps1

# Prepare dataset
python -m training.prepare_asvspoof5 --dataset-dir "D:\Datasets\ASVspoof5"

# Train
.\scripts\train_asvspoof5.ps1 -DatasetDir "D:\Datasets\ASVspoof5" -Epochs 30

# Evaluate
.\scripts\evaluate.ps1 -DatasetDir "D:\Datasets\ASVspoof5"

# Run backend
.\scripts\run_backend.ps1

# Run frontend
.\scripts\run_frontend.ps1

# Full demo (opens browser)
.\scripts\demo.ps1

# Run tests
python -m pytest tests/ -v
```

---

## License

This software is provided for research and educational use. The ASVspoof5 dataset has its own license and terms of use — see the official ASVspoof5 website.
