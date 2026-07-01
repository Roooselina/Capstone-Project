# A Two-Stage Deep Learning System for Audio Deepfake Detection

Undergraduate thesis, Soonchunhyang University, Department of Computer Engineering, 2026.

This repository contains the implementation of a two-stage deepfake speech detection
system that combines frame-level multi-acoustic feature extraction with temporal
sequence classification.

---

## Overview

Stage 1 extracts six complementary acoustic features per frame and concatenates them
into a 106-dimensional vector. Stage 2 trains and compares three sequence classifiers
on the same input representation under identical conditions.

### Stage 1: Multi-Acoustic Feature Extraction

| Feature | Dim | Purpose |
|---|---|---|
| MFCC | 13 | Mel-scale cepstral coefficients, phoneme and timbre |
| MFCC Delta | 13 | First-order temporal derivative, co-articulation patterns |
| MFCC Delta-Delta | 13 | Second-order derivative, temporal inconsistency in synthesis |
| LFCC | 20 | Linear-scale filterbank, high-frequency vocoder artifacts |
| Mel-Spectrogram | 40 | Time-frequency energy distribution |
| Spectral Contrast | 7 | Sub-band peak-valley contrast, flatness of synthesized speech |
| Total | 106 | Per-frame feature vector |

All six features share the same STFT frame grid (window 25 ms, hop 10 ms),
so temporal alignment is guaranteed automatically.

### Stage 2: Sequence Classifiers

| Model | Key Mechanism | Params | Latency (ms/batch) |
|---|---|---|---|
| BiLSTM | Gated bidirectional RNN, hidden=256, 2 layers | 4.47 M | 47.97 |
| Transformer | Pre-LN, d_model=256, heads=8, N=4 | 2.19 M | 19.15 |
| Hybrid (BiLSTM + Transformer) | BiLSTM hidden=128 then Transformer N=2 | 1.78 M | 41.27 |

All three models share the same input projection, SpecAugment augmentation,
attention pooling, and MLP classifier head.

---

## Results

Test set: 15,891 samples (ASVspoof 2019 LA + ASVspoof 2021 + Kaggle-based data).

| Model | Accuracy | Macro F1 | AUC | EER (%) | ECE | Misclassified |
|---|---|---|---|---|---|---|
| BiLSTM | 0.9992 | 0.9990 | 1.0000 | 0.082 | 0.0008 | 13 / 15,891 |
| Hybrid | 0.9992 | 0.9990 | 0.9999 | 0.111 | 0.0008 | 13 / 15,891 |
| Transformer | 0.9985 | 0.9982 | 0.9997 | 0.145 | 0.0014 | 24 / 15,891 |

BiLSTM is the best model under this setting. McNemar's test confirms the difference
between BiLSTM and Transformer is statistically significant (p=0.013). BiLSTM and
Hybrid are statistically indistinguishable (p=1.000). 5-fold cross-validation on BiLSTM
yields mean Macro F1 0.986, std 0.006, confirming consistency across splits.

Note: these results are specific to this dataset configuration (mean sequence length
~318 frames). On longer sequences or different synthesis methods, Transformer or
Hybrid may be more suitable.

---

## Repository Structure

```
deepfake-2stage-detection/
├── notebooks/
│   ├── data_combine.ipynb                    # dataset preprocessing and split
│   ├── deepfake_2stage_detection_v2.ipynb    # v2: initial multi-feature + sequence model
│   ├── deepfake_2stage_detection_v4.ipynb    # v4: BiLSTM-centered, full comparison
│   ├── deepfake_2stage_detection_final.ipynb # final version submitted with thesis
│   └── deepfake_final.ipynb                  # cleaned final notebook
├── results/
│   ├── figures/                              # training curves, confusion matrices, ROC,
│   │                                         # reliability diagrams, feature distribution plots
│   └── experiment_summary_v4.json            # metrics and hyperparameters from v4 run
├── models/
│   └── seq_scaler_v3.pkl                     # Welford-computed scaler (channel-wise mean/std)
├── data/
│   └── .gitkeep                              # dataset files are excluded (see Dataset section)
├── config.example.py                         # copy to config.py and set DATA_ROOT
├── .gitignore
└── README.md
```

---

## Dataset

This study uses ASVspoof 2019 LA, ASVspoof 2021, and Kaggle-based public deepfake
speech data combined into a single dataset (158,892 samples total, 8:1:1 split).

Dataset files are not included in this repository.

```
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('awsaf49/asvpoof-2019-dataset')"
```

After downloading, set the path in config.py:

```python
# config.py (copy from config.example.py)
DATA_ROOT = "/path/to/your/dataset"
```

---

## Environment

| Item | Version |
|---|---|
| OS | Windows 11 |
| Python | 3.10 (Conda) |
| PyTorch | 2.5.1 + CUDA 12.1 |
| librosa | 0.11.0 |
| scikit-learn | 1.7.2 |
| numpy | 2.0.1 |
| scipy | 1.15.2 |
| GPU | NVIDIA RTX 4070 Ti (12 GB VRAM) |
| RAM | 15 GB |

Install dependencies:

```bash
pip install torch torchaudio librosa scikit-learn numpy scipy
```

---

## Training Details

- Loss: class-weighted cross-entropy (real:fake ratio ~1:2.3)
- Optimizer: AdamW, lr=1e-4, weight decay=1e-2
- Scheduler: Cosine Annealing Warm Restarts (T0=10, Tmult=2)
- Early stopping: patience=10, monitored on validation Macro F1
- Mixed precision: PyTorch AMP with GradScaler
- Gradient clipping: max_norm=1.0
- Augmentation: SpecAugment (time_mask=30, freq_mask=10, training only)
- Pooling: attention pooling with length masking for variable-length sequences

Memory pipeline for 15 GB RAM / ~31 GB training data:
- Chunk-based checkpointing: features saved to disk in chunks, loaded via streaming
- Welford online algorithm: single-pass mean and variance estimation without full load

---

## Thesis

Lee Eunjung, "A Two-Stage Deep Learning System for Audio Deepfake Detection,"
B.Eng. thesis, Soonchunhyang University, 2026.
