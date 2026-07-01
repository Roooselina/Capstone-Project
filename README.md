# Deepfake Speech Detection - 2-Stage System

2-Stage deepfake speech detection system using multi-feature extraction and temporal sequence modeling.

## Structure

```
deepfake-2stage-detection/
├── notebooks/
│   ├── data_combine.ipynb                    # Dataset preprocessing and combination
│   ├── deepfake_2stage_detection_v2.ipynb    # 2-stage system v2
│   ├── deepfake_2stage_detection_v4.ipynb    # 2-stage system v4 (BiLSTM-centered)
│   ├── deepfake_2stage_detection_final.ipynb # Final version
│   └── deepfake_final.ipynb                  # Final cleaned version
├── results/
│   ├── figures/                              # Training curves, confusion matrices, ROC curves
│   └── experiment_summary_v4.json            # Experiment metadata and metrics
└── models/
    └── seq_scaler_v3.pkl                     # StandardScaler fitted on sequence features
```

## Overview

- Stage 1: Multi-feature extraction (MFCC, LFCC, Mel-Spectrogram, Spectral Contrast, Delta features)
- Stage 2: Temporal sequence classifier (Transformer / BiLSTM / Hybrid)
- Dataset: ASVspoof 2019

## Requirements

```
torch
torchaudio
librosa
scikit-learn
numpy
pandas
```
