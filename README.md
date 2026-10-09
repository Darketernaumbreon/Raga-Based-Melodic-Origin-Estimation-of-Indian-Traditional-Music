# Hindustani Raag Classification — Multi-Dataset Research Repository

A clean, reproducible **PyTorch research codebase** for classifying Hindustani Classical Music raags across multiple curated datasets (**Saraga 1.5**, **Indian Music Raag Dataset (IMRD)**, and **Thaat and Raga Forest (TRF)**).

---

## 📌 Repository Architecture

```text
Raag_Analysis/
├── README.md                           ← Project documentation & learning roadmap
├── requirements.txt                    ← Python dependencies
├── train.py                            ← Main experiment execution script
│
├── data/
│   ├── raw/                            ← Read-only raw audio and metadata corpora
│   │   ├── saraga/
│   │   ├── indian_music_raag/
│   │   └── Thaat and Raga Forest (TRF) Dataset/
│   │
│   └── processed/
│       ├── metadata/                   ← Curated training metadata and splits
│       │   ├── unified/
│       │   │   ├── final_metadata.csv  ← Master canonical recording inventory (1,142 files)
│       │   │   └── recommended_training_classes.csv
│       │   ├── dataset_A/              ← Dataset A — Baseline (38 classes, ≥15 recordings)
│       │   │   ├── metadata_A.csv
│       │   │   └── class_summary_A.csv
│       │   └── dataset_B/              ← Dataset B — Extended (60 classes, ≥10 recordings)
│       │       ├── metadata_B.csv
│       │       └── class_summary_B.csv
│       │
│       └── reports/                    ← Permanent research audit logs & validation reports
│           ├── dataset_build_report.md
│           ├── dataset_comparison.csv
│           ├── harmonization_report.csv
│           └── audio_validation_report.csv
│
├── src/                                ← PyTorch research package
│   ├── __init__.py
│   ├── preprocessing/                  ← [WORKING] Audio loading, resampling & feature extraction
│   │   ├── audio_utils.py
│   │   └── feature_extraction.py
│   ├── datasets/                       ← [WORKING] PyTorch Dataset & stratified splitting
│   │   ├── build_datasets.py
│   │   ├── raag_dataset.py
│   │   └── split_utils.py
│   ├── models/                         ← [TODO / LEARNING ZONE] Neural network architectures
│   │   ├── baseline_cnn.py             ← Scaffolded 2D CNN (ConvBlock, RaagClassifierCNN)
│   │   └── crnn.py                     ← Scaffolded Convolutional Recurrent Network (CRNN)
│   ├── training/                       ← [TODO / LEARNING ZONE] Training loops & optimization
│   │   └── trainer.py                  ← Scaffolded Trainer class
│   ├── evaluation/                     ← [TODO / LEARNING ZONE] Metrics & evaluation passes
│   │   └── metrics.py                  ← Scaffolded Top-K accuracy & classification report
│   └── utils/                          ← [WORKING] YAML configs & logger utilities
│       ├── config.py
│       └── logger.py
│
├── configs/                            ← YAML training hyperparameters
│   ├── baseline_A.yaml
│   └── baseline_B.yaml
│
├── checkpoints/                        ← Saved model weights (.pt files)
├── experiments/                        ← Experiment run outputs & tensorboard logs
├── notebooks/                          ← Exploratory data analysis & model debugging
└── logs/                               ← Execution logs
```

---

## 📊 Training Datasets Overview

Two distinct training datasets are pre-built from [`final_metadata.csv`](file:///C:/Users/Ayan/Desktop/Publication/Raag_Analysis/data/processed/metadata/unified/final_metadata.csv) at different recording frequency thresholds:

| Dataset | Threshold | Classes | Recordings | Total Duration | Use Case |
|---|---|---|---|---|---|
| **Dataset A** | ≥ 15 per class | **38** | **772** | **168.4 h** | Primary research baseline |
| **Dataset B** | ≥ 10 per class | **60** | **1,052** | **252.1 h** | Extended generalization experiment |

*Note: Dataset A is a strict subset of Dataset B. Stratified train/val/test splitting (`src/datasets/split_utils.py`) is performed exclusively at the **recording level** (`recording_id`), guaranteeing zero data leakage when audio is sliced into shorter time segments.*

---

## 🚀 Learning Roadmap — Building Your Model From Scratch

Since you are writing the model code to learn deep learning for audio from scratch, the data loading and preprocessing infrastructure is fully ready to go (`src/preprocessing/` and `src/datasets/`). 

Your tasks are located inside the **Learning Zones (`src/models/`, `src/training/`, `src/evaluation/`)**:

### Step 1: Design Your Convolutional Layers (`src/models/baseline_cnn.py`)
1. Open [`src/models/baseline_cnn.py`](file:///C:/Users/Ayan/Desktop/Publication/Raag_Analysis/src/models/baseline_cnn.py).
2. Implement `ConvBlock`: combine `nn.Conv2d` (kernel size 3, padding 1), `nn.BatchNorm2d`, `nn.ReLU`, and `nn.MaxPool2d`.
3. Implement `RaagClassifierCNN`: stack multiple `ConvBlock`s, apply `nn.AdaptiveAvgPool2d` to handle variable-duration audio segments, and add a final `nn.Linear` classification head.

### Step 2: Implement the Training Cycle (`src/training/trainer.py`)
1. Open [`src/training/trainer.py`](file:///C:/Users/Ayan/Desktop/Publication/Raag_Analysis/src/training/trainer.py).
2. Fill out `train_epoch()`:
   - Put model in `.train()` mode.
   - Loop over `(inputs, targets)` from `train_loader`.
   - Run forward pass, compute `loss = self.criterion(outputs, targets)`.
   - Run backward pass (`loss.backward()`) and step the optimizer (`self.optimizer.step()`).
3. Fill out `validate()`:
   - Put model in `.eval()` mode (`with torch.no_grad():`).
   - Loop over `val_loader` and compute validation loss and accuracy without backpropagation.

### Step 3: Implement Evaluation & Metrics (`src/evaluation/metrics.py`)
1. Open [`src/evaluation/metrics.py`](file:///C:/Users/Ayan/Desktop/Publication/Raag_Analysis/src/evaluation/metrics.py).
2. Implement `compute_topk_accuracy()` using `torch.topk` to measure Top-1, Top-3, and Top-5 accuracy across the raags.
3. Implement `evaluate_model()` to run the test set evaluation loop and generate predictions for `scikit-learn`'s `classification_report`.

### Step 4: Launch Your First Training Run! (`train.py`)
1. Open [`train.py`](file:///C:/Users/Ayan/Desktop/Publication/Raag_Analysis/train.py) and uncomment the block in the **LEARNING ZONE**.
2. Run your baseline experiment from terminal:
   ```bash
   python train.py --config configs/baseline_A.yaml
   ```
3. To switch to the 60-class extended dataset:
   ```bash
   python train.py --config configs/baseline_B.yaml
   ```

---

## 🛠️ Rebuilding / Modifying Datasets (Optional)

If you ever wish to adjust class criteria or regenerate `metadata_A.csv` / `metadata_B.csv`, run:
```bash
python -m src.datasets.build_datasets
```
This reads directly from `data/processed/metadata/unified/final_metadata.csv` and updates both dataset CSVs alongside the comparative analysis in `data/processed/reports/dataset_build_report.md`.
