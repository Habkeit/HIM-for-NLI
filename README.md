# HIM / Enhanced LSTM NLI Project

This repository contains an enhanced LSTM-based Natural Language Inference (NLI) implementation adapted for the HIM architecture, with training and testing scripts in `enhancedLSTM-main`.

## Prerequisites

- Python 3.8+
- PyTorch installed in a conda or virtual environment
- Required Python packages available in the `data_env` environment used for this project

## Clone and setup

```bash
git clone <repo-url> "HIM"
cd "HIM\enhancedLSTM-main"
```

If using conda, activate the environment used for this project:

```powershell
C:\Users\Admin\anaconda3\shell\condabin\conda-hook.ps1
conda activate data_env
```

## Run training

From `enhancedLSTM-main`:

```powershell
python train.py --config snli_training.json
```

This will load preprocessed SNLI data and start training the model.

## Run testing

From `enhancedLSTM-main`:

```powershell
python test.py
```

## Notes

- The main code and model files are under `enhancedLSTM-main/`.
- Data files such as `train_data.pkl`, `valid_data.pkl`, and `test_data.pkl` are used by the training/testing scripts.
- If you need to modify model settings, update `snli_training.json` and rerun training.
