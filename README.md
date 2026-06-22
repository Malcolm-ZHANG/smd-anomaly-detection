# SMD Anomaly Detection

This repository contains my notebook for the SMD anomaly detection project.

I compare two unsupervised methods on `machine-1-1`:

- LSTM autoencoder
- TiReX zero-shot forecasting

The main file is:

```text
notebooks/SMD_anomaly_detection_complete.ipynb
```

## Data

The data used by the notebook is included in:

```text
data/SMD/train/machine-1-1.txt
data/SMD/test/machine-1-1.txt
data/SMD/test_label/machine-1-1.txt
```

The training and test files each have 28,479 timestamps and 38 channels.

## Environment

One possible setup is:

```bash
conda create -n smd-anomaly python=3.11
conda activate smd-anomaly
pip install numpy pandas matplotlib scikit-learn torch jupyter tirex-ts==1.4.2
```

Then open the notebook and run all cells.

## Results

The saved metrics and plots are in:

```text
results/notebook_complete/
```

Current results:

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LSTM Autoencoder | 0.2380 | 0.9762 | 0.3827 | 0.5975 | 0.9170 |
| TiReX zero-shot | 0.3373 | 0.3352 | 0.3363 | 0.2898 | 0.7287 |

The notebook has saved outputs already. Rerunning it is possible, but the TiReX part can take a long time on CPU.
