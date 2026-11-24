# Cough Sound Analysis for Respiratory Health Screening (COUGHVID)

An end-to-end audio machine learning pipeline that tests whether cough recordings can help distinguish **healthy** from **unwell** (symptomatic or COVID-19) participants. The project compares classical ML, a CNN and pretrained audio embeddings, and evaluates them with clinical-style metrics and explainability.

> **Research prototype, not a medical device.** It must not be used for diagnosis or clinical decisions.

---

## Table of contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Pipeline](#pipeline)
4. [Models](#models)
5. [Evaluation protocol](#evaluation-protocol)
6. [Results](#results)
7. [Sanity check](#sanity-check)
8. [Limitations](#limitations)
9. [Ethics and responsible use](#ethics-and-responsible-use)
10. [Future work](#future-work)
11. [Citation and license](#citation-and-license)
12. [Author](#author)

---

## Overview

Cough audio is a promising, low-cost signal for respiratory screening, but public datasets are crowdsourced, noisy and self-labeled. This project asks a careful question: **how much health-related signal can standard audio ML models extract from such data, and how reliably can that be measured?**

What the project covers:

- Audio preprocessing (decoding, resampling, silence trimming, normalization)
- Feature engineering (MFCCs, deltas, spectral statistics) and log-mel spectrograms
- Three model families: SVM / Random Forest, a log-mel CNN, and pretrained AST embeddings with logistic regression
- Clinical-style evaluation: AUC with bootstrap 95% confidence intervals, sensitivity, specificity, PR-AUC, with the decision threshold chosen on validation data only
- Explainability with Grad-CAM on the CNN
- Error analysis, a pipeline sanity check and an honest discussion of limitations

---

## Dataset

**COUGHVID**: a crowdsourced cough dataset collected through a web app.

| | |
|---|---|
| **Download (version used)** | https://zenodo.org/records/4048312 (`public_dataset.zip`) |
| **Newer version (v3, ~2.3 GB)** | https://zenodo.org/records/7024894 (`public_dataset_v3.zip`) |
| **Paper** | https://doi.org/10.1038/s41597-021-00937-4 |
| **License** | CC BY 4.0 (attribution required) |
| **Size** | 25,000+ recordings; 2,800+ with expert physician annotations |

**What the dataset contains**

- Audio recordings (`.webm` / `.ogg`), one JSON metadata file per recording
- `metadata_compiled.csv` with one row per recording
- Key columns used here:
  - `uuid`: recording ID
  - `cough_detected`: probability (0 to 1) that the recording contains a cough
  - `status`: self-reported `healthy`, `symptomatic` or `COVID-19`
  - `age`, `gender`
  - Expert annotation columns (diagnosis, quality, cough type, wheezing and others) for a subset

**Labels used in this project**

| Task | Source | Classes |
|---|---|---|
| `status` (default) | Self-reported `status` column | 0 = healthy, 1 = unwell (symptomatic or COVID-19) |
| `expert` | Majority vote of physician diagnoses | 0 = healthy cough, 1 = any condition |

Only recordings with `cough_detected >= 0.8` are kept to exclude non-cough audio.

The data is **not included** in this repository. Download it from the link above and unzip it locally or in Colab.

---

## Pipeline

```
Raw audio (.webm/.ogg)
   -> ffmpeg: 16 kHz mono wav
   -> trim silence, peak-normalize, pad/crop to 5 s
   -> features:  MFCC + deltas + spectral stats (132-dim)   |   log-mel spectrogram (64 x 313)
   -> models:    SVM, Random Forest, CNN, AST embeddings + logistic regression
   -> evaluation on held-out test split (threshold picked on validation)
   -> error analysis + Grad-CAM
```

Splits are stratified by label: 70% train, 15% validation, 15% test (seed 42).

---

## Models

| Model | Input | Notes |
|---|---|---|
| **SVM (RBF)** | 132-dim handcrafted features | Grid search over C and gamma (5-fold CV on train only), sigmoid-calibrated probabilities, balanced class weights |
| **Random Forest** | 132-dim handcrafted features | 400 trees, balanced subsample class weights |
| **CNN** | 64 x 313 log-mel spectrogram | 4 conv blocks, SpecAugment and time-shift augmentation, weighted BCE loss, AdamW, early stopping on validation AUC |
| **AST + Logistic Regression** | Pretrained Audio Spectrogram Transformer embeddings (AudioSet) | Frozen encoder, linear classifier trained on top |

---

## Evaluation protocol

- **Metrics:** ROC-AUC (with 1,000-sample bootstrap 95% CI), PR-AUC, sensitivity (recall), specificity, precision, F1, balanced accuracy
- **Threshold:** chosen by Youden's J on the **validation** set only, never on test
- **Why not accuracy:** classes are imbalanced, and in a screening setting missed cases and false alarms have different costs
- **Explainability:** Grad-CAM heatmaps show which time-frequency regions drive the CNN's predictions

---

## Results

> Fill this table from `results/*/*_metrics.json` after the full-data run.

**Full dataset (`status` task, `cough_detected >= 0.8`)**

| Model | AUC (95% CI) | Sensitivity | Specificity | F1 | PR-AUC |
|---|---|---|---|---|---|
| SVM | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| CNN | TBD | TBD | TBD | TBD | TBD |
| AST + LogReg | TBD | TBD | TBD | TBD | TBD |

Dataset size after filtering: `TBD` recordings (healthy: `TBD`, unwell: `TBD`). Test set: `TBD` recordings.

**Observations from development**

- On a small balanced subset (300 per class, 90 test clips), all models performed near chance (for example AST + logistic regression reached AUC 0.542), and the CNN collapsed to a near-constant output. That subset was too small to train on.
- On the full dataset the CNN began producing a wide spread of probabilities. Report the final full-data numbers above.
- Error analysis showed the CNN and the classical models often disagreed on false alarms. See the analysis notebook cells for recording-quality checks (SNR, duration) and demographic breakdowns.

---

## Sanity check

To confirm the pipeline itself works, the same handcrafted features were used to predict **gender** from cough audio, a property that is clearly audible:

| Check | Result |
|---|---|
| Gender classification AUC (subset, n = 90 test clips) | **0.775** |

A well above chance result on an audible attribute shows that audio loading, conversion, feature extraction and splitting are working, so weak health-classification performance reflects the difficulty of the task and label noise, not a pipeline bug.

---

## Limitations

- **No participant ID.** COUGHVID does not identify speakers, so the same person may appear in both train and test, which can inflate results.
- **Self-reported labels.** `status` is noisy. Physician labels are more reliable but cover only about 2,800 recordings.
- **Crowdsourced recordings.** Phone microphones, background noise and recording conditions vary widely, and models can learn recording conditions instead of illness.
- **Possible shortcuts.** Gender, age and recording quality can correlate with labels. Check these before trusting any result (see the shortcut checks in the notebook).
- **Class imbalance.** Healthy recordings far outnumber unwell ones.
- **No external validation.** Models are not yet tested on another dataset (for example Coswara or ICBHI).
- **Not clinically validated.** No claim is made about real-world screening performance.

---

## Ethics and responsible use

- Uses only public, de-identified, openly licensed data, with attribution to the dataset authors.
- Predictions are probabilistic and should never replace a clinician.
- Healthcare AI must be validated across devices, demographics and environments before any deployment; this project has not done that.
- Privacy: any audio recorded from users should be processed with consent and stored securely.

---

### Requirements

- Python 3.9+
- `ffmpeg` (needed to decode `.webm` / `.ogg`)
- A GPU is recommended for the CNN (a free Colab T4 is enough)

```bash
pip install -r requirements.txt
# ffmpeg: sudo apt install ffmpeg | brew install ffmpeg | winget install ffmpeg
```

### Run with scripts

```bash
# 1) Download and unzip the dataset from https://zenodo.org/records/4048312
python prepare_data.py --data_dir /path/to/public_dataset --out_dir data/processed

# Quick test on a small subset first
python prepare_data.py --data_dir /path/to/public_dataset --out_dir data/debug --max_per_class 300

# 2) Classical baselines
python train_baseline.py --manifest data/processed/manifest.csv --out_dir results/baseline

# 3) CNN + Grad-CAM
python train_cnn.py --manifest data/processed/manifest.csv --out_dir results/cnn
```

Use `--task expert` in `prepare_data.py` to train on physician-labeled recordings instead of self-reported status.

### Run in Google Colab

1. Open the notebook in `notebook/` and set the runtime to **GPU**.
2. Run the install cell (ffmpeg, librosa) and download the dataset.
3. Set `MAX_PER_CLASS = 300` for a quick test, or `None` for the full dataset.
4. Run the data preparation, baseline, CNN, visualization and error-analysis cells in order.

### Reproducibility

All splits and models use `seed = 42`. Thresholds are chosen on the validation split only.

---

## Future work

- Group-aware splitting or external validation on Coswara and ICBHI
- Controlling for gender, age and recording quality to rule out shortcuts
- Fine-tuning pretrained audio models (AST, Wav2Vec2, BEATs) instead of frozen embeddings
- Predicting acoustically grounded labels (wet vs dry cough, audible wheezing) from the expert annotations
- Probability calibration analysis and decision-curve evaluation
- A small demo app (FastAPI or Streamlit) with a clear "not a medical test" disclaimer, containerized with Docker

---

## Citation and license

**Dataset citation (required by CC BY 4.0):**

> Orlandic, L., Teijeiro, T., Atienza, D. The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. *Scientific Data* 8, 156 (2021). https://doi.org/10.1038/s41597-021-00937-4

**Code license:** add a license of your choice (for example MIT) to this repository. The dataset remains under its own CC BY 4.0 license and is not redistributed here.

---

## Author

**Seif Ahmed**, AI & Data Science Engineer
GitHub: [github.com/seifahmed2004](https://github.com/seifahmed2004) | LinkedIn: [linkedin.com/in/seif-ahmedd](https://www.linkedin.com/in/seif-ahmedd/)
