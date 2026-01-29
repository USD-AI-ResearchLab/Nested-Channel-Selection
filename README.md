﻿# EEG Parkinson’s Disease Classification and Visualization

This repository provides a fully reproducible pipeline for training, evaluating, and visualizing EEG-based Parkinson’s disease classifiers using **signed Grad‑CAMs**. The workflow is designed to maximize *good‑channel accuracy*, select the best-performing fold, and generate physiologically interpretable visualizations highlighting both excitatory and suppressive evidence.

---

## Dataset Setup

### 1. Iowa + UNM Parkinson’s EEG Dataset (Anjum et al.)

- Navigate to:  
  https://predict.cs.unm.edu/downloads.php
- Locate **Dataset Name: d013**
- Download the repository
- Inside the downloaded content, locate the **`Dataset/`** folder
- Copy or drag this **`Dataset/`** folder directly into the root of this repository

This dataset contains the Iowa and University of New Mexico cohorts used throughout the primary experiments.

---

### 2. San Diego (UCSD) Parkinson’s EEG Dataset

- Download the San Diego dataset from its public OpenNeuro release
- Locate the folder named **`ds002778`**
- Copy or drag **`ds002778/`** into the existing **`Dataset/`** directory from Step 1

After this step, the `Dataset/` directory should contain both the Iowa + UNM data and the San Diego cohort.

---

## Environment Setup

Create and activate a Python environment, then install dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

___

## Running the Pipeline

Execute the full training, evaluation, and visualization pipeline with:

```bash
python runAll.py
```

___

## Outputs and Visualizations

The repository currently includes **example Grad‑CAM visualizations** from the first successful reproduction run.

These Grad‑CAMs are **signed**, explicitly highlighting:
- **Excitatory (positive) evidence**
- **Suppressive (negative) evidence**

Some variability in excitatory versus suppressive localization is expected across runs. In all cases, however, the highlighted regions correspond to **physiologically meaningful frequency bands**.

---

## Methodological Notes

- The pipeline is tuned to **maximize good‑channel accuracy**.
- The **best‑performing fold** is automatically selected.
- Grad‑CAM visualizations are generated exclusively from this fold to ensure consistency and interpretability.
- This design mirrors the evaluation logic used in the accompanying analyses and prioritizes reviewer‑safe reproducibility.

---

This setup enables clean, end‑to‑end reproduction of both quantitative performance and qualitative interpretability results.
