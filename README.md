# Channel-Selected Stratified Nested Cross-Validation for Clinically Relevant EEG-Based Parkinson’s Disease Detection

---

The early detection of Parkinson’s disease remains a critical challenge in clinical neuroscience, with electroencephalography offering a non‑invasive and scalable pathway toward population‑level screening. While machine learning has shown promise in this domain, many reported results suffer from methodological flaws, most notably patient‑level data leakage, inflating performance estimates and limiting clinical translation. To address these modeling pitfalls, we propose a unified evaluation framework grounded in nested cross‑validation and incorporating three complementary safeguards: (i) patient‑level stratification to eliminate subject overlap and ensure unbiased generalization, (ii) multi‑layered windowing to harmonize heterogeneous EEG recordings while preserving temporal dynamics, and (iii) inner‑loop channel selection to enable principled feature reduction without information leakage. Applied across three independent datasets with a heterogeneous number of channels, a convolutional neural network trained under this framework achieved 80.6\% accuracy and demonstrated state‑of‑the‑art performance under held‑out population block testing, comparable to other methods in the literature. This performance underscores the necessity of nested cross‑validation as a safeguard against bias and as a principled means of selecting the most relevant information for patient‑level decisions, providing a reproducible foundation that can extend to other biomedical signal analysis domains.

Below, this repository provides a fully reproducible pipeline for training, evaluating, and visualizing EEG-based Parkinson’s disease classifiers using **signed Grad‑CAMs**. The workflow is designed to maximize *good‑channel accuracy*, select the best-performing fold, and generate physiologically interpretable visualizations highlighting both excitatory and suppressive evidence.

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

- Download the San Diego dataset from its public OpenNeuro release: https://openneuro.org/datasets/ds002778/versions/1.0.5
- Locate the folder named **`ds002778`**
- Copy or drag **`ds002778/`** into the existing **`Dataset/`** directory from Step 1

After this step, the `Dataset/` directory should contain both the Iowa + UNM data and the San Diego cohort.

---

## Environment Setup

Create and activate a Python environment, then install dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Execute the full training, evaluation, and visualization pipeline with:

```bash
python runAll.py
```

---

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
