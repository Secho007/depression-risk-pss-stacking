# Depression Risk Prediction via Probability-Strength Stacking (PSS)

Code repository for the paper:

> **Development and Validation of a Probability-Strength Stacking Ensemble Model for Depression Risk Prediction in College Students: A SHAP-Based Interpretability Analysis**
> Jiuqi Sun, Chenchen Xu, Guohua Pan, Ming Lu — Nanjing Medical University
> *Journal of Affective Disorders* (under review)

---

## Overview

This repository contains the full analysis pipeline for developing and validating a Probability-Strength Stacking (PSS) ensemble framework for college student depression risk prediction.

Unlike conventional stacking, PSS passes continuous predicted probabilities from base learners — rather than discrete hard labels — to the meta-learner, preserving inter-model confidence information and improving probability calibration.

**Key results (independent test set, n = 5,581):**

| Metric | Value |
|--------|-------|
| AUC (95% CI) | 0.9193 (0.9125–0.9266) |
| Accuracy | 0.8461 |
| Sensitivity | 0.8843 |
| Specificity | 0.7920 |
| F1 score | 0.8706 |
| Brier score | 0.1112 |
| AUC vs hard-label stacking | +5.88 percentage points |
| Brier score vs hard-label stacking | −11.1% |

---

## Data

The dataset used in this study is publicly available on the Kaggle platform:

**[Student Mental Health Dataset](https://www.kaggle.com/datasets/ikynahidwin/depression-student-dataset)**

The original dataset contains 140,700 records. Working professionals (n = 112,799) were excluded; the final analytic sample comprised **27,901 college students** with 19 behavioral, psychological, and demographic features. The dataset is fully anonymized with no personally identifiable information.

**Setup:** Download the dataset from Kaggle, save it as `student.xlsx` in the project root directory, or update the `XLSX` path variable at the top of each script to match your local path.

---

## Repository Structure

```
depression-risk-pss-stacking/
│
├── jmir_fix.py               # Main modeling pipeline:
│                             #   data loading, preprocessing, model training,
│                             #   performance evaluation (Table 2), Table 1
│                             #   baseline statistics with effect sizes,
│                             #   5-fold CV stability, ablation experiment
│
├── fix_figures_final.py      # Figure generation pipeline:
│                             #   Fig. 1 (flowchart placeholder),
│                             #   Fig. 2 (ROC curves),
│                             #   Fig. 3 (ablation study),
│                             #   Fig. 4 (SHAP beeswarm),
│                             #   Fig. 5 (SHAP dependence plots, top-6),
│                             #   Fig. 6 (DCA),
│                             #   Suppl. Fig. 1 (SHAP individual cases),
│                             #   Suppl. Fig. 2 (violin plots),
│                             #   Suppl. Fig. 3 (Pearson heatmap),
│                             #   Suppl. Fig. 4 (CV stability)
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Installation

Python 3.9 or above is recommended.

```bash
# Clone the repository
git clone https://github.com/Secho007/depression-risk-pss-stacking.git
cd depression-risk-pss-stacking

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Run the two scripts in order:

### Step 1 — Main pipeline

```bash
python jmir_fix.py
```

Runs data loading, preprocessing, model training and evaluation, Table 1 baseline statistics, Table 2 performance comparison, 5-fold cross-validation, and the ablation experiment.

Outputs saved to: `jmir_fix_outputs/`

Estimated runtime: 15–25 minutes depending on hardware.

### Step 2 — Figure generation

```bash
python fix_figures_final.py
```

Generates all manuscript figures (Fig. 2–6) and supplementary figures (Suppl. Fig. 1–4) in PNG, PDF, and EPS formats.

Outputs saved to: `routeA_fix_outputs/`

**Note:** Step 2 requires the trained models from Step 1. Run Step 1 first.

---

## Models

| Model | Type |
|-------|------|
| XGBoost | Single base learner |
| LightGBM | Single base learner |
| Random Forest | Single base learner |
| SVM | Comparator (not included in ensemble) |
| CatBoost | Single base learner |
| Stacking (Label-OOF) | Hard-label stacking (ablation baseline) |
| **PSS Stacking-v2 ★** | **Probability-Strength Stacking (proposed)** |

PSS Stacking-v2 uses CatBoost, XGBoost, LightGBM, and Random Forest as base learners, with logistic regression as the meta-learner trained on out-of-fold (OOF) continuous probability outputs.

---

## Output Files

### Main figures

| File | Figure |
|------|--------|
| `fig1_roc_all_models_stackingv2.*` | Fig. 2 — ROC curves, all models |
| `fig2_ablation_updated.*` | Fig. 3 — Ablation study (PSS vs hard-label stacking) |
| `fig3_shap_beeswarm_stackingv2.*` | Fig. 4 — SHAP beeswarm plot |
| `fig4_shap_dependence_top6.*` | Fig. 5 — SHAP dependence plots, top-6 predictors |
| `fig5_dca_stackingv2.*` | Fig. 6 — Decision curve analysis |

### Supplementary figures

| File | Figure |
|------|--------|
| `supplfig1_shap_panel_3cases_stackingv2.*` | Suppl. Fig. 1 — SHAP individual case analysis |
| `supplfig2_violin_continuous.*` | Suppl. Fig. 2 — Feature distribution (violin plots) |
| `supplfig3_pearson_heatmap.*` | Suppl. Fig. 3 — Pearson correlation heatmap |
| `supplfig4_cv_stability_stackingv2.*` | Suppl. Fig. 4 — 5-fold CV stability |

### Statistics files

| File | Content |
|------|---------|
| `routeA_final_statistics.txt` | Final model statistics (AUC, CI, Brier, CV, DCA range) |
| `Table1_baseline_with_effect_size.csv` | Table 1 data |
| `model_comparison_updated.csv` | Table 2 data |

---

## Reproducibility

All stochastic elements are fixed for reproducibility:

- `random_state = 42` throughout (data splitting, model training, bootstrap)
- Stratified 80:20 train/test split (`StratifiedShuffleSplit`)
- 5-fold cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Bootstrap confidence intervals: 1,000 iterations, `np.random.RandomState(42)`
- Feature scaling: `RobustScaler` fitted on training set only, applied to test set

---

## Citation

If you use this code in your research, please cite:

```
Sun J, Xu C, Pan G, Lu M. Development and Validation of a Probability-Strength
Stacking Ensemble Model for Depression Risk Prediction in College Students:
A SHAP-Based Interpretability Analysis. Journal of Affective Disorders. 2026
(under review).
```

---

## License

This code is released for academic research purposes only, consistent with the terms of the source dataset license on Kaggle.

---

## Contact

For questions regarding the code or analysis, please contact:

**Ming Lu** (Corresponding author)
Nanjing Medical University, Nanjing, China
E-mail: mlu@njmu.edu.cn

**Guohua Pan** (Corresponding author)
Nanjing Medical University, Nanjing, China
E-mail: Pan518@njmu.edu.cn
