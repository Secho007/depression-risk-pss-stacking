# Depression Risk Prediction via Probability-Strength Stacking (PSS)

Code repository for the paper:

> **Development and Validation of a Probability-Strength Stacking Ensemble Learning Model for Depression Risk Prediction in College Students: SHAP-Based Interpretability Analysis**
> Jiuqi Sun, Chenchen Xu — Nanjing Medical University
> *Journal of Affective Disorders* (under review)

---

## Overview

This repository contains the full analysis pipeline for developing and validating a novel **Probability-Strength Stacking (PSS)** ensemble learning framework for college student depression risk prediction. The PSS framework retains continuous probability outputs from base learners as meta-features, improving calibration precision and discriminative performance over conventional hard-label stacking.

**Key results (independent test set, n=5,581):**
- AUC = 0.9193 (95% CI: 0.9125–0.9266)
- Brier Score = 0.1112
- Sensitivity = 0.8843 | Specificity = 0.7920
- AUC improvement over hard-label stacking: +5.88 percentage points
- Brier score improvement: −11.1%

---

## Data

The dataset used in this study is publicly available on the Kaggle platform:

> [Student Depression Dataset](https://www.kaggle.com/code/annastasy/mental-health-eda-ensemble/notebook)

The dataset contains 27,898 valid records of college students aged 18–35 years, encompassing 19 behavioral, psychological, and demographic features. It is fully anonymized with no personally identifiable information.

**Note:** Download the dataset from Kaggle and place the file at:
```
D:/Medical_AI_Project/筛student.xlsx
```
Or update the file path variable `XLSX` at the top of each script to match your local path.

---

## Repository Structure

```
depression-risk-pss-stacking/
│
├── depression_pipeline_v3.py     # Main pipeline: baseline statistics, ROC,
│                                 # calibration, DCA, SHAP (original version)
│
├── jmir_fix.py                   # Table 1 (with effect sizes), 5-fold CV
│                                 # stability, ablation experiment
│
├── fix_figures_routeA.py         # Final Route A figures: Fig1 ROC,
│                                 # Fig3 SHAP beeswarm, Fig5 DCA,
│                                 # Suppl.Fig4 CV stability (all using
│                                 # Stacking-v2 PSS, 4 base learners)
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Installation

**Python version:** 3.9 or above recommended.

```bash
# Clone the repository
git clone https://github.com/[your-username]/depression-risk-pss-stacking.git
cd depression-risk-pss-stacking

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Run the scripts in the following order:

### Step 1 — Main pipeline (baseline statistics, ROC, DCA, SHAP)
```bash
python depression_pipeline_v3.py
```
Outputs saved to: `pipeline_v3_outputs/`

### Step 2 — Table 1, ablation experiment, CV stability
```bash
python jmir_fix.py
```
Outputs saved to: `jmir_fix_outputs/`

### Step 3 — Final Route A figures (recommended for reproduction)
```bash
python fix_figures_routeA.py
```
Outputs saved to: `routeA_fix_outputs/`

Each script generates figures in PNG, PDF, and EPS formats, along with a summary statistics text file.

**Estimated runtime:** Step 3 takes approximately 15–25 minutes depending on hardware (mainly Stacking-v2 training + 5-fold CV).

---

## Models

The following models are implemented and compared:

| Model | Type |
|---|---|
| XGBoost | Single base learner |
| LightGBM | Single base learner |
| Random Forest | Single base learner |
| SVM | Single base learner (excluded from ensemble) |
| CatBoost | Single base learner |
| Stacking (Label-OOF) | Hard-label stacking (ablation baseline) |
| **Stacking-v2 (PSS) ★** | **Probability-Strength Stacking (proposed)** |

The PSS Stacking-v2 uses CatBoost + XGBoost + LightGBM + Random Forest as base learners, with Logistic Regression as the meta-learner trained on probabilistic OOF meta-features.

---

## Output Files

### Main figures (for manuscript)
| File | Description |
|---|---|
| `fig1_roc_all_models_stackingv2.*` | Fig. 1 — ROC curves, all models |
| `fig2_ablation_updated.*` | Fig. 2 — Ablation study |
| `fig3_shap_beeswarm_stackingv2.*` | Fig. 3 — SHAP beeswarm plot |
| `fig7_shap_dependence_academic_financial.*` | Fig. 4 — SHAP dependence plot |
| `fig5_dca_stackingv2.*` | Fig. 5 — Decision curve analysis |

### Supplementary figures
| File | Description |
|---|---|
| `supplfig1_shap_panel_3cases_stackingv2.*` | Suppl. Fig. 1 — SHAP individual cases |
| `violin_numerical_features.*` | Suppl. Fig. 2 — Violin plots |
| `pearson_heatmap.*` | Suppl. Fig. 3 — Pearson correlation heatmap |
| `supplfig4_cv_stability_stackingv2.*` | Suppl. Fig. 4 — CV stability |

### Statistics
| File | Description |
|---|---|
| `routeA_final_statistics.txt` | Final model statistics (AUC, CI, Brier, CV, DCA range) |
| `Table1_baseline_with_effect_size.csv` | Table 1 data |
| `model_comparison_updated.csv` | Table 2 data |

---

## Reproducibility Notes

- All random seeds are fixed at `random_state=42` throughout
- Data splitting uses stratified 8:2 train/test ratio
- 5-fold cross-validation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Bootstrap confidence intervals use 1,000 iterations with `RandomState(42)`
- Feature scaling uses `RobustScaler` fitted on training set only

---

## Citation

If you use this code in your research, please cite:

```
Sun J, Xu C. Development and Validation of a Probability-Strength Stacking
Ensemble Learning Model for Depression Risk Prediction in College Students:
SHAP-Based Interpretability Analysis. Journal of Affective Disorders. 2025
(under review).
```

---

## License

This code is released for academic research purposes only, consistent with the terms of the Kaggle dataset license.

---

## Contact

Corresponding author: Chenchen Xu
Nanjing Medical University, Nanjing, China
Email: [mlu@njmu.edu.cn]
