# Ridge Regression (L2) — Serum Creatinine (Heart Failure Clinical Records, UCI)

## Problem
Can we estimate a patient’s **serum creatinine** level (a kidney function biomarker) from demographic and cardiovascular measurements?

## Data
UCI Machine Learning Repository — **Heart Failure Clinical Records**  
Raw file: `data/raw/heart_failure_clinical_records_dataset.csv`  
Target: `serum_creatinine`

## ML Task
Supervised **Regression**

---

## Approach

### 1) Data split
- Hold out **30%** as a final **test set** (`test_size: 0.30`, `random_state: 42`)
- Use a **validation split** from the training set for iteration (`training.mode: val`, `val_size: 0.2`)

### 2) Target transform
- Apply `log1p` to the target (`log_target: true`) to reduce skew and stabilize variance.
- During evaluation, predictions and targets are converted back to the **original creatinine scale** using `expm1`.

### 3) Feature preprocessing (fit on train only → no leakage)
- Numeric: median imputation + standard scaling  
- Categorical (if present): most-frequent imputation + one-hot encoding  

Artifacts saved:
- Processed matrices: `data/processed/`
- Preprocessor: `models/preprocessor.joblib`
- Feature names: `data/processed/feature_names.joblib`

### 4) Model: Ridge Regression (L2)
- Model: **Ridge Regression** (linear model with L2 regularization)
- `fit_intercept: true`
- `alpha: 1.0` (regularization strength)

**Why Ridge?**  
Ridge regularization shrinks coefficients, improving stability when predictors are correlated and helping generalization on small/noisy datasets.

### 5) Evaluation
- Metrics reported on the **original target scale** (most interpretable):
  - MAE, RMSE, R²
- Residual diagnostics:
  - Residual distribution histogram
  - Residuals vs predictions
- Sanity-check baseline:
  - **Mean baseline predictor** (predict the mean training creatinine for everyone)

---

## Results (Held-out Test Set, Original Scale)

### Mean Baseline (predict mean creatinine for all patients)
- **MAE:** 0.5970  
- **RMSE:** 1.1099  
- **R²:** -0.0023  

### Ridge Regression (alpha = 1.0)
- **MAE:** 0.5685  
- **RMSE:** 1.1084  
- **R²:** 0.0004  

**Interpretation:**  
Ridge regression **slightly outperforms** the mean baseline on MAE/RMSE, but R² remains ~0. This suggests that (1) the relationship between features and creatinine may be weak in this dataset, (2) the mapping is likely **non-linear** and/or dominated by **outliers**, and (3) a linear model (even regularized) may be insufficient.

---

## Residual Diagnostics (What the plots show)
- Residuals on the original scale are **right-skewed**, with a small number of large positive residuals (severe under-predictions).
- Residuals vs predictions show **outliers** and mild heteroskedasticity, indicating difficulty modeling higher-creatinine cases.

Plots saved to:
- `reports/figures/residual_hist_orig.png`
- `reports/figures/residuals_vs_pred_orig.png`

---

## Limitations
- Ridge is still a **linear model** (additive effects, no interactions unless engineered).
- Clinical biomarkers often involve **non-linear relationships** and **heavy-tailed** error distributions.
- Small datasets can be sensitive to train/test splits, so performance estimates can vary.

---

## Next Steps (Improvements)
- Tune `alpha` using k-fold cross-validation (`training.mode: cv`) with a grid such as:
  - `[0.001, 0.01, 0.1, 1, 10, 100]`
- Try robust regression for outliers:
  - HuberRegressor / RANSAC
- Try non-linear models (often stronger on clinical data):
  - DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor

---


## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
