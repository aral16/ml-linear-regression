# Linear Regression — House Prices (Ames, Iowa)

## Problem
Predict residential sale prices using structural, demographic, and location-based housing features.

## Data
Kaggle — **House Prices: Advanced Regression Techniques**  
Training file: `data/raw/train.csv`

# 📥 How to Get the Dataset (Manual Kaggle Download)

Kaggle datasets cannot be redistributed, so you must download the data manually.

---

## Step-by-step Instructions

1. Go to the Kaggle competition page:  
   👉 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

2. Log into your Kaggle account.

3. Click the **Download** button (top-right of the page).

4. Extract the downloaded ZIP file.

5. Copy the following files into the project directory:



## ML Task
Supervised Regression

---

## Approach
1. Split data into Train/Test (20% test held out)
2. Log-transform target (`log1p(SalePrice)`) to reduce skew and stabilize variance
3. Preprocess features:
   - Numeric: median imputation + standard scaling
   - Categorical: most-frequent imputation + one-hot encoding
4. Train: Linear Regression
5. Evaluate:
   - 5-Fold Cross-Validation (log space)
   - Test metrics (converted back to original dollar scale)
6. Residual diagnostics + coefficient interpretation

---

## Cross-Validation Stability Guide

We measure variability across folds using:

\[
CV\% = \frac{std}{mean} \times 100
\]

| CV% Range | Interpretation |
|-----------|----------------|
| < 5% | Very stable model |
| 5–10% | Stable |
| 10–20% | Moderate variability |
| > 20% | Unstable model |

---

## Results

### Cross-Validation (log-price space)
| Metric | Mean | Std | CV% | Interpretation |
|--------|------|-----|-----|----------------|
| MAE | 0.1045 | 0.0036 | 3.4% | Excellent stability |
| RMSE | 0.1907 | 0.0350 | 18.4% | Moderate variability |
| R² | 0.7517 | 0.1062 | 14.1% | Acceptable variability |

### Test Set (original dollars)
| Metric | Value |
|--------|-------|
| MAE | $15,237 |
| RMSE | $22,496 |
| R² | **0.9275** |

See full metrics in `reports/metrics.yaml` and plots in `reports/figures/`.

## How to Interpret the Metrics (Concrete)

Understanding regression metrics is critical to evaluate how good the model really is.

### MAE — Mean Absolute Error = Average Dollar Mistake
MAE computes the average absolute difference between the real sale price and the model prediction.

\[
MAE = average(|y_{true} - y_{pred}|)
\]

**Your result:** \$15,237  

**Concrete meaning:**  
On average, the model is off by about **15k dollars per house**.

Example:
| Real Price | Predicted | Error |
|------------|-----------|-------|
| 200,000 | 215,000 | 15,000 |
| 300,000 | 285,000 | 15,000 |

MAE answers:
> “For a typical house, how wrong is the prediction in dollars?”

Smaller MAE = better accuracy.

---

### RMSE — Root Mean Squared Error = Penalizes Big Mistakes
RMSE squares each prediction error before averaging, which heavily penalizes large misses.

\[
RMSE = \sqrt{average((y_{true} - y_{pred})^2)}
\]

**Your result:** \$22,495  

**Concrete meaning:**  
The model occasionally makes larger errors (especially on expensive houses), and RMSE highlights those big mistakes.

Example:
| Error | Squared Contribution |
|-------|----------------------|
| 10k | 100M |
| 50k | 2,500M |

RMSE answers:
> “When the model is wrong, how severe are the worst errors?”

If RMSE is much larger than MAE → the model struggles on extreme cases.

---

### R² — Variance Explained = How Much Signal the Model Captures
R² measures how much of the variation in house prices is explained by the features.

\[
R^2 = 1 - \frac{model\ error}{variance\ of\ prices}
\]

**Your result:** 0.927  

**Concrete meaning:**  
The model explains **92.7% of the differences in house prices** across properties.  
Only ~7% of price variation remains unexplained noise.

Benchmarks:
| R² | Interpretation |
|----|---------------|
| 0.5 | Weak |
| 0.7 | Decent |
| 0.8 | Strong |
| 0.9+ | Very strong |

R² answers:
> “How much of the housing price structure is captured by the model?”

---

### Summary
- MAE → typical error per house (dollars)
- RMSE → highlights large mistakes
- R² → overall explanatory power of the model

Together, these metrics provide both accuracy and reliability insights.

---

## Residual Diagnostics

### Residual Histogram (original scale)
What to look for:
- Centered near 0 → unbiased predictions
- Symmetric distribution → no systematic over/under prediction
- Long tails → rare/extreme homes produce large errors

Observation:
- Residuals are centered near 0 with long tails, indicating good average fit but some extreme misses.

### Residuals vs Predictions (original scale)
What to look for:
- Random cloud around 0 → model captures structure well
- Funnel shape (spread increases with predictions) → heteroskedasticity
- Patterns/curves → non-linear structure not captured

Observation:
- Residual variance increases with predicted price (funnel shape), suggesting non-linear effects and weaker performance on high-value properties.

---

## Key Feature Insights (from coefficients)

> Coefficients were learned in **log-price space**. Positive coefficients increase predicted log-price; negative coefficients decrease it.

### Strong Positive Drivers (examples)
- Premium roof materials (Metal, Tar&Grv, WdShngl)
- Excellent garage quality/condition
- Premium neighborhoods (StoneBr, NridgHt, Crawfor)
- Larger living area (`GrLivArea`), higher quality (`OverallQual`)
- Newer homes (`YearBuilt`)

### Strong Negative Drivers (examples)
- Commercial zoning (`MSZoning_C (all)`)
- Poor garage quality/condition
- Inferior neighborhoods (e.g., MeadowV, Edwards)
- Abnormal sale conditions / rare categories (may be unstable due to low frequency)

Full coefficient table: `models/coefficients.csv`

---

## Conclusion
Linear Regression provides a strong baseline (R² ≈ 0.93 on the test set) and captures major global trends.  
Residual diagnostics indicate heteroskedasticity and non-linear relationships, suggesting that tree-based or boosting models are the next step for improved performance (Decision Trees / Random Forest / Gradient Boosting).

---

## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
