# Predicting Building Heating Load with Linear Regression

## Problem
Estimate the heating energy required for residential buildings based on architectural and structural characteristics.

## Dataset
Energy Efficiency Dataset (UCI Machine Learning Repository)

Features include:
- Relative compactness
- Surface area
- Wall area
- Roof area
- Height
- Orientation
- Glazing area

Target:
- Heating Load (Y1)

## Approach
1. Baseline model predicting mean heating load
2. Feature scaling with StandardScaler
3. Linear Regression training using 5-fold cross-validation
4. Residual diagnostics to assess model assumptions

## Results

| Model | MAE | RMSE | R² |
|------|------|------|------|
| Baseline (mean) | 9.12 | 10.11 | -0.007 |
| Linear Regression (Test) | **2.15** | **2.97** | **0.913** |

Cross-validation confirmed strong generalization:
- CV MAE: 2.07 ± 0.20
- CV RMSE: 2.93 ± 0.26
- CV R²: 0.913 ± 0.023

## Residual Analysis
Residuals are centered around zero but show clear non-random structure and increasing variance at higher predicted values, indicating violations of linear assumptions and the presence of non-linear relationships between architectural features and heating load.

## Key Insights
- Linear regression captures the dominant signal in the dataset.
- Strong multicollinearity and geometric interactions limit model accuracy.
- Residual patterns suggest that non-linear models (e.g., Decision Trees or Gradient Boosting) are likely to improve performance.

## Conclusion
Linear regression provides a strong and interpretable baseline for heating load prediction but fails to fully model complex architectural interactions. This motivates the use of more flexible non-linear models in future work.

---


## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
