# src/train.py
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_matrix(path_base: str):
    """
    Loads X from either .npz (sparse) or .npy (dense).
    Provide base like 'data/processed/X_train' OR an actual path returned by preprocessing.
    """
    if path_base.endswith(".npz"):
        return sparse.load_npz(path_base)
    if path_base.endswith(".npy"):
        return np.load(path_base, allow_pickle=False)
    # If base provided without extension
    if os.path.exists(path_base + ".npz"):
        return sparse.load_npz(path_base + ".npz")
    return np.load(path_base + ".npy", allow_pickle=False)


def regression_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}



def train(config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)

    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    ensure_dir(models_dir)

    mode = cfg["training"].get("mode", "cv").lower()

    # Load processed data
    X_train = load_matrix(os.path.join(processed_dir, "X_train"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    feature_names = joblib.load(os.path.join(processed_dir, "feature_names.joblib"))

    model = LinearRegression(fit_intercept=cfg["model"].get("fit_intercept", True))

    train_report = {"training_mode": mode}

    if mode == "cv":
        k = int(cfg["training"].get("cv_folds", 5)) 
        kf = KFold(n_splits=k, shuffle=True, random_state=cfg["data"]["random_state"]) # Here, it splits training data into k folds, shuffles before splitting (otherwise folds might be ordered)

        fold_metrics = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(y_train), start=1): #Here, it's generating  pairs of indices like: '([train_indices], [validation_indices])'
            X_tr = X_train[tr_idx] if sparse.issparse(X_train) else X_train[tr_idx, :] # Here I'm selecting only rows that have index in tr_idx.
            X_va = X_train[va_idx] if sparse.issparse(X_train) else X_train[va_idx, :] # Here I'm selecting only rows that have index in va_idx.
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            model_fold = LinearRegression(fit_intercept=cfg["model"].get("fit_intercept", True))
            model_fold.fit(X_tr, y_tr) # here it's training the model for the actual fold only
            y_va_pred = model_fold.predict(X_va) # here it's predicts only for that fold.

            m = regression_metrics(y_va, y_va_pred)
            m["fold"] = fold
            fold_metrics.append(m) # Here, it stores MAE/RMSE/R² for each fold

        df = pd.DataFrame(fold_metrics)
        train_report["cv"] = {
            "folds": k,
            "MAE_mean": float(df["MAE"].mean()),
            "MAE_std": float(df["MAE"].std(ddof=1)),
            "RMSE_mean": float(df["RMSE"].mean()),
            "RMSE_std": float(df["RMSE"].std(ddof=1)),
            "R2_mean": float(df["R2"].mean()),
            "R2_std": float(df["R2"].std(ddof=1)),
        } # This creates mean and standard deviation across folds.

        # Fit final model on ALL training data after CV
        model.fit(X_train, y_train) # Here, After CV (which was just evaluation), it trains final model using all training data.

    elif mode == "val":
        X_val = load_matrix(os.path.join(processed_dir, "X_val"))
        y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

        model.fit(X_train, y_train) # Training once on val set
        y_val_pred = model.predict(X_val) # Predict result after val train
        train_report["val"] = regression_metrics(y_val, y_val_pred) # Here, we're evaluating on val set

    else:
        raise ValueError("training.mode must be 'cv' or 'val'")

    # Save model
    model_name = cfg["model"]["name"]
    model_path = os.path.join(models_dir, model_name)
    joblib.dump(model, model_path)
    train_report["model_path"] = model_path

    # Save coefficients (if feature names exist)
    if feature_names:
        coef_df = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_}) # Why do we need the coefficeient? For linear regression, coefficients are the “explanation”. this is how we write: “Top drivers of price were …etc..” 
        coef_df = coef_df.sort_values("coefficient", ascending=False)
        coef_path = os.path.join(models_dir, "coefficients.csv")
        coef_df.to_csv(coef_path, index=False)
        train_report["coefficients_path"] = coef_path
    else:
        train_report["coefficients_path"] = None

    
    reports_dir = cfg["paths"]["reports_dir"]
    ensure_dir(reports_dir)
    with open(os.path.join(reports_dir, "training_report.yaml"), "w", encoding="utf-8") as f: # This part writes everything learned during training to a human-readable file.
        yaml.safe_dump(train_report, f, sort_keys=False)

    return train_report


if __name__ == "__main__":
    report = train()
    print("✅ Training done.")
    print(report)
