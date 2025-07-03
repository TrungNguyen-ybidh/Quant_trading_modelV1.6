# === File: machine_learning/ml_utils_combined.py ===

import pandas as pd
import numpy as np
import joblib

def apply_ml_models(df, clf_model, clf_scaler, reg_model, reg_scaler):
    df = df.copy()

    # --- Preprocessing ---
    drop_cols = [
        "timestamp", "symbol", "exit_reason", "entry_price", "exit_price",
        "stop_loss", "tp1", "tp2", "r_multiple", "ml_valid_entry", 
        "ml_prob", "ml_r_pred", "final_entry"
    ]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
    X = X.fillna(0)

    # One-hot encode categoricals
    cat_cols = X.select_dtypes(include="object").columns
    X = pd.get_dummies(X, columns=cat_cols)
    # === Align with classifier model columns ===
    expected_clf_cols = clf_model.feature_names_in_
    for col in expected_clf_cols:
         if col not in features.columns:
                features[col] = 0  # Fill missing cols with 0
    features = features[expected_clf_cols]


    # Align columns to model training format (fill missing columns if any)
    expected_cols = clf_scaler.feature_names_in_
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_cols]  # Reorder

    # --- Scaling ---
    X_scaled_clf = clf_scaler.transform(X)
    X_scaled_reg = reg_scaler.transform(X)

    # --- Predictions ---
    prob = clf_model.predict_proba(X_scaled_clf)[:, 1]
    pred_class = (prob > 0.7).astype(int)  # Threshold can be tuned

    r_pred = reg_model.predict(X_scaled_reg)

    # --- Output ---
    df["ml_valid_entry"] = pred_class
    df["ml_prob"] = prob
    df["ml_r_pred"] = r_pred

    return df
