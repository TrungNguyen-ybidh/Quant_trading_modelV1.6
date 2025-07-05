import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve
)

# === Universal Utilities ===
def one_hot_encode(df):
    df = df.copy().fillna(0)
    cat_cols = df.select_dtypes(include="object").columns
    return pd.get_dummies(df, columns=cat_cols)

# === CLASSIFICATION ===
def load_and_prepare_classifier_data(path):
    df = pd.read_csv(path)
    drop_cols = [
        "timestamp", "symbol", "exit_reason", "entry_price", "exit_price",
        "stop_loss", "tp1", "tp2", "r_multiple"
    ]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["is_valid_entry"]
    X = one_hot_encode(X)
    return X, y

def train_classifier(X, y, use_scaling=True, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler() if use_scaling else None
    X_train_scaled = scaler.fit_transform(X_train) if use_scaling else X_train
    X_test_scaled = scaler.transform(X_test) if use_scaling else X_test

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    model.feature_names_in_ = X.columns.tolist()  # For column alignment later

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)

    # Calculate best threshold from precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_threshold = thresholds[f1_scores.argmax()] if len(thresholds) > 0 else 0.5
    model.best_threshold = best_threshold  # Store for use in apply

    return model, scaler, report, y_pred, y_proba

def cross_validate_classifier(model, X, y, cv=5):
    pipeline = make_pipeline(StandardScaler(), model)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"\n🔁 Cross-Validation Accuracy (CV={cv}): {scores.mean():.4f} ± {scores.std():.4f}")
    return scores

# === REGRESSION ===
def load_and_prepare_regression_data(path, valid_only=True):
    df = pd.read_csv(path)
    if valid_only:
        df = df[df["is_valid_entry"] == 1]

    drop_cols = [
        "timestamp", "symbol", "exit_reason", "entry_price", "exit_price",
        "stop_loss", "tp1", "tp2", "is_valid_entry"
    ]
    X = df.drop(columns=drop_cols + ["r_multiple"], errors="ignore")
    y = df["r_multiple"]
    X = one_hot_encode(X)
    return X, y

def train_regressor(X, y, use_scaling=True, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler() if use_scaling else None
    X_train_scaled = scaler.fit_transform(X_train) if use_scaling else X_train
    X_test_scaled = scaler.transform(X_test) if use_scaling else X_test

    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    model.feature_names_in_ = X.columns.tolist()

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred)
    }
    return model, scaler, metrics, y_test, y_pred

def cross_validate_regressor(model, X, y, cv=5):
    pipeline = make_pipeline(StandardScaler(), model)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
    print(f"\n🔁 Cross-Validation R2 Score (CV={cv}): {scores.mean():.4f} ± {scores.std():.4f}")
    return scores

# === SHARED UTILITIES ===
def print_feature_importance(model):
    # If it's a pipeline, extract the final model
    if hasattr(model, "named_steps"):
        model = model.named_steps.get("randomforestclassifier", model)
    
    if not hasattr(model, "feature_importances_"):
        print("⚠️ Model does not support feature_importances_.")
        return

    print("\n📊 Feature Importances:")
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    for idx in sorted_idx[:20]:
        print(f"{model.feature_names_in_[idx]:<25} : {importances[idx]:.4f}")


def save_model(model, scaler, path_prefix):
    save_dir = "machine_learning"
    model_path = os.path.join(save_dir, f"{path_prefix}.pkl")
    joblib.dump(model, model_path)

    if scaler:
        scaler_path = os.path.join(save_dir, f"{path_prefix}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"✅ Saved: {model_path} + {scaler_path}")
    else:
        print(f"✅ Saved: {model_path}")

def load_model(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def apply_ml_models(df, clf_model, clf_scaler, reg_model, reg_scaler):
    df = df.copy()
    drop_cols = [
        "timestamp", "symbol", "exit_reason", "entry_price", "exit_price",
        "stop_loss", "tp1", "tp2", "r_multiple", "ml_valid_entry",
        "ml_prob", "ml_r_pred", "final_entry"
    ]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
    X = one_hot_encode(X)

    # Align classifier input
    clf_cols = clf_model.feature_names_in_
    for col in clf_cols:
        if col not in X.columns:
            X[col] = 0
    X_clf = X[clf_cols]
    X_clf_scaled = clf_scaler.transform(X_clf)

    # Align regressor input
    reg_cols = reg_model.feature_names_in_
    for col in reg_cols:
        if col not in X.columns:
            X[col] = 0
    X_reg = X[reg_cols]
    X_reg_scaled = reg_scaler.transform(X_reg)

    # Predict
    prob = clf_model.predict_proba(X_clf_scaled)[:, 1]
    threshold = getattr(clf_model, "best_threshold", 0.7)
    pred_class = (prob >= threshold).astype(int)
    r_pred = reg_model.predict(X_reg_scaled)

    df["ml_valid_entry"] = pred_class
    df["ml_prob"] = prob
    df["ml_r_pred"] = r_pred

    return df
