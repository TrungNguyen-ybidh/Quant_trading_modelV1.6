# ml_utils_regressor.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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

    X = X.fillna(0)
    cat_cols = X.select_dtypes(include="object").columns
    X = pd.get_dummies(X, columns=cat_cols)

    return X, y


def train_regressor(X, y, use_scaling=True, test_size=0.2, random_state=42):
    scaler = StandardScaler() if use_scaling else None
    X_scaled = scaler.fit_transform(X) if use_scaling else X

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    metrics = {"r2": r2, "mae": mae, "mse": mse}
    return model, scaler, metrics, y_test, y_pred



def print_feature_importance(model, feature_names):
    print("\n📊 Feature Importances:")
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    for idx in sorted_idx[:20]:
        print(f"{feature_names[idx]:<25} : {importances[idx]:.4f}")


def cross_validate_regressor(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print(f"\n🔁 Cross-Validation R2 Score (CV={cv}): {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def save_model(model, scaler, path_prefix="ml_model_r_multiple"):
    joblib.dump(model, f"{path_prefix}.pkl")
    if scaler:
        joblib.dump(scaler, f"{path_prefix}_scaler.pkl")
    print(f"✅ Model and scaler saved as {path_prefix}.pkl and {path_prefix}_scaler.pkl")

import joblib
import pandas as pd

def load_model(model_path="ml_model_reg.pkl", scaler_path="ml_model_reg_scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_with_model(model, scaler, df):
    df = df.copy().fillna(0)
    cat_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=cat_cols)

    X_scaled = scaler.transform(df)
    preds = model.predict(X_scaled)

    df["ml_r_pred"] = preds
    return df
