import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def load_and_prepare_data(path):
    df = pd.read_csv(path)

    drop_cols = [
        "timestamp", "symbol", "exit_reason", "entry_price", "exit_price",
        "stop_loss", "tp1", "tp2", "r_multiple"
    ]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["is_valid_entry"]

    # Fill missing values and encode categoricals
    X = X.fillna(0)

    cat_cols = X.select_dtypes(include="object").columns
    X = pd.get_dummies(X, columns=cat_cols)

    return X, y


def train_classifier(X, y, use_scaling=True, test_size=0.2, random_state=42):
    scaler = StandardScaler() if use_scaling else None
    X_scaled = scaler.fit_transform(X) if use_scaling else X

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # 👇 Store column names manually
    model.feature_names_in_ = X.columns.tolist()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, scaler, report, y_pred, y_proba


def predict_with_threshold(model, X_scaled, threshold=0.75):
    proba = model.predict_proba(X_scaled)[:, 1]
    return (proba > threshold).astype(int), proba

def print_feature_importance(model, feature_names):
    print("\n📊 Feature Importances:")
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    for idx in sorted_idx:
        print(f"{feature_names[idx]:<25} : {importances[idx]:.4f}")

def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"\n🔁 Cross-Validation Accuracy (CV={cv}): {scores.mean():.4f} ± {scores.std():.4f}")
    return scores

def save_model(model, scaler, path_prefix="ml_model"):
    joblib.dump(model, f"{path_prefix}_clf.pkl")
    if scaler:
        joblib.dump(scaler, f"{path_prefix}_scaler.pkl")
    print(f"✅ Model and scaler saved as {path_prefix}_clf.pkl and _scaler.pkl")


def apply_ml_models(df, clf_model, clf_scaler, reg_model, reg_scaler):
    features = df.copy().fillna(0)
    cat_cols = features.select_dtypes(include="object").columns
    features = pd.get_dummies(features, columns=cat_cols)

    # Align columns to what model expects
    clf_input = clf_scaler.transform(features)
    reg_input = reg_scaler.transform(features)

    # Predict
    probas = clf_model.predict_proba(clf_input)[:, 1]
    r_preds = reg_model.predict(reg_input)
    entry_preds = (probas >= 0.7).astype(int)

    df["ml_valid_entry"] = entry_preds
    df["ml_prob"] = probas
    df["ml_r_pred"] = r_preds

    return df


def load_model(model_path="ml_model_clf.pkl", scaler_path="ml_model_clf_scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_with_model(model, scaler, df, threshold=0.7):
    df = df.copy().fillna(0)
    cat_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=cat_cols)

    X_scaled = scaler.transform(df)
    proba = model.predict_proba(X_scaled)[:, 1]
    preds = (proba >= threshold).astype(int)

    df["ml_valid_entry"] = preds
    df["ml_prob"] = proba
    return df
