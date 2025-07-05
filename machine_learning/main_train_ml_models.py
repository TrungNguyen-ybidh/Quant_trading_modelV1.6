import pandas as pd
from ml_utils_combined import (
    load_and_prepare_classifier_data, train_classifier, print_feature_importance,
    cross_validate_classifier, save_model as save_clf
)
from ml_utils_combined import (
    load_and_prepare_regression_data, train_regressor, print_feature_importance as print_reg_importance,
    cross_validate_regressor, save_model as save_reg
)

def main():
    data_path = "trade_backtest_master_log.csv"

    print("📦 Loading data for classifier...")
    X_clf, y_clf = load_and_prepare_classifier_data(data_path)
    print("✅ Classifier data shape:", X_clf.shape)

    clf_model, clf_scaler, clf_report, clf_preds, clf_proba = train_classifier(X_clf, y_clf)
    print("📊 Classifier report:")
    print(pd.DataFrame(clf_report).transpose())
    print_feature_importance(clf_model)
    cross_validate_classifier(clf_model.__class__(), X_clf, y_clf)
    save_clf(clf_model, clf_scaler, "ml_model_clf")

    print("\n📦 Loading data for regressor...")
    X_reg, y_reg = load_and_prepare_regression_data(data_path, valid_only=True)
    print("✅ Regressor data shape:", X_reg.shape)

    reg_model, reg_scaler, reg_metrics, _, _ = train_regressor(X_reg, y_reg)
    print("📊 Regressor metrics:")
    for k, v in reg_metrics.items():
        print(f"{k.upper():<5}: {v:.4f}")
    print_reg_importance(reg_model)
    cross_validate_regressor(reg_model.__class__(), X_reg, y_reg)
    save_reg(reg_model, reg_scaler, "ml_model_reg")

if __name__ == "__main__":
    main()

