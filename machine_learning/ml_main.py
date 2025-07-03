from ml_utils_classification import load_and_prepare_data, train_classifier, print_feature_importance, save_model

X, y = load_and_prepare_data("trade_backtest_master_log.csv")
model, scaler, report, y_pred, y_proba = train_classifier(X, y)

print("🎯 Classification Report:")
for label, metrics in report.items():
    print(f"{label}: {metrics}")

print_feature_importance(model, X.columns)
save_model(model, scaler)
