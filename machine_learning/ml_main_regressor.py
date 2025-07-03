# 1. Import utilities and setup
from ml_utils_regressor import (
    load_and_prepare_regression_data,
    train_regressor,
    print_feature_importance,
    cross_validate_regressor,
    save_model
)

# 2. Load dataset
X, y = load_and_prepare_regression_data("trade_backtest_master_log.csv")

# 3. Train the regression model
model, scaler, metrics, y_test, y_pred = train_regressor(X, y)

# 4. Show metrics
print("\n📈 Regression Metrics:")
for key, val in metrics.items():
    print(f"{key:<20}: {val:.4f}")

# 5. Feature importance
print_feature_importance(model, X.columns)

# 6. Cross-validation (R²)
cross_validate_regressor(model, X, y)

# 7. Save model + scaler
save_model(model, scaler, path_prefix="ml_model_reg")