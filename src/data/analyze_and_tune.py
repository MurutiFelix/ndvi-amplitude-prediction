# src/data/analyze_and_tune.py
import os
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from src.models.baselines import NDVIBaselines


def main():
    # Load configuration
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Loading tabular dataset and preparing matrices...")
    evaluator = NDVIBaselines(config=config)
    processed_path = os.path.join(config['paths']['processed_dir'], "tabular_dataset.csv")
    df = pd.read_csv(processed_path)

    # Split data chronologically and apply standard scaling
    X_train, X_test, y_train, y_test = evaluator.prepare_data(
        df, train_split_year=config['features']['train_split_year']
    )

    # Train baselines and calculate metrics
    print(f"\nExecuting baseline evaluations across {X_train.shape[1]} features...")
    results = evaluator.evaluate_all(X_train, X_test, y_train, y_test)
    metrics_df = pd.DataFrame(results).T
    print("\n=== BASELINE MODEL PERFORMANCE ON FORWARD TEST SET ===")
    print(metrics_df.to_string())

    # --- GLM summary ---
    print("\n=== GLM GAUSSIAN SUMMARY ===")
    print(evaluator.glm_results.summary())
    glm_summary_path = os.path.join(config['paths']['processed_dir'], "glm_summary.txt")
    with open(glm_summary_path, "w") as f:
        f.write(str(evaluator.glm_results.summary()))
    print(f"GLM summary saved to {glm_summary_path}")

    # --- Save metrics ---
    metrics_df.to_csv(os.path.join(config['paths']['processed_dir'], "baseline_metrics.csv"))

    # --- Random Forest feature importances ---
    print("\nSaving Random Forest Feature Importances...")
    rf_fi = evaluator.rf_feature_importance
    rf_fi.to_csv(
        os.path.join(config['paths']['processed_dir'], "rf_feature_importance.csv"),
        header=["importance"]
    )
    print("Top 10 RF Features:\n", rf_fi.head(10))

    # --- XGBoost feature importances ---
    print("\nExtracting XGBoost Feature Importances...")
    xgb_model = evaluator.models["XGBoost"]
    xgb_fi = pd.Series(
        xgb_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    xgb_fi.to_csv(
        os.path.join(config['paths']['processed_dir'], "xgb_feature_importance.csv"),
        header=["importance"]
    )
    print("Top 10 XGBoost Features:\n", xgb_fi.head(10))

    # --- Residual analysis for spatial mapping ---
    print("\nComputing test set residuals for spatial mapping error analysis...")
    xgb_preds = xgb_model.predict(X_test)
    residuals  = y_test - xgb_preds

    error_diagnostics = pd.DataFrame({
        'true_log_ndvi' : y_test.values,
        'pred_log_ndvi' : xgb_preds,
        'residual'      : residuals.values
    })
    error_diagnostics.to_csv(
        os.path.join(config['paths']['processed_dir'], "test_residuals_dataframe.csv"),
        index=False
    )
    print(f"Residual analysis saved. Mean Absolute Error: {np.abs(residuals).mean():.5f}")

    # --- XGBoost hyperparameter tuning with chronologically safe CV ---
    print("\nInitializing hyperparameter tuning sweep on XGBoost matrix...")
    param_dist = {
        'max_depth'        : [6, 8, 10, 12],
        'learning_rate'    : [0.03, 0.05, 0.1],
        'subsample'        : [0.7, 0.8, 0.9],
        'colsample_bytree' : [0.6, 0.8, 1.0],
        'reg_alpha'        : [0, 0.1, 1.0],
        'reg_lambda'       : [1.0, 5.0, 10.0]
    }

    tuning_xgb = xgb.XGBRegressor(
        n_estimators=300,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    # TimeSeriesSplit preserves temporal order — no future data leaks into training folds
    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=tuning_xgb,
        param_distributions=param_dist,
        n_iter=30,
        cv=tscv,
        scoring='r2',
        random_state=42,
        n_jobs=1,
        verbose=2
    )

    print("Fitting Randomized Search across parameter permutations...")
    search.fit(X_train, y_train)

    print("\n=== SWEEP COMPLETED SUCCESSFULLY ===")
    print("Best Hyperparameters Discovered:", search.best_params_)
    print(f"Optimal Search Fold R2 Score:    {search.best_score_:.6f}")

    # Export discovered hyperparameters
    pd.DataFrame([search.best_params_]).to_csv(
        os.path.join(config['paths']['processed_dir'], "optimized_xgb_hyperparameters.csv"),
        index=False
    )
    print("Parameters saved to data/processed/.")


if __name__ == "__main__":
    main()