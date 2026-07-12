import os
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from src.models.baselines import NDVIBaselines

def main():
    # Load configuration
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("Loading tabular dataset and preparing matrices...")
    evaluator = NDVIBaselines(config=config)
    processed_path = os.path.join(config['paths']['processed_dir'], "tabular_dataset.csv")
    df = pd.read_csv(processed_path)
    
    # Define the full set of 13 continuous features to scale
    evaluator.continuous_cols = [
        'lst_driver_lag1', 'lst_driver_lag2', 'lst_driver_lag3',
        'log_precip_driver_lag1', 'log_precip_driver_lag2', 'log_precip_driver_lag3',
        'ndvi_spatial_lag', 'pop_density',
        'twi', 'year_trend', 'lst_x_precip', 'twi_x_precip', 'twi_x_lst'
    ]
    
    # Split data chronologically and apply standard scaling
    X_train, X_test, y_train, y_test = evaluator.prepare_data(df)
    
    # Train baselines and calculate metrics
    print(f"\nExecuting baseline evaluations across {X_train.shape[1]} features...")
    results = evaluator.evaluate_all(X_train, X_test, y_train, y_test)
    
    # Process Random Forest feature importances
    print("\nSaving Random Forest Feature Importances...")
    rf_fi = evaluator.rf_feature_importance
    rf_fi.to_csv("data/processed/rf_feature_importance.csv")
    print("Top 10 RF Features:\n", rf_fi.head(10))
    
    # Process XGBoost feature importances
    print("\nExtracting XGBoost Feature Importances...")
    xgb_model = evaluator.models["XGBoost"]
    xgb_fi = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    xgb_fi.to_csv("data/processed/xgb_feature_importance.csv")
    print("Top 10 XGBoost Features:\n", xgb_fi.head(10))

    # Calculate model errors on the test split for spatial mapping
    print("\nComputing test set residuals for spatial mapping error analysis...")
    xgb_preds = xgb_model.predict(X_test)
    residuals = y_test - xgb_preds
    
    error_diagnostics = pd.DataFrame({
        'true_log_ndvi': y_test,
        'pred_log_ndvi': xgb_preds,
        'residual': residuals
    })
    error_diagnostics.to_csv("data/processed/test_residuals_dataframe.csv", index=False)
    print(f"Residual analysis dumped. Mean Absolute Error: {np.abs(residuals).mean():.5f}")

    # Set up random search cross validation for XGBoost parameters
    print("\nInitializing hyperparameter tuning sweep on XGBoost matrix...")
    param_dist = {
        'max_depth': [6, 8, 10, 12],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [1.0, 5.0, 10.0]
    }
    
    tuning_xgb = xgb.XGBRegressor(
        n_estimators=300, 
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    
    # Execute a 3-fold chronological split optimization sweep
    search = RandomizedSearchCV(
        estimator=tuning_xgb,
        param_distributions=param_dist,
        n_iter=8,
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=1
    )
    
    print("Fitting Randomized Search Grid across parameter permutations...")
    search.fit(X_train, y_train)
    
    print("\n=== SWEEP COMPLETED SUCCESSFULLY ===")
    print("Best Hyperparameters Discovered:", search.best_params_)
    print(f"Optimal Search Fold R2 Score: {search.best_score_:.6f}")
    
    # Export discovered hyperparameter mappings
    pd.DataFrame([search.best_params_]).to_csv("data/processed/optimized_xgb_hyperparameters.csv", index=False)
    print("Parameters saved successfully to data/processed/.")

if __name__ == "__main__":
    main()