# src/models/tune_baselines.py
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from src.models.baselines import NDVIBaselines

def run_tuning():
    # Load configuration
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("Loading data via baseline data prep pipeline...")
    evaluator = NDVIBaselines(config=config)
    
    # Load your processed tabular dataset
    df = pd.read_csv(config['paths']['processed_df'])
    X_train, X_test, y_train, y_test = evaluator.prepare_data(df)
    
    # Define a focused search space around your current baseline
    param_dist = {
        'max_depth': [6, 8, 10, 12],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1.0],      # L1 regularization
        'reg_lambda': [1.0, 5.0, 10.0]    # L2 regularization
    }
    
    base_xgb = xgb.XGBRegressor(
        n_estimators=300, # slightly lower for faster tuning
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    
    print("Launching Randomized Search CV across parameter distribution space...")
    # Using 3-fold CV to honor time constraints on large tabular rows
    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=10, 
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=1 # Let XGBoost handle internal parallelization via n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    print("\n=== OPTIMAL PARAMETERS FOUND ===")
    print(search.best_params_)
    print(f"Best CV R2 Score: {search.best_score_:.6f}")
    
    # Save parameters to processed directory
    pd.DataFrame([search.best_params_]).to_csv("data/processed/best_xgb_params.csv", index=False)

if __name__ == "__main__":
    run_tuning()