# src/models/train.py
import os
import yaml
import numpy as np
import pandas as pd
from src.data.dataset import build_tabular_dataset
from src.models.baselines import BaselineModelEvaluator

def main():
    # Load configuration parameters
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config['paths']['processed_dir'], exist_ok=True)
    processed_csv_path = os.path.join(config['paths']['processed_dir'], "tabular_dataset.csv")
    
    # If matrix is missing from data/processed/, compile it cleanly right now
    if not os.path.exists(processed_csv_path):
        print("Tabular dataset matrix not found in data/processed/. Running spatial compilation pipeline...")
        df_compiled = build_tabular_dataset(config)
        
        if df_compiled.empty:
            print("\n[ERROR]: Compiled dataset is completely empty! Check raster spatial intersections.")
            return
            
        df_compiled.to_csv(processed_csv_path, index=False)
        print(f"Successfully compiled and saved matrix to {processed_csv_path}")
    
    print("Loading compiled matrix for baseline validation...")
    df = pd.read_csv(processed_csv_path)
    
    # HARD TYPE CAST: Explicitly strip any 'object' wrappers from disk storage ---
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        
    # Drop any row that couldn't convert cleanly to clear anomalies
    df = df.dropna()
    
    # Initialize baseline engine
    evaluator = BaselineModelEvaluator(config)
    
    print("Executing chronological train/test splits...")
    X_train, X_test, y_train, y_test = evaluator.prepare_data(
        df, 
        train_split_year=config['features']['train_split_year']
    )
    
    # Clean out any remaining underlying pandas block-manager types
    X_train = pd.DataFrame(X_train).astype(np.float64)
    X_test = pd.DataFrame(X_test).astype(np.float64)
    y_train = pd.Series(y_train).astype(np.float64)
    y_test = pd.Series(y_test).astype(np.float64)
    
    print(f"Training shapes -> Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print("Training OLS, GLM, and Random Forest baselines ")
    
    results = evaluator.evaluate_all(X_train, X_test, y_train, y_test)
    metrics_df = pd.DataFrame(results).T

    print("\n BASELINE MODEL PERFORMANCE ON FORWARD TEST SET")
    print(metrics_df.to_string())

    print("\n GLM GAUSSIAN SUMMARY ")
    print(evaluator.glm_results.summary())
    
    # Save performance metrics next to your spatial targets
    metrics_output_path = os.path.join(config['paths']['processed_dir'], "baseline_metrics.csv")
    metrics_df.to_csv(metrics_output_path)
    print(f"\nBaseline metrics saved to {metrics_output_path}")
    print("Baseline modeling complete.")

if __name__ == "__main__":
    main()