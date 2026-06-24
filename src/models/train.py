# src/models/train.py
import os
import yaml
import pandas as pd
from src.data.dataset import build_tabular_dataset
from src.models.baselines import BaselineModelEvaluator

def main():
    # Load configuration parameters
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    processed_csv_path = os.path.join(config['paths']['processed_dir'], "tabular_dataset.csv")
    
    # If matrix is missing from data/processed/, compile it cleanly right now
    if not os.path.exists(processed_csv_path):
        print("Tabular dataset matrix not found in data/processed/. Running spatial compilation pipeline...")
        df_compiled = build_tabular_dataset(config)
        df_compiled.to_csv(processed_csv_path, index=False)
        print(f"Successfully compiled and saved matrix to {processed_csv_path}")
    
    print("Loading compiled matrix for baseline validation...")
    df = pd.read_csv(processed_csv_path)
    
    # Initialize baseline engine
    evaluator = BaselineModelEvaluator(config)
    
    print("Executing chronological train/test splits...")
    X_train, X_test, y_train, y_test = evaluator.prepare_data(
        df, 
        train_split_year=config['features']['train_split_year']
    )
    
    print(f"Training shapes -> Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print("Training OLS, GLM, and Random Forest baselines side-by-side...")
    
    metrics_df = evaluator.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print("\n=== BASELINE MODEL PERFORMANCE ON FORWARD TEST SET (2022-2025) ===")
    print(metrics_df.to_string())
    
    # Save performance metrics next to your Mann-Kendall trend targets
    metrics_df.to_csv(os.path.join(config['paths']['processed_dir'], "baseline_metrics.csv"))
    print("\nBaseline modeling complete. Output metrics successfully saved.")

if __name__ == "__main__":
    main()