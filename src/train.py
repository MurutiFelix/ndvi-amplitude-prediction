# src/train.py
"""
Root execution orchestrator for the NDVI prediction pipeline.

Routes to either:
    - Baseline pipeline (OLS, GLM, RF, XGBoost) via analyze_and_tune.py
    - Deep Learning pipeline (STID, DCRNN, GRUGCNModel, GraphWaveNet) via models/train.py

Controlled by config.yaml:
    features:
        mode: "baselines"  # or "dl"
"""

import os
import sys
import yaml


def main():
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    mode = config['features'].get('mode', 'dl').lower().strip()

    print(f"{'='*60}")
    print(f"  NDVI Prediction Pipeline")
    print(f"  Mode: {mode.upper()}")
    print(f"{'='*60}\n")

    if mode == 'baselines':
        print("Routing to baseline pipeline (OLS, GLM, RF, XGBoost)...")
        from src.data.analyze_and_tune import main as run_baselines
        run_baselines()

    elif mode == 'dl':
        print("Routing to deep learning pipeline (TSL graph models)...")
        from src.models.train import main as run_dl
        run_dl()

    else:
        print(f"[ERROR] Unknown mode '{mode}' in config.yaml.")
        print("        Set features.mode to 'baselines' or 'dl'.")
        sys.exit(1)


if __name__ == "__main__":
    main()