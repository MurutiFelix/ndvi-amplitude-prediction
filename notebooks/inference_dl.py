import os
# Prevent OpenMP and cluster threading crashes
os.environ["OMP_NUM_THREADS"] = "1"

import yaml
import torch
import numpy as np
import pandas as pd

from src.data.dataset import build_datasets


def generate_residuals():
    # 1. Parse Config Path safely
    possible_config_paths = ["src/config.yaml", "../../src/config.yaml", "../src/config.yaml"]
    config_path = next((p for p in possible_config_paths if os.path.exists(p)), None)
    
    if config_path is None:
        raise FileNotFoundError("Could not find config.yaml file.")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_dir = config['paths']['processed_dir']
    if not os.path.exists(processed_dir):
        processed_dir = next((p for p in ["data/processed", "../processed", "."] if os.path.exists(p)), ".")

    # 2. Extract Test Dataset & Target Scaler
    print("Building test dataset and extracting target scaler...")
    _, _, test_dataset = build_datasets(config, window_size=config['features']['window_size'])
    
    scaler = getattr(test_dataset, 'scaler', None)
    
    # Load tabular dataset for index mapping
    csv_path = os.path.join(processed_dir, "tabular_dataset.csv")
    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.sort_values(['year', 'month', 'pixel_idx']).reset_index(drop=True)
    
    timesteps = df_raw[['year', 'month']].drop_duplicates().sort_values(['year', 'month']).reset_index(drop=True)
    val_end_year = config['features']['val_end_year']
    test_t_indices = np.where((timesteps['year'] > val_end_year).values)[0]
    
    window_size = config['features']['window_size']
    valid_test_t = [t for t in test_t_indices if t - window_size >= 0]
    
    # Reconstruct spatio-temporal alignment matrix
    records = []
    for t in valid_test_t:
        row_time = timesteps.iloc[t]
        sub = df_raw[(df_raw['year'] == row_time['year']) & (df_raw['month'] == row_time['month'])]
        
        sub_indexed = sub.set_index('pixel_idx')
        for pix in range(test_dataset.n_nodes):
            if pix in sub_indexed.index:
                val = sub_indexed.loc[pix, 'log_ndvi']
                if isinstance(val, pd.Series):
                    val = val.values[0]
                records.append({'pixel_idx': pix, 'true_log_ndvi': val})
            else:
                records.append({'pixel_idx': pix, 'true_log_ndvi': np.nan})

    out_df = pd.DataFrame(records)
    N_expected = len(out_df)

    # 3. Model Checkpoints & Un-scaling
    model_checkpoints = {
        'STID': 'checkpoint_STID.pt',
        'DCRNN': 'checkpoint_DCRNN.pt',
        'GRUGCNModel': 'checkpoint_GRUGCNModel.pt',
        'GraphWaveNet': 'checkpoint_GraphWaveNet.pt'
    }

    for model_key, ckpt_name in model_checkpoints.items():
        ckpt_path = os.path.join(processed_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  [Warning] Checkpoint {ckpt_name} not found. Skipping {model_key}.")
            continue

        print(f"Processing checkpoint {ckpt_name}...")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        preds = None
        if isinstance(checkpoint, dict):
            for k in ['test_preds', 'predictions', 'preds', 'y_pred']:
                if k in checkpoint:
                    preds = checkpoint[k]
                    break
        elif isinstance(checkpoint, torch.Tensor):
            preds = checkpoint

        if preds is not None:
            if isinstance(preds, torch.Tensor):
                preds = preds.numpy()
            
            # Target (log_ndvi) is column 0 out of 9 dynamic features in scaler
            if scaler is not None and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                mean_val = scaler.mean_[0] if len(scaler.mean_) > 1 else scaler.mean_
                scale_val = scaler.scale_[0] if len(scaler.scale_) > 1 else scaler.scale_
                preds = (preds * scale_val) + mean_val
            elif 'scaler_mean' in checkpoint and 'scaler_std' in checkpoint:
                preds = (preds * checkpoint['scaler_std']) + checkpoint['scaler_mean']
            
            flat_preds = preds.flatten()
            
            # Align prediction length with test space length
            if len(flat_preds) >= N_expected:
                out_df[f'{model_key}_pred'] = flat_preds[:N_expected]
            else:
                padded = np.full(N_expected, np.nan)
                padded[:len(flat_preds)] = flat_preds
                out_df[f'{model_key}_pred'] = padded
                
            print(f"  -> Successfully extracted & inverse-scaled {model_key}")
        else:
            print(f"  [Error] Checkpoint tensor missing for {model_key}")

    # Remove non-observed region pixels
    out_df = out_df.dropna(subset=['true_log_ndvi']).reset_index(drop=True)

    # 4. Save CSV Output
    output_path = os.path.join(processed_dir, "dl_test_residuals_dataframe.csv")
    out_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated {output_path} ({len(out_df)} rows).")


if __name__ == "__main__":
    generate_residuals()