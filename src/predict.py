# src/predict.py
"""
Inference script for spatiotemporal NDVI prediction.

Loads a trained model checkpoint and generates a spatial
prediction map (GeoTIFF) for a specified target month.

Usage:
    python -m src.predict \
        --model GraphWaveNet \
        --year 2024 \
        --month 6

Output:
    data/processed/predicted_NDVI_{model}_{year}_{month:02d}.tif
    data/processed/residual_NDVI_{model}_{year}_{month:02d}.tif
"""

import os
import argparse
import yaml
import numpy as np
import torch
import rioxarray
import xarray as xr
from sklearn.metrics import r2_score, mean_squared_error

from src.data.dataset import NDVIGraphDataset, build_datasets
from src.models.spatio_temporal import get_model, MODEL_REGISTRY
from src.utils.spatial import get_edge_index

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 12


def load_model(model_name, n_nodes, n_dynamic, n_static, checkpoint_path, window_size):
    """Instantiate and load model state dictionary weights."""
    model = get_model(
        name=model_name, 
        n_nodes=n_nodes, 
        n_dynamic=n_dynamic, 
        n_static=n_static, 
        window_size=window_size
    ).to(DEVICE)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def find_window(dataset, target_year, target_month):
    """Find dataset sequence index matching target time parameters."""
    for idx, (start, t) in enumerate(dataset.windows):
        ts = dataset.timesteps.iloc[t]
        if ts['year'] == target_year and ts['month'] == target_month:
            return idx
    raise ValueError(f"No window found with target {target_year}-{target_month:02d}.")


@torch.no_grad()
def predict_month(model, dataset, window_idx, edge_index, model_name):
    """Run structural inference step conforming to exact framework parameters."""
    x, u, y = dataset[window_idx]
    
    # Reshape features to match batch layout: [Batch=1, Time, Nodes, Features]
    x = x.unsqueeze(0).to(DEVICE)   
    u = u.unsqueeze(0).to(DEVICE)   
    ei = edge_index.to(DEVICE)

    if model_name == 'STID':
        out = model(x, u=None)
    elif model_name in ('DCRNN', 'GRUGCNModel', 'GraphWaveNet'):
        out = model(x, edge_index=ei, u=u)
    else:
        out = model(x, edge_index=ei)

    if out.dim() == 4:
        out = out[:, 0, :, :]

    y_pred = out.squeeze().cpu().numpy()
    y_true = y.squeeze().cpu().numpy()
    return y_pred, y_true


def save_geotiff(values, height, width, template_path, output_path):
    """Convert flat spatial predictions back into aligned geospatial GeoTIFF targets."""
    grid = values.reshape(height, width).astype(np.float32)
    template = rioxarray.open_rasterio(template_path).squeeze()
    
    output = xr.DataArray(
        grid[np.newaxis, :, :],
        dims   = ['band', 'y', 'x'],
        coords = {'band': [1], 'y': template.y.values, 'x': template.x.values}
    )
    output = output.rio.write_crs(template.rio.crs)
    output = output.rio.write_transform(template.rio.transform())
    output.rio.to_raster(output_path)
    print(f"GeoTIFF saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GraphWaveNet', choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    args = parser.parse_args()

    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    height, width = config['spatial']['height'], config['spatial']['width']
    train_dataset, test_dataset = build_datasets(config, window_size=WINDOW_SIZE)
    dataset = test_dataset if args.split == 'test' else train_dataset

    edge_index = get_edge_index(height, width, os.path.join(config['paths']['processed_dir'], "edge_index.pt"))
    checkpoint_path = os.path.join(config['paths']['processed_dir'], f"checkpoint_{args.model}.pt")
    
    model = load_model(args.model, dataset.n_nodes, dataset.n_dynamic_features, dataset.n_static_features, checkpoint_path, WINDOW_SIZE)
    window_idx = find_window(dataset, args.year, args.month)
    
    y_pred, y_true = predict_month(model, dataset, window_idx, edge_index, args.model)

    valid = ~np.isnan(y_true) & (y_true != 0)
    if valid.sum() > 0:
        print(f"Timestep metrics — R²: {r2_score(y_true[valid], y_pred[valid]):.4f} | RMSE: {np.sqrt(mean_squared_error(y_true[valid], y_pred[valid])):.4f}")

    save_geotiff(y_pred, height, width, config['paths']['ndvi_template'], os.path.join(config['paths']['processed_dir'], f"predicted_NDVI_{args.model}_{args.year}_{args.month:02d}.tif"))
    save_geotiff(y_true - y_pred, height, width, config['paths']['ndvi_template'], os.path.join(config['paths']['processed_dir'], f"residual_NDVI_{args.model}_{args.year}_{args.month:02d}.tif"))


if __name__ == "__main__":
    main()