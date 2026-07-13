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
"""

import os
import argparse
import yaml
import numpy as np
import torch
import rioxarray
import xarray as xr
from rioxarray.exceptions import NoDataInBounds

from src.data.dataset import NDVIGraphDataset, build_datasets
from src.models.spatio_temporal import get_model, MODEL_REGISTRY
from src.utils.spatial import get_edge_index


DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 12


def load_model(model_name, n_nodes, n_dynamic, n_static,
               checkpoint_path, window_size):
    """Load model architecture and restore trained weights."""
    model = get_model(
        name        = model_name,
        n_nodes     = n_nodes,
        n_dynamic   = n_dynamic,
        n_static    = n_static,
        window_size = window_size,
    ).to(DEVICE)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train the model first with: python -m src.train"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def find_window(dataset, target_year, target_month):
    """
    Find the dataset window index whose target timestep
    matches the requested year and month.

    Returns:
        idx (int): Index into dataset.windows
    """
    for idx, (start, t) in enumerate(dataset.windows):
        ts = dataset.timesteps.iloc[t]
        if ts['year'] == target_year and ts['month'] == target_month:
            return idx
    raise ValueError(
        f"No window found with target {target_year}-{target_month:02d}. "
        f"Ensure this timestep exists in the test split."
    )


@torch.no_grad()
def predict_month(model, dataset, window_idx, edge_index, model_name):
    """
    Run inference for a single window.

    Returns:
        y_pred (np.ndarray): Predicted log_ndvi, shape [n_nodes]
        y_true (np.ndarray): Observed log_ndvi, shape [n_nodes]
    """
    x, u, y = dataset[window_idx]

    x  = x.unsqueeze(0).to(DEVICE)   # [1, T, N, F]
    u  = u.unsqueeze(0).to(DEVICE)   # [1, N, F_static]
    ei = edge_index.to(DEVICE)

    if model_name == 'STID':
        out = model(x)
    elif model_name in ('DCRNN', 'GRUGCNModel'):
        out = model(x, edge_index=ei, u=u)
    elif model_name == 'GraphWaveNet':
        out = model(x, edge_index=ei, u=u)
    else:
        out = model(x, edge_index=ei)

    if out.dim() == 4:
        out = out[:, 0, :, :]   # [1, N, 1]

    y_pred = out.squeeze().cpu().numpy()    # [N]
    y_true = y.squeeze().cpu().numpy()      # [N]

    return y_pred, y_true


def save_geotiff(values, height, width, template_path, output_path):
    """
    Reshape flat [N] array back to [H, W] and save as GeoTIFF,
    inheriting CRS and spatial metadata from the NDVI template raster.

    Args:
        values        : Flat array of shape [N] (predicted log_ndvi)
        height        : Raster height (159)
        width         : Raster width (181)
        template_path : Path to NDVI_2006_01.tif for CRS/transform
        output_path   : Output .tif path
    """
    grid = values.reshape(height, width).astype(np.float32)

    template = rioxarray.open_rasterio(template_path).squeeze()

    output = xr.DataArray(
        grid[np.newaxis, :, :],           # [1, H, W]
        dims   = ['band', 'y', 'x'],
        coords = {
            'band': [1],
            'y'   : template.y.values,
            'x'   : template.x.values,
        }
    )
    output = output.rio.write_crs(template.rio.crs)
    output = output.rio.write_transform(template.rio.transform())
    output.rio.to_raster(output_path)
    print(f"GeoTIFF saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate spatial NDVI prediction map for a target month."
    )
    parser.add_argument(
        '--model', type=str, default='GraphWaveNet',
        choices=list(MODEL_REGISTRY.keys()),
        help="Model name to use for inference."
    )
    parser.add_argument(
        '--year', type=int, required=True,
        help="Target prediction year (e.g. 2024)."
    )
    parser.add_argument(
        '--month', type=int, required=True,
        help="Target prediction month 1-12 (e.g. 6)."
    )
    parser.add_argument(
        '--split', type=str, default='test',
        choices=['train', 'test'],
        help="Dataset split to draw the window from. Default: test."
    )
    args = parser.parse_args()

    # --- Load config ---
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    height = config['spatial']['height']
    width  = config['spatial']['width']

    # --- Build dataset ---
    print(f"\nBuilding {args.split} dataset...")
    train_dataset, test_dataset = build_datasets(
        config, window_size=WINDOW_SIZE
    )
    dataset = test_dataset if args.split == 'test' else train_dataset

    # --- Build edge index ---
    cache_path = os.path.join(
        config['paths']['processed_dir'], "edge_index.pt"
    )
    edge_index = get_edge_index(height, width, cache_path)

    # --- Load model ---
    checkpoint_path = os.path.join(
        config['paths']['processed_dir'],
        f"checkpoint_{args.model}.pt"
    )
    model = load_model(
        model_name      = args.model,
        n_nodes         = dataset.n_nodes,
        n_dynamic       = dataset.n_dynamic_features,
        n_static        = dataset.n_static_features,
        checkpoint_path = checkpoint_path,
        window_size     = WINDOW_SIZE,
    )

    # --- Find window for target month ---
    print(f"\nFinding window for {args.year}-{args.month:02d}...")
    window_idx = find_window(dataset, args.year, args.month)
    print(f"Found at dataset index {window_idx}")

    # --- Run inference ---
    print("Running inference...")
    y_pred, y_true = predict_month(
        model, dataset, window_idx, edge_index, args.model
    )

    # --- Metrics for this timestep ---
    from sklearn.metrics import r2_score, mean_squared_error
    valid = ~np.isnan(y_true) & (y_true != 0)
    if valid.sum() > 0:
        r2   = r2_score(y_true[valid], y_pred[valid])
        rmse = np.sqrt(mean_squared_error(y_true[valid], y_pred[valid]))
        print(f"Timestep metrics — R²: {r2:.4f} | RMSE: {rmse:.4f}")

    # --- Save predicted GeoTIFF ---
    output_filename = (
        f"predicted_NDVI_{args.model}_{args.year}_{args.month:02d}.tif"
    )
    output_path = os.path.join(
        config['paths']['processed_dir'], output_filename
    )
    save_geotiff(
        values        = y_pred,
        height        = height,
        width         = width,
        template_path = config['paths']['ndvi_template'],
        output_path   = output_path,
    )

    # --- Also save residual map ---
    residual_filename = (
        f"residual_NDVI_{args.model}_{args.year}_{args.month:02d}.tif"
    )
    residual_path = os.path.join(
        config['paths']['processed_dir'], residual_filename
    )
    save_geotiff(
        values        = y_true - y_pred,
        height        = height,
        width         = width,
        template_path = config['paths']['ndvi_template'],
        output_path   = residual_path,
    )

    print(f"\nDone. Prediction and residual maps saved to data/processed/")


if __name__ == "__main__":
    main()