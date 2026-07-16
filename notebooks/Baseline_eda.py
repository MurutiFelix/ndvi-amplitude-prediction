# data/processed/Baseline_eda.py
"""
Baseline EDA & Spatial Visualizations for Upper Ewaso Nyiro Basin (UENRB).
Optimized for headless execution on HPC clusters or inline in JupyterLab.
"""

import os
import yaml
import numpy as np
import pandas as pd

# 1. Safe backend fallback for headless HPC cluster environments
import matplotlib
try:
    get_ipython()
    print("Detected Jupyter environment. Interactive/Inline plotting enabled.")
except NameError:
    print("Detected headless CLI environment. Switching matplotlib to 'Agg' backend.")
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def reconstruct_spatial_grid(series, height, width):
    """
    Maps grouped series values (where index = pixel_idx) back to their 
    correct geographical 2D spatial positions on a height x width grid.
    """
    grid = np.full((height * width), np.nan)
    # Map each value directly to its true flat 1D coordinate index
    grid[series.index.astype(int)] = series.values
    return grid.reshape((height, width))


def write_geotiff(filename, data, config, template_meta=None):
    """
    Writes a 2D numpy array to a georeferenced GeoTIFF conforming to UENRB bounds
    using template metadata or fallback values.
    """
    if template_meta:
        meta = template_meta.copy()
        meta.update({
            'driver': 'GTiff',
            'dtype': 'float32',
            'count': 1,
            'nodata': np.nan
        })
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(data.astype(np.float32), 1)
    else:
        # Fallback manual creation
        height, width = data.shape
        spatial_cfg = config.get('spatial', {})
        west = spatial_cfg.get('west', 36.5)
        north = spatial_cfg.get('north', 1.0)
        x_res = spatial_cfg.get('pixel_size_x', (37.75 - 36.5) / width)
        y_res = spatial_cfg.get('pixel_size_y', (1.0 - (-0.25)) / height)
        transform = from_origin(west, north, x_res, y_res)
        
        from rasterio.crs import CRS
        crs = CRS.from_epsg(4326)
        
        with rasterio.open(
            filename, 'w',
            driver='GTiff',
            height=height, width=width,
            count=1,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(data.astype(np.float32), 1)
            
    print(f"  -> Exported GeoTIFF to {filename}")


# ------------------------------------------------------------------
# Main Execution Pipeline
# ------------------------------------------------------------------

def run_eda_visualizations():
    # Dynamic config and folder path lookup
    possible_config_paths = ["src/config.yaml", "../../src/config.yaml", "../src/config.yaml"]
    config_path = None
    for p in possible_config_paths:
        if os.path.exists(p):
            config_path = p
            break
            
    if config_path is None:
        raise FileNotFoundError("Could not find config.yaml file.")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to where script is executed
    processed_dir = config['paths']['processed_dir']
    if not os.path.exists(processed_dir):
        possible_processed_dirs = ["data/processed", "../processed", "current"]
        for pd_path in possible_processed_dirs:
            if os.path.exists(pd_path):
                processed_dir = pd_path
                break
        else:
            processed_dir = "."

    static_dir = "data/static"
    if not os.path.exists(static_dir):
        static_dir = "../../data/static"
        if not os.path.exists(static_dir):
            static_dir = "../static"

    # ------------------------------------------------------------------
    # DYNAMIC RASTER DIMENSION DETECTION
    # ------------------------------------------------------------------
    # Look for TWI.tif or any raw raster to extract true dimensions & projection
    reference_raster_path = os.path.join(static_dir, "TWI.tif")
    if not os.path.exists(reference_raster_path):
        # Fallback to look inside data/raw for any .tif
        raw_dir = "data/raw"
        if os.path.exists(raw_dir):
            tifs = [f for f in os.listdir(raw_dir) if f.endswith(".tif")]
            if tifs:
                reference_raster_path = os.path.join(raw_dir, tifs[0])

    height = config['spatial']['height']
    width = config['spatial']['width']
    template_meta = None

    if os.path.exists(reference_raster_path):
        print(f"  Using reference raster to auto-detect grid size: {reference_raster_path}")
        with rasterio.open(reference_raster_path) as src:
            height = src.height
            width = src.width
            template_meta = src.meta.copy()
        print(f"  [Success] Auto-detected True Resolution: {height}x{width}")
    else:
        print(f"  [Warning] No reference raster found. Falling back to config size: {height}x{width}")

    # Output directory for presentation images
    fig_dir = os.path.join(processed_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Reading datasets from {processed_dir} and exporting maps to {fig_dir}...")

    # Load test residuals CSV
    residuals_csv = os.path.join(processed_dir, "test_residuals_dataframe.csv")
    if not os.path.exists(residuals_csv):
        print(f"[Error] Missing residuals file at {residuals_csv}. Please run your analyze_and_tune.py script first!")
        return

    df = pd.read_csv(residuals_csv)
    print(f"  Successfully loaded residuals with columns: {list(df.columns)}")
    
    # Handle possible baseline dataframe column remapping
    df = df.rename(columns={
        'actual': 'true_log_ndvi',
        'predicted': 'pred_log_ndvi',
        'error': 'residual'
    }, errors='ignore')

    # --- RECONSTRUCT SPATIAL MAP USING TABULAR DATA ---
    tabular_csv = os.path.join(processed_dir, "tabular_dataset.csv")
    if not os.path.exists(tabular_csv):
        tabular_csv = "tabular_dataset.csv" if os.path.exists("tabular_dataset.csv") else "../processed/tabular_dataset.csv"

    assigned_spatial_ids = False

    if os.path.exists(tabular_csv):
        print("  Aligning spatial node coordinates from tabular dataset...")
        tabular_df = pd.read_csv(tabular_csv)
        
        spatial_col = 'pixel_idx'
        
        # Filter tabular dataset exactly the same way baselines prepared the test mask
        train_split_year = config.get('features', {}).get('train_split_year', 2021)
        test_mask = tabular_df['year'] > train_split_year
        test_tabular = tabular_df[test_mask].copy()
        
        # Verify alignment sequence matches
        if len(test_tabular) == len(df):
            print(f"  [Success] Test set dimensions match! Assigning spatial values from '{spatial_col}'.")
            df['pixel_idx'] = test_tabular[spatial_col].values
            assigned_spatial_ids = True
        else:
            print(f"  [Warning] Row count mismatch (Residuals: {len(df)}, Test Tabular: {len(test_tabular)}).")
            unique_nodes = test_tabular[spatial_col].unique()
            df['pixel_idx'] = np.tile(unique_nodes, len(df) // len(unique_nodes) + 1)[:len(df)]
            assigned_spatial_ids = True
    
    if not assigned_spatial_ids:
        print("  [Warning] Could not map spatial columns from CSV. Using sequential index fallback.")
        df['pixel_idx'] = df.index % (height * width)

    # ==================================================================
    # MAP 1: XGBoost Mean Absolute Error (MAE) Residual Map
    # ==================================================================
    print("\n[1/4] Generating Spatial Residual Maps...")
    df['abs_residual'] = df['residual'].abs()
    
    mae_per_node = df.groupby('pixel_idx')['abs_residual'].mean()
    mae_grid = reconstruct_spatial_grid(mae_per_node, height, width)

    plt.figure(figsize=(8, 6))
    plt.imshow(mae_grid, cmap='YlOrRd', origin='upper')
    plt.title("XGBoost Mean Absolute Error (MAE) over UENRB")
    plt.colorbar(label="Mean Absolute Error (log NDVI)")
    plt.axis('off')
    plt.savefig(os.path.join(fig_dir, "xgboost_mae_spatial_map.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Create spatial bias map (Directional: Under (+) vs. Over (-))
    bias_per_node = df.groupby('pixel_idx')['residual'].mean()
    bias_grid = reconstruct_spatial_grid(bias_per_node, height, width)
    max_bias = np.nanmax(np.abs(bias_grid))

    plt.figure(figsize=(8, 6))
    plt.imshow(bias_grid, cmap='bwr', origin='upper', vmin=-max_bias, vmax=max_bias)
    plt.title("XGBoost Mean Spatial Bias (Actual - Predicted)")
    plt.colorbar(label="Underprediction (Red) vs. Overprediction (Blue)")
    plt.axis('off')
    plt.savefig(os.path.join(fig_dir, "xgboost_bias_spatial_map.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # MAP 2: Mean Observed vs Predicted (Back-Transformed NDVI)
    # ==================================================================
    print("\n[2/4] Generating Observed vs. Predicted NDVI Maps...")
    df['true_ndvi'] = np.exp(df['true_log_ndvi'])
    df['pred_ndvi'] = np.exp(df['pred_log_ndvi'])
    
    mean_true = df.groupby('pixel_idx')['true_ndvi'].mean()
    mean_pred = df.groupby('pixel_idx')['pred_ndvi'].mean()
    
    true_grid = reconstruct_spatial_grid(mean_true, height, width)
    pred_grid = reconstruct_spatial_grid(mean_pred, height, width)

    # Export GeoTIFF formats for direct visualization inside QGIS
    write_geotiff(os.path.join(processed_dir, "observed_mean_ndvi.tif"), true_grid, config, template_meta)
    write_geotiff(os.path.join(processed_dir, "predicted_mean_ndvi.tif"), pred_grid, config, template_meta)

    # Make side-by-side PNG figures for presentation slides
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(true_grid, cmap='YlGn', origin='upper', vmin=0.1, vmax=0.7)
    axes[0].set_title("Observed Mean NDVI (Test Set)")
    axes[0].axis('off')
    
    im = axes[1].imshow(pred_grid, cmap='YlGn', origin='upper', vmin=0.1, vmax=0.7)
    axes[1].set_title("XGBoost Predicted Mean NDVI (Test Set)")
    axes[1].axis('off')
    
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.6, label="Vegetation Index (NDVI)")
    plt.savefig(os.path.join(fig_dir, "observed_vs_predicted_ndvi_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # MAP 3: Spatial Context of the Top Feature (Precipitation Lag-1)
    # ==================================================================
    print("\n[3/4] Fetching spatial patterns of top feature...")
    if os.path.exists(tabular_csv):
        raw_df = pd.read_csv(tabular_csv)
        precip_col = 'log_precip_driver_lag1'
        if precip_col in raw_df.columns:
            mean_precip = raw_df.groupby('pixel_idx')[precip_col].mean()
            precip_grid = reconstruct_spatial_grid(mean_precip, height, width)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(precip_grid, cmap='Blues', origin='upper')
            plt.title("Spatial Context: Mean log_precip_driver_lag1 (Top Driver)")
            plt.colorbar(label="Log-transformed Precipitation Lag-1")
            plt.axis('off')
            plt.savefig(os.path.join(fig_dir, "top_feature_precipitation_map.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            write_geotiff(os.path.join(processed_dir, "feature_mean_precip_lag1.tif"), precip_grid, config, template_meta)
        else:
            print(f"  [Skip] Variable '{precip_col}' not found in the dataset.")
    else:
        print("  [Skip] Tabular dataset CSV missing.")

    # ==================================================================
    # MAP 4: TWI & Soil Type Overlaid with MAE Hotspots
    # ==================================================================
    print("\n[4/4] Extracting topographic features and clustering hot-spots...")
    
    # Identify 90th-percentile error cells
    high_error_threshold = np.nanpercentile(mae_per_node.values, 90)
    high_error_mask = (mae_per_node >= high_error_threshold).astype(float)
    error_mask_grid = reconstruct_spatial_grid(high_error_mask, height, width)

    twi_file = os.path.join(static_dir, "TWI.tif")
    if os.path.exists(twi_file):
        with rasterio.open(twi_file) as src:
            twi_raw = src.read(1)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(twi_raw, cmap='bone', origin='upper')
            # Overlay model hotspots in magenta
            plt.imshow(error_mask_grid, cmap='spring', alpha=0.35, origin='upper')
            plt.title("Topographic Wetness Index (TWI) with Top 10% MAE Hotspots (Magenta)")
            plt.axis('off')
            plt.savefig(os.path.join(fig_dir, "twi_with_error_hotspots.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("  -> Saved TWI hotspot overlay.")
    else:
        print(f"  [Skip] No TWI.tif found at {twi_file}")

    print(f"\nPipeline finished. Check your visual assets inside: {fig_dir}")


if __name__ == "__main__":
    run_eda_visualizations()