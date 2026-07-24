import os
import yaml
import numpy as np
import pandas as pd
import matplotlib

try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.transform import from_origin


# ==================================================================
# 1. SPATIAL GRID RECONSTRUCTION & GEOTIFF EXPORTER
# ==================================================================
def reconstruct_spatial_grid(series, height, width, valid_mask=None):
    """Maps grouped series values back to geographical 2D spatial grid using Row-Major order."""
    grid = np.full((height * width), np.nan)
    valid_idx_mask = (series.index >= 0) & (series.index < (height * width))
    valid_series = series[valid_idx_mask]

    grid[valid_series.index.astype(int)] = valid_series.values
    grid_2d = grid.reshape((height, width), order='C')
    
    if valid_mask is not None:
        grid_2d[~valid_mask] = np.nan
        
    return grid_2d


def write_geotiff(filename, data, config, template_meta=None):
    """Saves a 2D spatial array directly as a GeoTIFF raster (.tif)."""
    if template_meta:
        meta = template_meta.copy()
        meta.update({'driver': 'GTiff', 'dtype': 'float32', 'count': 1, 'nodata': np.nan})
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(data.astype(np.float32), 1)
    else:
        height, width = data.shape
        spatial_cfg = config.get('spatial', {})
        west = spatial_cfg.get('west', 36.5)
        north = spatial_cfg.get('north', 1.0)
        x_res = spatial_cfg.get('pixel_size_x', (37.75 - 36.5) / width)
        y_res = -abs(spatial_cfg.get('pixel_size_y', (1.0 - (-0.25)) / height))
        transform = from_origin(west, north, x_res, abs(y_res))
        from rasterio.crs import CRS
        with rasterio.open(filename, 'w', driver='GTiff', height=height, width=width, count=1,
                           dtype=rasterio.float32, crs=CRS.from_epsg(4326), transform=transform, nodata=np.nan) as dst:
            dst.write(data.astype(np.float32), 1)


# ==================================================================
# 2. MAIN EDA PIPELINE
# ==================================================================
def run_dl_eda_visualizations():
    possible_config_paths = ["src/config.yaml", "../../src/config.yaml", "../src/config.yaml"]
    config_path = next((p for p in possible_config_paths if os.path.exists(p)), None)
    if config_path is None:
        raise FileNotFoundError("Could not find config.yaml file.")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_dir = config['paths']['processed_dir']
    if not os.path.exists(processed_dir):
        processed_dir = next((p for p in ["data/processed", "../processed", "."] if os.path.exists(p)), ".")

    # Isolated output directory for all DL artifacts
    dl_out_dir = os.path.join(processed_dir, "dl_processed")
    fig_dir = os.path.join(dl_out_dir, "figures")
    tif_dir = os.path.join(dl_out_dir, "spatial_rasters")
    
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tif_dir, exist_ok=True)

    static_dir = next((p for p in ["data/static", "../../data/static", "../static"] if os.path.exists(p)), ".")
    reference_raster_path = os.path.join(static_dir, "TWI.tif")
    
    height = config['spatial']['height']
    width = config['spatial']['width']
    template_meta = None
    spatial_shape_mask = None

    if os.path.exists(reference_raster_path):
        with rasterio.open(reference_raster_path) as src:
            height, width = src.height, src.width
            template_meta = src.meta.copy()
            twi_data = np.ascontiguousarray(src.read(1))
            spatial_shape_mask = ~np.isnan(twi_data) & (twi_data > -9000)

    dl_residuals_csv = os.path.join(processed_dir, "dl_test_residuals_dataframe.csv")
    if not os.path.exists(dl_residuals_csv):
        print(f"[Notice] {dl_residuals_csv} not found.")
        return

    df = pd.read_csv(dl_residuals_csv)

    num_spatial_nodes = height * width
    if 'pixel_idx' in df.columns:
        df['pixel_idx'] = df['pixel_idx'] % num_spatial_nodes
    else:
        df['pixel_idx'] = df.index % num_spatial_nodes

    models_to_process = ['STID', 'DCRNN', 'GRUGCNModel', 'GraphWaveNet']

    cmap_ndvi = plt.cm.YlGn.copy()
    cmap_ndvi.set_bad(color='none')

    cmap_mae = plt.cm.YlOrRd.copy()
    cmap_mae.set_bad(color='none')

    cmap_bias = plt.cm.RdBu_r.copy()
    cmap_bias.set_bad(color='none')

    # 1. OBSERVED GROUND TRUTH RASTER
    print("\n[1/4] Processing Observed Mean NDVI...")
    df['true_ndvi'] = np.exp(df['true_log_ndvi'])
    mean_true = df.groupby('pixel_idx')['true_ndvi'].mean()
    true_grid = reconstruct_spatial_grid(mean_true, height, width, valid_mask=spatial_shape_mask)
    
    # Save Observed GeoTIFF
    write_geotiff(os.path.join(tif_dir, "observed_mean_ndvi.tif"), true_grid, config, template_meta)

    mae_grids = {}
    bias_grids = {}

    # 2. INDIVIDUAL OBSERVED VS PREDICTED PNGs + PREDICTION TIFs
    print("\n[2/4] Generating individual model figures and prediction rasters...")
    for model in models_to_process:
        pred_col = f'{model}_pred'
        if pred_col not in df.columns:
            continue

        df[f'{model}_pred_ndvi'] = np.exp(df[pred_col])
        mean_pred = df.groupby('pixel_idx')[f'{model}_pred_ndvi'].mean()
        pred_grid = reconstruct_spatial_grid(mean_pred, height, width, valid_mask=spatial_shape_mask)

        # Export Model Prediction GeoTIFF
        write_geotiff(os.path.join(tif_dir, f"{model.lower()}_pred_ndvi.tif"), pred_grid, config, template_meta)

        # Compute MAE & Spatial Bias
        df[f'{model}_abs_err'] = (df['true_ndvi'] - df[f'{model}_pred_ndvi']).abs()
        df[f'{model}_bias'] = df[f'{model}_pred_ndvi'] - df['true_ndvi']

        mae_node = df.groupby('pixel_idx')[f'{model}_abs_err'].mean()
        bias_node = df.groupby('pixel_idx')[f'{model}_bias'].mean()

        mae_grid = reconstruct_spatial_grid(mae_node, height, width, valid_mask=spatial_shape_mask)
        bias_grid = reconstruct_spatial_grid(bias_node, height, width, valid_mask=spatial_shape_mask)

        mae_grids[model] = mae_grid
        bias_grids[model] = bias_grid

        # Export MAE & Bias GeoTIFFs
        write_geotiff(os.path.join(tif_dir, f"{model.lower()}_mae.tif"), mae_grid, config, template_meta)
        write_geotiff(os.path.join(tif_dir, f"{model.lower()}_bias.tif"), bias_grid, config, template_meta)

        # Individual Model Figure: Observed vs Predicted
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        axes[0].imshow(true_grid, cmap=cmap_ndvi, origin='upper', vmin=0.1, vmax=0.7)
        axes[0].set_title("Observed Mean NDVI", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        im = axes[1].imshow(pred_grid, cmap=cmap_ndvi, origin='upper', vmin=0.1, vmax=0.7)
        axes[1].set_title(f"{model.replace('Model','')} Predicted Mean NDVI", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        fig.subplots_adjust(bottom=0.15)
        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
        cbar.set_label("NDVI", fontsize=11, fontweight='bold')

        plt.savefig(os.path.join(fig_dir, f"observed_vs_{model.lower()}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Generated observed_vs_{model.lower()}.png & associated GeoTIFFs")

    # 3. COMBINED 4-MODEL MAE & BIAS FIGURES
    print("\n[3/4] Generating combined MAE & Bias comparison grids...")
    
    # MAE Grid (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=300)
    axes_flat = axes.flatten()
    for idx, model in enumerate(models_to_process):
        if model in mae_grids:
            im_mae = axes_flat[idx].imshow(mae_grids[model], cmap=cmap_mae, origin='upper', vmin=0.0, vmax=0.15)
            axes_flat[idx].set_title(f"{model.replace('Model','')} MAE", fontsize=11, fontweight='bold')
            axes_flat[idx].axis('off')

    fig.subplots_adjust(bottom=0.12)
    cbar_ax = fig.add_axes([0.2, 0.06, 0.6, 0.025])
    cbar = fig.colorbar(im_mae, cax=cbar_ax, orientation='horizontal', extend='max')
    cbar.set_label("Mean Absolute Error (NDVI Space)", fontsize=11, fontweight='bold')
    plt.savefig(os.path.join(fig_dir, "all_models_mae_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Bias Grid (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=300)
    axes_flat = axes.flatten()
    for idx, model in enumerate(models_to_process):
        if model in bias_grids:
            im_bias = axes_flat[idx].imshow(bias_grids[model], cmap=cmap_bias, origin='upper', vmin=-0.1, vmax=0.1)
            axes_flat[idx].set_title(f"{model.replace('Model','')} Spatial Bias", fontsize=11, fontweight='bold')
            axes_flat[idx].axis('off')

    fig.subplots_adjust(bottom=0.12)
    cbar_ax = fig.add_axes([0.2, 0.06, 0.6, 0.025])
    cbar = fig.colorbar(im_bias, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_label("Spatial Bias (Predicted - Observed)", fontsize=11, fontweight='bold')
    plt.savefig(os.path.join(fig_dir, "all_models_bias_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. TRAINING HISTORY LOSS CURVES
    print("\n[4/4] Generating training & validation loss curves...")
    history_files = {
        'STID': 'history_STID.csv',
        'DCRNN': 'history_DCRNN.csv',
        'GRUGCNModel': 'history_GRUGCNModel.csv',
        'GraphWaveNet': 'history_GraphWaveNet.csv'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes_flat = axes.flatten()
    has_history = False

    for idx, (model_name, hist_file) in enumerate(history_files.items()):
        hist_path = os.path.join(processed_dir, hist_file)
        ax = axes_flat[idx]

        if os.path.exists(hist_path):
            has_history = True
            h_df = pd.read_csv(hist_path)
            epochs = h_df['epoch'] if 'epoch' in h_df.columns else h_df.index + 1
            
            train_col = next((c for c in ['train_loss', 'loss', 'train_mae'] if c in h_df.columns), None)
            val_col = next((c for c in ['val_loss', 'val_mae', 'test_loss'] if c in h_df.columns), None)

            if train_col and train_col in h_df.columns:
                ax.plot(epochs, h_df[train_col], label='Train Loss', color='#1f77b4', linewidth=2)
            if val_col and val_col in h_df.columns:
                ax.plot(epochs, h_df[val_col], label='Val Loss', color='#ff7f0e', linewidth=2, linestyle='--')

            ax.set_title(f"{model_name.replace('Model','')} Loss History", fontsize=11, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Loss", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, f"No history log found\n({hist_file})", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(f"{model_name} Loss History", fontsize=11, fontweight='bold')
            ax.axis('off')

    if has_history:
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "dl_training_loss_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Generated dl_training_loss_curves.png")
    else:
        plt.close()

    print(f"\nExecution Complete!")
    print(f"All figures saved to: {fig_dir}")
    print(f"All GeoTIFFs saved to: {tif_dir}")

if __name__ == "__main__":
    run_dl_eda_visualizations()