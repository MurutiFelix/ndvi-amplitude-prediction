import os
import yaml
import numpy as np
import pandas as pd
import matplotlib

# Safe backend fallback for headless HPC cluster environments
try:
    get_ipython()
    print("Detected Jupyter environment. Interactive/Inline plotting enabled.")
except NameError:
    print("Detected headless CLI environment. Switching matplotlib to 'Agg' backend.")
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.transform import from_origin

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def reconstruct_spatial_grid(series, height, width, valid_mask=None):
    """
    Maps grouped series values back to geographical 2D spatial grid,
    applying the basin mask to keep outside boundary transparent.
    """
    grid = np.full((height * width), np.nan)
    valid_idx_mask = (series.index >= 0) & (series.index < (height * width))
    valid_series = series[valid_idx_mask]

    grid[valid_series.index.astype(int)] = valid_series.values
    grid_2d = grid.reshape((height, width))
    
    # Apply spatial study-area shape mask if available
    if valid_mask is not None:
        grid_2d[~valid_mask] = np.nan
        
    return grid_2d


def write_geotiff(filename, data, config, template_meta=None):
    """
    Writes 2D numpy array to georeferenced GeoTIFF with explicit NaN nodata mask.
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
        height, width = data.shape
        spatial_cfg = config.get('spatial', {})
        west = spatial_cfg.get('west', 36.5)
        north = spatial_cfg.get('north', 1.0)
        
        x_res = spatial_cfg.get('pixel_size_x', (37.75 - 36.5) / width)
        y_res = -abs(spatial_cfg.get('pixel_size_y', (1.0 - (-0.25)) / height))
        transform = from_origin(west, north, x_res, abs(y_res))
        
        from rasterio.crs import CRS
        crs = CRS.from_epsg(4326)
        
        with rasterio.open(
            filename, 'w', driver='GTiff', height=height, width=width, count=1,
            dtype=rasterio.float32, crs=crs, transform=transform, nodata=np.nan
        ) as dst:
            dst.write(data.astype(np.float32), 1)
            
    print(f"  -> Exported GeoTIFF: {filename}")


# ------------------------------------------------------------------
# Main Deep Learning EDA Pipeline
# ------------------------------------------------------------------

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

    static_dir = next((p for p in ["data/static", "../../data/static", "../static"] if os.path.exists(p)), ".")

    # Reference raster dimension & shape mask detection
    reference_raster_path = os.path.join(static_dir, "TWI.tif")
    height = config['spatial']['height']
    width = config['spatial']['width']
    template_meta = None
    spatial_shape_mask = None

    if os.path.exists(reference_raster_path):
        with rasterio.open(reference_raster_path) as src:
            height, width = src.height, src.width
            template_meta = src.meta.copy()
            twi_data = src.read(1)
            # Define shape boundary mask from valid raster cells
            spatial_shape_mask = ~np.isnan(twi_data) & (twi_data > -9000)

    fig_dir = os.path.join(processed_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ==================================================================
    # 1. Combined Continuous Loss Plot
    # ==================================================================
    print("\n[1/5] Processing Model Training Histories...")
    dl_models = ['STID', 'DCRNN', 'GRUGCNModel', 'GraphWaveNet']
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sharp_colors = {
        'STID': '#0055FF',         # Vivid Royal Blue
        'DCRNN': '#FF5500',        # Sharp Orange-Red
        'GRUGCNModel': '#009933',   # Rich Emerald Green
        'GraphWaveNet': '#9900CC'  # Bold Deep Purple
    }
    
    found_any_history = False
    for model_name in dl_models:
        hist_file = os.path.join(processed_dir, f"history_{model_name}.csv")
        if os.path.exists(hist_file):
            found_any_history = True
            h_df = pd.read_csv(hist_file)
            color = sharp_colors.get(model_name, '#111111')
            epochs = h_df['epoch'] if 'epoch' in h_df.columns else range(1, len(h_df) + 1)
            
            train_col = next((c for c in ['train_loss', 'loss'] if c in h_df.columns), None)
            val_col = next((c for c in ['val_loss', 'validation_loss'] if c in h_df.columns), None)
            display_name = model_name.replace('Model', '')
            
            if train_col:
                ax.plot(epochs, h_df[train_col], label=f'{display_name} (Train)', 
                        color=color, linestyle='-', alpha=0.45, linewidth=1.8)
            if val_col:
                ax.plot(epochs, h_df[val_col], label=f'{display_name} (Val)', 
                        color=color, linestyle='-', linewidth=2.8)

    if found_any_history:
        ax.set_title("Deep Learning Architecture Convergence (Loss Curves)", fontsize=13, pad=15, fontweight='bold')
        ax.set_xlabel("Epochs", fontsize=11, fontweight='bold')
        ax.set_ylabel("Loss", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, frameon=True)
        plt.savefig(os.path.join(fig_dir, "dl_combined_loss_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # ==================================================================
    # 2. Check and Load Residuals Dataframe
    # ==================================================================
    dl_residuals_csv = os.path.join(processed_dir, "dl_test_residuals_dataframe.csv")
    if not os.path.exists(dl_residuals_csv):
        print(f"\n[Notice] {dl_residuals_csv} not found.")
        return

    df = pd.read_csv(dl_residuals_csv)

    num_spatial_nodes = height * width
    if 'pixel_idx' in df.columns:
        df['pixel_idx'] = df['pixel_idx'] % num_spatial_nodes
    elif 'node_idx' in df.columns:
        df['pixel_idx'] = df['node_idx'] % num_spatial_nodes
    else:
        df['pixel_idx'] = df.index % num_spatial_nodes

    models_to_process = ['STID', 'DCRNN', 'GRUGCN', 'GraphWaveNet']

    # Set background color for transparent masked boundary
    cmap_mae = plt.cm.YlOrRd.copy()
    cmap_mae.set_bad(color='none')
    
    cmap_bias = plt.cm.RdBu_r.copy()
    cmap_bias.set_bad(color='none')
    
    cmap_ndvi = plt.cm.YlGn.copy()
    cmap_ndvi.set_bad(color='none')

    # ==================================================================
    # 3. Spatial MAE Maps
    # ==================================================================
    print("\n[2/5] Generating Spatial MAE Maps...")
    for model in models_to_process:
        pred_col = f'{model}_pred' if f'{model}_pred' in df.columns else f'{model}Model_pred'
        if pred_col not in df.columns:
            continue
            
        df[f'{model}_abs_residual'] = (df['true_log_ndvi'] - df[pred_col]).abs()
        mae_per_node = df.groupby('pixel_idx')[f'{model}_abs_residual'].mean()
        mae_grid = reconstruct_spatial_grid(mae_per_node, height, width, valid_mask=spatial_shape_mask)

        write_geotiff(os.path.join(processed_dir, f"{model.lower()}_mae.tif"), mae_grid, config, template_meta)

        plt.figure(figsize=(9, 8), dpi=300)
        im = plt.imshow(mae_grid, cmap=cmap_mae, origin='upper')
        plt.title(f"{model} Mean Absolute Error (MAE) over UENRB", fontsize=13, pad=15)
        cbar = plt.colorbar(im, orientation='horizontal', shrink=0.75, pad=0.08, extend='max')
        cbar.set_label("Mean Absolute Error (log NDVI)", fontsize=10, labelpad=5)
        plt.axis('off')
        plt.savefig(os.path.join(fig_dir, f"{model.lower()}_mae_spatial_map.png"), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

    # ==================================================================
    # 4. Spatial Residual Bias Maps
    # ==================================================================
    print("\n[3/5] Generating Spatial Residual Bias Maps...")
    for model in models_to_process:
        pred_col = f'{model}_pred' if f'{model}_pred' in df.columns else f'{model}Model_pred'
        if pred_col not in df.columns:
            continue

        df[f'{model}_residual'] = df['true_log_ndvi'] - df[pred_col]
        bias_per_node = df.groupby('pixel_idx')[f'{model}_residual'].mean()
        bias_grid = reconstruct_spatial_grid(bias_per_node, height, width, valid_mask=spatial_shape_mask)

        write_geotiff(os.path.join(processed_dir, f"{model.lower()}_spatial_bias_residuals.tif"), bias_grid, config, template_meta)

        norm = mcolors.CenteredNorm(vcenter=0.0)
        plt.figure(figsize=(10, 8), dpi=300)
        im = plt.imshow(bias_grid, cmap=cmap_bias, norm=norm, origin='upper')
        plt.title(f"{model} Mean Spatial Bias (Actual - Predicted)", fontsize=13, pad=15)
        cbar = plt.colorbar(im, orientation='horizontal', shrink=0.7, pad=0.08, extend='both')
        cbar.set_label("Overprediction (Blue)  <--- 0 --->  Underprediction (Red)", fontsize=10, labelpad=8)
        plt.axis('off')
        plt.savefig(os.path.join(fig_dir, f"{model.lower()}_bias_spatial_map.png"), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

    # ==================================================================
    # 5. Observed vs Predicted NDVI Maps
    # ==================================================================
    print("\n[4/5] Generating Predicted NDVI Maps...")
    df['true_ndvi'] = np.exp(df['true_log_ndvi'])
    mean_true = df.groupby('pixel_idx')['true_ndvi'].mean()
    true_grid = reconstruct_spatial_grid(mean_true, height, width, valid_mask=spatial_shape_mask)
    write_geotiff(os.path.join(processed_dir, "dl_observed_mean_ndvi.tif"), true_grid, config, template_meta)

    for model in models_to_process:
        pred_col = f'{model}_pred' if f'{model}_pred' in df.columns else f'{model}Model_pred'
        if pred_col not in df.columns:
            continue

        df[f'{model}_pred_ndvi'] = np.exp(df[pred_col])
        mean_pred = df.groupby('pixel_idx')[f'{model}_pred_ndvi'].mean()
        pred_grid = reconstruct_spatial_grid(mean_pred, height, width, valid_mask=spatial_shape_mask)

        write_geotiff(os.path.join(processed_dir, f"{model.lower()}_predicted_mean_ndvi.tif"), pred_grid, config, template_meta)

        fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=300)
        axes[0].imshow(true_grid, cmap=cmap_ndvi, origin='upper', vmin=0.1, vmax=0.7)
        axes[0].set_title("Observed Mean NDVI", fontsize=12)
        axes[0].axis('off')

        im = axes[1].imshow(pred_grid, cmap=cmap_ndvi, origin='upper', vmin=0.1, vmax=0.7)
        axes[1].set_title(f"{model} Predicted Mean NDVI", fontsize=12)
        axes[1].axis('off')

        fig.subplots_adjust(bottom=0.15)
        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
        cbar.set_label("NDVI", fontsize=11)
        plt.savefig(os.path.join(fig_dir, f"observed_vs_{model.lower()}_ndvi_comparison.png"), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

    # ==================================================================
    # 6. TWI Hotspot Map
    # ==================================================================
    print("\n[5/5] Generating TWI Hotspot Map...")
    target_model_col = next((c for c in df.columns if c.endswith('_abs_residual')), None)
    if target_model_col:
        mae_node = df.groupby('pixel_idx')[target_model_col].mean()
        high_err = (mae_node >= np.nanpercentile(mae_node.values, 90)).astype(float)
        high_err[high_err == 0.0] = np.nan
        err_grid = reconstruct_spatial_grid(high_err, height, width, valid_mask=spatial_shape_mask)

        twi_file = os.path.join(static_dir, "TWI.tif")
        if os.path.exists(twi_file):
            with rasterio.open(twi_file) as src:
                twi_raw = src.read(1)
                twi_raw[twi_raw <= -9000] = np.nan
                
                plt.figure(figsize=(9, 8), dpi=300)
                plt.imshow(twi_raw, cmap='bone', origin='upper')
                plt.imshow(err_grid, cmap='autumn_r', alpha=0.7, origin='upper')
                plt.title("TWI overlaid with Top 10% DL MAE Hotspots", fontsize=13, pad=15)
                plt.axis('off')
                plt.savefig(os.path.join(fig_dir, "twi_with_dl_error_hotspots.png"), dpi=300, bbox_inches='tight', transparent=True)
                plt.close()

    print(f"\nExecution finished cleanly! Masked spatial outputs saved in: {fig_dir}")


if __name__ == "__main__":
    run_dl_eda_visualizations()