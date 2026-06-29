# src/data/dataset.py
import glob
import os
import re
import pandas as pd
import numpy as np
import rioxarray
from tqdm import tqdm
from src.data.raster_processor import align_raster

def build_tabular_dataset(config):
    """
    Compiles monthly spatial data into a unified 2D DataFrame 
    where all inputs are matched GeoTIFF raster layers.
    """
    raw_dir = config['paths']['raw_dir']
    template_path = config['paths']['ndvi_template']
    
    print("Loading static landscape variables (TWI and Soil)...")
    twi_flat = rioxarray.open_rasterio(config['paths']['twi']).squeeze().values.flatten()
    soil_flat = rioxarray.open_rasterio(config['paths']['soil_raster']).squeeze().values.flatten()
    
    # Gather all sorted monthly response and driver sets
    ndvi_files = sorted(glob.glob(os.path.join(raw_dir, "NDVI_*.tif")))
    lst_files = sorted(glob.glob(os.path.join(raw_dir, "LST_*.tif")))
    precip_files = sorted(glob.glob(os.path.join(raw_dir, "precipitation_*.tif")))
    
    if not (len(ndvi_files) == len(lst_files) == len(precip_files)):
        print(f"Warning: Temporal asset count mismatch! NDVI: {len(ndvi_files)}, LST: {len(lst_files)}, Precip: {len(precip_files)}")
        
    all_rows = []
    
    print("Flattening and indexing space-time columns...")
    # Time-Lag Loop: Target month t pairs with drivers from month t-1
    for i in tqdm(range(1, len(ndvi_files)), desc="Compiling Timesteps"):
        ndvi_filename = os.path.basename(ndvi_files[i])
        match = re.search(r"(\d{4})_(\d{2})", ndvi_filename)
        if not match:
            continue
            
        year = int(match.group(1))
        month = int(match.group(2))
        
        # Read Target month values (t)
        ndvi_t = rioxarray.open_rasterio(ndvi_files[i]).squeeze().values.flatten()
        
        # Read Driver values at lagged interval (t-1)
        lst_minus1 = align_raster(lst_files[i-1], template_path).values.flatten()
        precip_minus1 = align_raster(precip_files[i-1], template_path).values.flatten()
        
        # --- Guard against GEE background/negative NoData masks ---
        precip_minus1 = np.where(precip_minus1 < 0, np.nan, precip_minus1)
        
        # Pull annual population density based on the target year (Pop_Density_YYYY.tif)
        pop_path = os.path.join(raw_dir, f"Pop_Density_{year}.tif")
        if os.path.exists(pop_path):
            pop_flat = align_raster(pop_path, template_path).values.flatten()
        else:
            pop_flat = np.full_like(ndvi_t, np.nan)
            
        # --- Safe Mathematical transformations ---
        # Initialize empty arrays for logs to preserve original structural array dimensions
        log_ndvi = np.full_like(ndvi_t, np.nan)
        log_precip = np.full_like(precip_minus1, np.nan)
        
        # Vectorized mask calculation: evaluate only where values are positive and real
        valid_ndvi_mask = (ndvi_t > 0) & (~np.isnan(ndvi_t))
        valid_precip_mask = (precip_minus1 >= 0) & (~np.isnan(precip_minus1))
        
        log_ndvi[valid_ndvi_mask] = np.log(ndvi_t[valid_ndvi_mask])
        log_precip[valid_precip_mask] = np.log(precip_minus1[valid_precip_mask] + 1)
        
        # Extract active spatial values, skipping water masks, clouds, and null regions
        # --- Vectorized validity mask across ALL variables simultaneously ---
        valid = (
            (ndvi_t > 0)         & ~np.isnan(ndvi_t)      &
            ~np.isnan(lst_minus1)                          &
            ~np.isnan(precip_minus1)                       &
            ~np.isnan(twi_flat)                            &
            ~np.isnan(soil_flat)                           &
            ~np.isnan(pop_flat)
        )

        n_valid = valid.sum()
        if n_valid == 0:
            print(f"  Warning: No valid pixels for {year}-{month:02d}. Skipping.")
            continue

        all_rows.append(pd.DataFrame({
            'year'                   : np.full(n_valid, year,  dtype=np.int32),
            'month'                  : np.full(n_valid, month, dtype=np.int32),
            'log_ndvi'               : log_ndvi[valid],
            'lst_driver_lag1'        : lst_minus1[valid],
            'log_precip_driver_lag1' : log_precip[valid],
            'pop_density'            : pop_flat[valid],
            'twi'                    : twi_flat[valid],
            'soil_snum'              : soil_flat[valid],
        }))
            
    if not all_rows:
        print("[ERROR]: No valid data blocks compiled. Check raster spatial intersections.")
        return pd.DataFrame()

    print(f"\nConcatenating {len(all_rows)} timestep blocks...")
    df = pd.concat(all_rows, ignore_index=True)
    print(f"Final dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df