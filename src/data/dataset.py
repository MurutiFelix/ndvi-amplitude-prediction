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
        
        # Pull annual population density based on the target year (Pop_Density_YYYY.tif)
        pop_path = os.path.join(raw_dir, f"Pop_Density_{year}.tif")
        if os.path.exists(pop_path):
            pop_flat = align_raster(pop_path, template_path).values.flatten()
        else:
            pop_flat = np.full_like(ndvi_t, np.nan)
            
        # Mathematical transformations: log(y) and log(x + 1)
        log_ndvi = np.log(ndvi_t)
        log_precip = np.log(precip_minus1 + 1)
        
        # Extract active spatial values, skipping water masks and null regions
        for idx in range(len(ndvi_t)):
            if np.isnan(ndvi_t[idx]) or ndvi_t[idx] <= 0 or np.isnan(lst_minus1[idx]) or np.isnan(precip_minus1[idx]):
                continue
                
            all_rows.append({
                'year': year,
                'month': month,
                'log_ndvi': log_ndvi[idx],
                'lst_driver_lag1': lst_minus1[idx],
                'log_precip_driver_lag1': log_precip[idx],
                'pop_density': pop_flat[idx],
                'twi': twi_flat[idx],
                'soil_snum': soil_flat[idx]
            })
            
    return pd.DataFrame(all_rows)