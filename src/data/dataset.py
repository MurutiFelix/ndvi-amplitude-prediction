# src/data/dataset.py
import glob
import os
import pandas as pd
import numpy as np
import rioxarray
from src.data.raster_processor import process_soil_from_zip, process_precipitation_nc

def build_tabular_dataset(config):
    """Loops through all timesteps, aligns layers, applies lags, and flattens to 2D table."""
    template_path = config['paths']['ndvi_template']
    
    # 1. Load static layers
    twi = rioxarray.open_rasterio(config['paths']['twi']).squeeze().values.flatten()
    soil = process_soil_from_zip(config['paths']['soil_zip'], template_path).values.flatten()
    
    # 2. Process NetCDF precipitation stack
    precip_cube = process_precipitation_nc(config['paths']['precip_nc'], template_path)
    
    # 3. Collect lists of all your monthly NDVI and LST paths (ordered chronologically)
    ndvi_files = sorted(glob.glob(os.path.join(config['paths']['raw_dir'], "NDVI_*.tif")))
    lst_files = sorted(glob.glob(os.path.join(config['paths']['raw_dir'], "LST_*.tif")))
    
    all_rows = []
    
    # 4. Step through time, applying a 1-month lag (Target t pairs with drivers t-1)
    for i in range(1, len(ndvi_files)):
        # Target month t
        ndvi_t = rioxarray.open_rasterio(ndvi_files[i]).squeeze().values.flatten()
        
        # Drivers month t-1
        lst_minus1 = rioxarray.open_rasterio(lst_files[i-1]).squeeze().values.flatten()
        precip_minus1 = precip_cube.isel(time=i-1).values.flatten()
        
        # Extract corresponding year for population matching
        # (Assuming file format contains year e.g., 'NDVI_2006_02.tif')
        year = int(os.path.basename(ndvi_files[i]).split('_')[1])
        
        # Apply literature transformations: log(y) and log(x + 1)
        log_ndvi = np.log(ndvi_t)
        log_precip = np.log(precip_minus1 + 1)
        
        # Build 2D matrix rows for every pixel in this time step
        for pixel_idx in range(len(ndvi_t)):
            # Skip water/NoData masks using NDVI validity bounds
            if np.isnan(ndvi_t[pixel_idx]) or ndvi_t[pixel_idx] <= 0:
                continue
                
            all_rows.append({
                'year': year,
                'month': i,
                'log_ndvi': log_ndvi[pixel_idx],
                'lst_driver': lst_minus1[pixel_idx],
                'log_precip_driver': log_precip[pixel_idx],
                'twi': twi[pixel_idx],
                'soil_snum': soil[pixel_idx]
            })
            
    return pd.DataFrame(all_rows)