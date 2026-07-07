# src/data/dataset.py
import glob
import os
import re
import pandas as pd
import numpy as np
import rioxarray
from scipy.ndimage import uniform_filter
from tqdm import tqdm
from src.data.raster_processor import align_raster

def build_tabular_dataset(config):
    """
    Compiles monthly spatial data into a unified 2D DataFrame
    where all inputs are matched GeoTIFF raster layers.

    Temporal logic:
        - Target variable  : NDVI at time t
        - Dynamic drivers  : LST and Precipitation at t-1, t-2, t-3 (3-month lag memory)
        - Spatial lag      : 3x3 neighbourhood mean of NDVI at t-1
        - Annual driver    : Population Density for the year of t (dynamic, annual)
        - Static variables : TWI and Soil (time-invariant landscape properties)

    Performance:
        - Pixel-level iteration replaced with fully vectorized NumPy masking.
        - Each timestep produces a DataFrame block; all blocks concatenated once.
    """
    raw_dir       = config['paths']['raw_dir']
    template_path = config['paths']['ndvi_template']

    # --- Load static landscape variables once — shared across all timesteps
    
    print("Loading static landscape variables (TWI and Soil)...")
    twi_flat  = rioxarray.open_rasterio(config['paths']['twi']).squeeze().values.flatten()
    soil_flat = rioxarray.open_rasterio(config['paths']['soil_raster']).squeeze().values.flatten()

    # Gather and sort all monthly raster file lists
    
    ndvi_files   = sorted(glob.glob(os.path.join(raw_dir, "NDVI_*.tif")))
    lst_files    = sorted(glob.glob(os.path.join(raw_dir, "LST_*.tif")))
    precip_files = sorted(glob.glob(os.path.join(raw_dir, "precipitation_*.tif")))

    if not (len(ndvi_files) == len(lst_files) == len(precip_files)):
        print(
            f"Warning: Temporal asset count mismatch! "
            f"NDVI: {len(ndvi_files)}, LST: {len(lst_files)}, "
            f"Precip: {len(precip_files)}"
        )

    
    # Temporal loop — index starts at 3 to allow t-1, t-2, t-3 lags
   
    all_rows = []

    print("Compiling space-time matrix with vectorized pixel masking...")
    for i in tqdm(range(3, len(ndvi_files)), desc="Compiling Timesteps"):

        # --- Parse year and month from NDVI filename ---
        ndvi_filename = os.path.basename(ndvi_files[i])
        match = re.search(r"(\d{4})_(\d{2})", ndvi_filename)
        if not match:
            continue

        year  = int(match.group(1))
        month = int(match.group(2))

        # --- Read target NDVI at time t (keep 2D for spatial lag) ---
        ndvi_t_2d = rioxarray.open_rasterio(ndvi_files[i]).squeeze().values

        # --- Spatial lag: 3x3 neighbourhood mean of NDVI at t-1 ---
        ndvi_prev_2d = rioxarray.open_rasterio(ndvi_files[i - 1]).squeeze().values
        ndvi_spatial_lag_2d = uniform_filter(
            np.where(ndvi_prev_2d > 0, ndvi_prev_2d, np.nan),
            size=3,
            mode='nearest'
        )

        # Flatten both for tabular structure
        ndvi_t           = ndvi_t_2d.flatten()
        ndvi_spatial_lag = ndvi_spatial_lag_2d.flatten()

        # --- Read dynamic drivers at t-1, t-2, t-3 lags ---
        lst_minus1 = align_raster(lst_files[i - 1], template_path).values.flatten()
        lst_minus2 = align_raster(lst_files[i - 2], template_path).values.flatten()
        lst_minus3 = align_raster(lst_files[i - 3], template_path).values.flatten()

        precip_minus1 = align_raster(precip_files[i - 1], template_path).values.flatten()
        precip_minus2 = align_raster(precip_files[i - 2], template_path).values.flatten()
        precip_minus3 = align_raster(precip_files[i - 3], template_path).values.flatten()

        # Guard against GEE background/negative NoData in precipitation
        precip_minus1 = np.where(precip_minus1 < 0, np.nan, precip_minus1)
        precip_minus2 = np.where(precip_minus2 < 0, np.nan, precip_minus2)
        precip_minus3 = np.where(precip_minus3 < 0, np.nan, precip_minus3)

        # --- Load annual population density dynamically for target year ---
        pop_path = os.path.join(raw_dir, f"Pop_Density_{year}.tif")
        if os.path.exists(pop_path):
            pop_flat = align_raster(pop_path, template_path).values.flatten()
        else:
            print(f"  Warning: Pop_Density_{year}.tif not found — filling with NaN.")
            pop_flat = np.full_like(ndvi_t, np.nan, dtype=np.float64)

        # ---log transforms (vectorized, NaN-preserving) ---
        log_ndvi = np.where(
            (ndvi_t > 0) & ~np.isnan(ndvi_t),
            np.log(ndvi_t),
            np.nan
        )
        log_precip_1 = np.where(
            (precip_minus1 >= 0) & ~np.isnan(precip_minus1),
            np.log(precip_minus1 + 1),
            np.nan
        )
        log_precip_2 = np.where(
            (precip_minus2 >= 0) & ~np.isnan(precip_minus2),
            np.log(precip_minus2 + 1),
            np.nan
        )
        log_precip_3 = np.where(
            (precip_minus3 >= 0) & ~np.isnan(precip_minus3),
            np.log(precip_minus3 + 1),
            np.nan
        )

        # --- Vectorized validity mask across ALL variables simultaneously ---
        valid = (
            (ndvi_t > 0)             & ~np.isnan(ndvi_t)       &
            ~np.isnan(lst_minus1)                               &
            ~np.isnan(lst_minus2)                               &
            ~np.isnan(lst_minus3)                               &
            ~np.isnan(precip_minus1)                            &
            ~np.isnan(precip_minus2)                            &
            ~np.isnan(precip_minus3)                            &
            ~np.isnan(ndvi_spatial_lag)                         &
            ~np.isnan(twi_flat)                                 &
            ~np.isnan(soil_flat)                                &
            ~np.isnan(pop_flat)
        )

        n_valid = valid.sum()
        if n_valid == 0:
            print(f"  Warning: No valid pixels for {year}-{month:02d}. Skipping.")
            continue

        # --- Build timestep block directly from masked arrays ---
        block = pd.DataFrame({
            'year'                   : np.full(n_valid, year,  dtype=np.int32),
            'month'                  : np.full(n_valid, month, dtype=np.int32),
            'log_ndvi'               : log_ndvi[valid],
            'lst_driver_lag1'        : lst_minus1[valid],
            'lst_driver_lag2'        : lst_minus2[valid],
            'lst_driver_lag3'        : lst_minus3[valid],
            'log_precip_driver_lag1' : log_precip_1[valid],
            'log_precip_driver_lag2' : log_precip_2[valid],
            'log_precip_driver_lag3' : log_precip_3[valid],
            'ndvi_spatial_lag'       : ndvi_spatial_lag[valid],
            'pop_density'            : pop_flat[valid],
            'twi'                    : twi_flat[valid],
            'soil_snum'              : soil_flat[valid],
        })

        all_rows.append(block)

    
    # Single concatenation of all timestep blocks
    
    if not all_rows:
        print("[ERROR]: No valid data blocks compiled. Check raster spatial intersections.")
        return pd.DataFrame()

    print(f"\nConcatenating {len(all_rows)} timestep blocks...")
    df = pd.concat(all_rows, ignore_index=True)
    print(f"Final dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df