# src/data/dataset.py
"""
Two responsibilities:
1. build_tabular_dataset() — compiles rasters to CSV (includes pixel_idx column)
2. NDVIGraphDataset        — PyTorch Dataset for TSL graph models
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import torch
import rioxarray
from scipy.ndimage import uniform_filter
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ------------------------------------------------------------------ #
#  Raster compiler  (writes pixel_idx so spatial placement is exact)  #
# ------------------------------------------------------------------ #

def align_raster(raster_path, template_path):
    from src.data.raster_processor import align_raster as _align
    return _align(raster_path, template_path)


def build_tabular_dataset(config):
    """
    Compiles monthly GeoTIFF rasters into a tabular CSV.
    Stores pixel_idx (flattened grid position) so NDVIGraphDataset
    can place each row back into the correct spatial node.
    """
    raw_dir       = config['paths']['raw_dir']
    template_path = config['paths']['ndvi_template']

    print("Loading static landscape variables (TWI and Soil)...")
    twi_flat  = rioxarray.open_rasterio(config['paths']['twi']).squeeze().values.flatten()
    soil_flat = rioxarray.open_rasterio(config['paths']['soil_raster']).squeeze().values.flatten()

    ndvi_files   = sorted(glob.glob(os.path.join(raw_dir, "NDVI_*.tif")))
    lst_files    = sorted(glob.glob(os.path.join(raw_dir, "LST_*.tif")))
    precip_files = sorted(glob.glob(os.path.join(raw_dir, "precipitation_*.tif")))

    if not (len(ndvi_files) == len(lst_files) == len(precip_files)):
        print(f"Warning: file count mismatch — NDVI:{len(ndvi_files)} "
              f"LST:{len(lst_files)} Precip:{len(precip_files)}")

    all_rows = []

    print("Compiling space-time matrix...")
    for i in tqdm(range(3, len(ndvi_files)), desc="Timesteps"):
        ndvi_filename = os.path.basename(ndvi_files[i])
        match = re.search(r"(\d{4})_(\d{2})", ndvi_filename)
        if not match:
            continue

        year  = int(match.group(1))
        month = int(match.group(2))

        ndvi_t_2d    = rioxarray.open_rasterio(ndvi_files[i]).squeeze().values
        ndvi_prev_2d = rioxarray.open_rasterio(ndvi_files[i - 1]).squeeze().values
        ndvi_spatial_lag_2d = uniform_filter(
            np.where(ndvi_prev_2d > 0, ndvi_prev_2d, np.nan),
            size=3, mode='nearest'
        )
        ndvi_t           = ndvi_t_2d.flatten()
        ndvi_spatial_lag = ndvi_spatial_lag_2d.flatten()

        from src.data.raster_processor import align_raster
        lst_minus1 = align_raster(lst_files[i - 1], template_path).values.flatten()
        lst_minus2 = align_raster(lst_files[i - 2], template_path).values.flatten()
        lst_minus3 = align_raster(lst_files[i - 3], template_path).values.flatten()

        precip_minus1 = align_raster(precip_files[i - 1], template_path).values.flatten()
        precip_minus2 = align_raster(precip_files[i - 2], template_path).values.flatten()
        precip_minus3 = align_raster(precip_files[i - 3], template_path).values.flatten()

        for arr in [precip_minus1, precip_minus2, precip_minus3]:
            arr[:] = np.where(arr < 0, np.nan, arr)

        pop_path = os.path.join(raw_dir, f"Pop_Density_{year}.tif")
        pop_flat = align_raster(pop_path, template_path).values.flatten() \
            if os.path.exists(pop_path) \
            else np.full_like(ndvi_t, np.nan, dtype=np.float64)

        log_ndvi     = np.where((ndvi_t > 0) & ~np.isnan(ndvi_t), np.log(ndvi_t), np.nan)
        log_precip_1 = np.where((precip_minus1 >= 0) & ~np.isnan(precip_minus1), np.log(precip_minus1 + 1), np.nan)
        log_precip_2 = np.where((precip_minus2 >= 0) & ~np.isnan(precip_minus2), np.log(precip_minus2 + 1), np.nan)
        log_precip_3 = np.where((precip_minus3 >= 0) & ~np.isnan(precip_minus3), np.log(precip_minus3 + 1), np.nan)

        valid = (
            (ndvi_t > 0)             & ~np.isnan(ndvi_t)  &
            ~np.isnan(lst_minus1)    & ~np.isnan(lst_minus2)    & ~np.isnan(lst_minus3)    &
            ~np.isnan(precip_minus1) & ~np.isnan(precip_minus2) & ~np.isnan(precip_minus3) &
            ~np.isnan(ndvi_spatial_lag) &
            ~np.isnan(twi_flat)      & ~np.isnan(soil_flat)      & ~np.isnan(pop_flat)
        )

        n_valid = valid.sum()
        if n_valid == 0:
            continue

        # pixel_idx stores the exact flattened grid position of each valid pixel
        pixel_idx = np.where(valid)[0]

        block = pd.DataFrame({
            'year'                   : np.full(n_valid, year,  dtype=np.int32),
            'month'                  : np.full(n_valid, month, dtype=np.int32),
            'pixel_idx'              : pixel_idx,                        # ← KEY
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

    if not all_rows:
        print("[ERROR]: No valid data blocks compiled.")
        return pd.DataFrame()

    print(f"\nConcatenating {len(all_rows)} timestep blocks...")
    df = pd.concat(all_rows, ignore_index=True)
    print(f"Final dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ------------------------------------------------------------------ #
#  PyTorch Dataset for TSL graph models                               #
# ------------------------------------------------------------------ #

class NDVIGraphDataset(Dataset):
    """
    PyTorch Dataset for spatiotemporal NDVI prediction using TSL graph models.

    Reads tabular_dataset.csv (must contain pixel_idx column) and
    reconstructs 3D spatial arrays [T, N, F] with exact pixel placement.

    Input structure per sample:
        x : [window_size, n_nodes, n_dynamic_features]
        u : [n_nodes, n_static_features]
        y : [n_nodes, 1]  — target log_ndvi at t
    """

    DYNAMIC_COLS = [
        'log_ndvi',
        'lst_driver_lag1', 'lst_driver_lag2', 'lst_driver_lag3',
        'log_precip_driver_lag1', 'log_precip_driver_lag2', 'log_precip_driver_lag3',
        'ndvi_spatial_lag',
        'pop_density',
    ]

    def __init__(
        self,
        csv_path   : str,
        height     : int             = 159,
        width      : int             = 181,
        window_size: int             = 12,
        split      : str             = 'train',
        split_year : int             = 2021,
        scaler     : StandardScaler  = None,
    ):
        self.height      = height
        self.width       = width
        self.n_nodes     = height * width
        self.window_size = window_size
        self.split       = split
        self.split_year  = split_year

        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Validate pixel_idx column exists
        if 'pixel_idx' not in df.columns:
            raise ValueError(
                "tabular_dataset.csv is missing the 'pixel_idx' column. "
                "Please recompile by deleting the CSV and rerunning build_tabular_dataset()."
            )

        # --- One-hot encode soil ---
        df = pd.get_dummies(df, columns=['soil_snum'], drop_first=True)
        soil_cols = [c for c in df.columns if c.startswith('soil_snum_')]
        self.static_cols = ['twi'] + soil_cols

        # --- Sort chronologically ---
        df = df.sort_values(['year', 'month', 'pixel_idx']).reset_index(drop=True)

        # --- Unique timesteps ---
        timesteps = df[['year', 'month']].drop_duplicates().sort_values(
            ['year', 'month']
        ).reset_index(drop=True)
        self.timesteps = timesteps

        # --- Build 3D arrays [T, N, F] using pixel_idx for exact placement ---
        print("Pivoting tabular data to spatial grid format...")
        T         = len(timesteps)
        n_dynamic = len(self.DYNAMIC_COLS)
        n_static  = len(self.static_cols)

        dynamic_array = np.full((T, self.n_nodes, n_dynamic), np.nan, dtype=np.float32)
        static_array  = np.full((self.n_nodes, n_static),     np.nan, dtype=np.float32)
        static_filled = False

        for t_idx, (_, row) in enumerate(timesteps.iterrows()):
            mask = (df['year'] == row['year']) & (df['month'] == row['month'])
            sub  = df[mask]
            if len(sub) == 0:
                continue

            pix = sub['pixel_idx'].values.astype(int)   # exact grid positions

            for f_idx, col in enumerate(self.DYNAMIC_COLS):
                if col in sub.columns:
                    dynamic_array[t_idx, pix, f_idx] = sub[col].values.astype(np.float32)

            if not static_filled:
                for f_idx, col in enumerate(self.static_cols):
                    if col in sub.columns:
                        static_array[pix, f_idx] = sub[col].values.astype(np.float32)
                static_filled = True

        # --- Chronological split ---
        train_mask = (timesteps['year'] <= split_year).values
        test_mask  = (timesteps['year'] >  split_year).values
        valid_t    = np.where(train_mask if split == 'train' else test_mask)[0]

        # --- Sliding windows ---
        self.windows = []
        for t in valid_t:
            start = t - window_size
            if start < 0:
                continue
            self.windows.append((start, t))

        print(f"  Split '{split}': {len(self.windows)} windows "
              f"from {len(valid_t)} valid timesteps")

        # --- Normalize dynamic features (fit on train only) ---
        train_t_indices = np.where(train_mask)[0]
        train_dynamic   = dynamic_array[train_t_indices]
        flat_train      = train_dynamic.reshape(-1, n_dynamic)
        valid_rows      = ~np.isnan(flat_train).any(axis=1)

        print(f"  Valid training rows for scaler: {valid_rows.sum():,}")

        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(flat_train[valid_rows])
        else:
            self.scaler = scaler

        flat_all    = dynamic_array.reshape(-1, n_dynamic)
        valid_all   = ~np.isnan(flat_all).any(axis=1)
        flat_scaled = flat_all.copy()
        flat_scaled[valid_all] = self.scaler.transform(
            flat_all[valid_all]
        ).astype(np.float32)
        self.dynamic_array = flat_scaled.reshape(T, self.n_nodes, n_dynamic)

        # --- Normalize static features ---
        static_scaler = StandardScaler()
        valid_static  = ~np.isnan(static_array).any(axis=1)
        static_scaled = static_array.copy()
        if valid_static.sum() > 0:
            static_scaled[valid_static] = static_scaler.fit_transform(
                static_array[valid_static]
            ).astype(np.float32)
        self.static_array = static_scaled

        # --- Fill remaining NaNs (masked pixels) with 0 ---
        self.dynamic_array = np.nan_to_num(self.dynamic_array, nan=0.0)
        self.static_array  = np.nan_to_num(self.static_array,  nan=0.0)

        self.n_dynamic_features = n_dynamic
        self.n_static_features  = n_static

        print(f"  Dynamic features : {n_dynamic}")
        print(f"  Static features  : {n_static}")
        print(f"  Nodes            : {self.n_nodes:,}")
        print(f"  Window size      : {window_size} months")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, t = self.windows[idx]

        x = torch.tensor(self.dynamic_array[start:t],    dtype=torch.float32)
        u = torch.tensor(self.static_array,               dtype=torch.float32)
        y = torch.tensor(self.dynamic_array[t, :, 0:1],  dtype=torch.float32)

        return x, u, y


# ------------------------------------------------------------------ #
#  Convenience builder                                                 #
# ------------------------------------------------------------------ #

def build_datasets(config, window_size=12):
    """
    Build train and test NDVIGraphDataset instances sharing the same scaler.
    If tabular_dataset.csv lacks pixel_idx, recompiles it first.
    """
    csv_path   = os.path.join(config['paths']['processed_dir'], "tabular_dataset.csv")
    split_year = config['features']['train_split_year']
    height     = config['spatial']['height']
    width      = config['spatial']['width']

    # --- Auto-recompile if pixel_idx missing ---
    if os.path.exists(csv_path):
        probe = pd.read_csv(csv_path, nrows=1)
        if 'pixel_idx' not in probe.columns:
            print("tabular_dataset.csv missing pixel_idx — recompiling...")
            os.remove(csv_path)

    if not os.path.exists(csv_path):
        df = build_tabular_dataset(config)
        if df.empty:
            raise RuntimeError("Dataset compilation failed — check raster paths.")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")

    train_dataset = NDVIGraphDataset(
        csv_path    = csv_path,
        height      = height,
        width       = width,
        window_size = window_size,
        split       = 'train',
        split_year  = split_year,
    )

    test_dataset = NDVIGraphDataset(
        csv_path    = csv_path,
        height      = height,
        width       = width,
        window_size = window_size,
        split       = 'test',
        split_year  = split_year,
        scaler      = train_dataset.scaler,
    )

    return train_dataset, test_dataset