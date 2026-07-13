# src/data/dataset.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class NDVIGraphDataset(Dataset):
    """
    PyTorch Dataset for spatiotemporal NDVI prediction using TSL graph models.

    Converts the compiled tabular_dataset.csv into sliding window sequences
    over the spatial graph of 159x181 pixels (28,779 nodes).

    Input structure per sample:
        x : [window_size, n_nodes, n_dynamic_features]  — dynamic feature sequence
        u : [n_nodes, n_static_features]                — static features per node
        y : [n_nodes, 1]                                — target log_ndvi at t+1

    Temporal logic:
        - Window of 12 consecutive months as input sequence
        - Target is log_ndvi at the month immediately following the window
        - Chronological split: train ≤ 2021, test > 2021

    Dynamic features (per timestep):
        log_ndvi, lst_driver_lag1/2/3,
        log_precip_driver_lag1/2/3, ndvi_spatial_lag,
        pop_density  → 10 channels

    Static features (per node, time-invariant):
        twi, soil_snum (one-hot expanded) → varies by soil classes
    """

    DYNAMIC_COLS = [
        'log_ndvi',
        'lst_driver_lag1', 'lst_driver_lag2', 'lst_driver_lag3',
        'log_precip_driver_lag1', 'log_precip_driver_lag2', 'log_precip_driver_lag3',
        'ndvi_spatial_lag',
        'pop_density',
    ]

    STATIC_COLS_RAW = ['twi', 'soil_snum']

    def __init__(
        self,
        csv_path   : str,
        height     : int  = 159,
        width      : int  = 181,
        window_size: int  = 12,
        split      : str  = 'train',
        split_year : int  = 2021,
        scaler     : StandardScaler = None,
    ):
        """
        Args:
            csv_path    : Path to tabular_dataset.csv
            height      : Raster grid height (159)
            width       : Raster grid width (181)
            window_size : Number of input timesteps (12 months)
            split       : 'train' or 'test'
            split_year  : Year boundary for chronological split
            scaler      : Fitted StandardScaler (pass train scaler to test set)
        """
        self.height      = height
        self.width       = width
        self.n_nodes     = height * width
        self.window_size = window_size
        self.split       = split
        self.split_year  = split_year

        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

        # --- One-hot encode soil ---
        df = pd.get_dummies(df, columns=['soil_snum'], drop_first=True)
        soil_cols = [c for c in df.columns if c.startswith('soil_snum_')]
        self.static_cols = ['twi'] + soil_cols

        # --- Sort chronologically ---
        df = df.sort_values(['year', 'month']).reset_index(drop=True)

        # --- Get unique timesteps ---
        timesteps = df[['year', 'month']].drop_duplicates().sort_values(
            ['year', 'month']
        ).reset_index(drop=True)
        self.timesteps = timesteps

        # --- Build 3D spatial arrays: [T, H*W, F] ---
        print("Pivoting tabular data to spatial grid format...")
        T = len(timesteps)
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

            # Pixel index within flattened grid — positional order from dataset
            pixel_indices = sub.index % self.n_nodes

            for f_idx, col in enumerate(self.DYNAMIC_COLS):
                if col in sub.columns:
                    dynamic_array[t_idx, pixel_indices, f_idx] = sub[col].values.astype(np.float32)

            if not static_filled:
                for f_idx, col in enumerate(self.static_cols):
                    if col in sub.columns:
                        static_array[pixel_indices, f_idx] = sub[col].values.astype(np.float32)
                static_filled = True

        # --- Chronological split on timestep indices ---
        train_mask = (timesteps['year'] <= split_year).values
        test_mask  = (timesteps['year'] >  split_year).values

        if split == 'train':
            valid_t = np.where(train_mask)[0]
        else:
            valid_t = np.where(test_mask)[0]

        # --- Build sliding windows ---
        # Each sample: window_size input steps + 1 target step
        # Target must be within valid split range
        self.windows = []
        all_t = np.arange(T)

        for t in valid_t:
            # Need window_size steps before t, and t itself as target
            start = t - window_size
            if start < 0:
                continue
            # All window steps must exist (no gap checking for now)
            self.windows.append((start, t))

        print(f"  Split '{split}': {len(self.windows)} windows "
              f"from {len(valid_t)} valid timesteps")

        # --- Normalize dynamic features ---
        # Fit scaler on train dynamic data only
        train_t_indices = np.where(train_mask)[0]
        train_dynamic   = dynamic_array[train_t_indices]  # [T_train, N, F]

        # Reshape to [T_train * N, F] for scaler, fit on non-NaN values
        flat_train = train_dynamic.reshape(-1, n_dynamic)
        valid_rows = ~np.isnan(flat_train).any(axis=1)

        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(flat_train[valid_rows])
        else:
            self.scaler = scaler

        # Apply scaler to all timesteps
        flat_all   = dynamic_array.reshape(-1, n_dynamic)
        valid_all  = ~np.isnan(flat_all).any(axis=1)
        flat_scaled = flat_all.copy()
        flat_scaled[valid_all] = self.scaler.transform(flat_all[valid_all]).astype(np.float32)
        self.dynamic_array = flat_scaled.reshape(T, self.n_nodes, n_dynamic)

        # --- Normalize static features ---
        static_scaler  = StandardScaler()
        valid_static   = ~np.isnan(static_array).any(axis=1)
        static_scaled  = static_array.copy()
        static_scaled[valid_static] = static_scaler.fit_transform(
            static_array[valid_static]
        ).astype(np.float32)
        self.static_array = static_scaled

        # Replace remaining NaNs with 0 (masked pixels)
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

        # Input sequence: [window_size, n_nodes, n_dynamic_features]
        x = torch.tensor(
            self.dynamic_array[start:t],
            dtype=torch.float32
        )

        # Static features: [n_nodes, n_static_features]
        u = torch.tensor(
            self.static_array,
            dtype=torch.float32
        )

        # Target: log_ndvi at timestep t — feature index 0
        y = torch.tensor(
            self.dynamic_array[t, :, 0:1],  # [n_nodes, 1]
            dtype=torch.float32
        )

        return x, u, y


def build_datasets(config, window_size=12):
    """
    Convenience function to build train and test datasets,
    sharing the same scaler fitted on training data.

    Args:
        config      : Loaded config.yaml dict
        window_size : Lookback window in months (default: 12)

    Returns:
        train_dataset, test_dataset
    """
    csv_path   = os.path.join(config['paths']['processed_dir'], "tabular_dataset.csv")
    split_year = config['features']['train_split_year']
    height     = config['spatial']['height']
    width      = config['spatial']['width']

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
        scaler      = train_dataset.scaler,   # share fitted scaler
    )

    return train_dataset, test_dataset