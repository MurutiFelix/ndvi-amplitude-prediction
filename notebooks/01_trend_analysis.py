"""
01_trend_analysis.py
Calculates pixel-wise Mann-Kendall and Sen's Slope trends for monthly NDVI and LST stacks.
Outputs are written to data/processed/.
"""

import os
import numpy as np
import rasterio
import pymannkendall as mk
from tqdm import tqdm
from pathlib import Path

# Setup paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

YEARS = list(range(2006, 2026))
MONTHS = [f"{m:02d}" for m in range(1, 13)]
TIME_STEPS = [f"{y}_{m}" for y in YEARS for m in MONTHS]

NDVI_PATTERN = "NDVI_{step}.tif"
LST_PATTERN  = "LST_{step}.tif"
ALPHA = 0.05
NODATA = -9999.0

def load_stack(directory: Path, pattern: str, time_steps: list) -> tuple[np.ndarray, dict]:
    files = [directory / pattern.format(step=ts) for ts in time_steps]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files:\n" + "\n".join(missing))

    with rasterio.open(files[0]) as src:
        profile = src.profile.copy()
        rows, cols = src.height, src.width

    stack = np.full((len(time_steps), rows, cols), np.nan, dtype=np.float32)
    for i, f in enumerate(tqdm(files, desc=f"Loading {directory.name}")):
        with rasterio.open(f) as src:
            data = src.read(1).astype(np.float32)
            raster_nodata = src.nodata if src.nodata is not None else NODATA
            data[data == raster_nodata] = np.nan
            stack[i] = data

    return stack, profile

def pixelwise_mk_sens(stack: np.ndarray):
    n, rows, cols = stack.shape
    z_score    = np.full((rows, cols), np.nan, dtype=np.float32)
    p_value    = np.full((rows, cols), np.nan, dtype=np.float32)
    trend_flag = np.full((rows, cols), np.nan, dtype=np.float32)
    sens_slope = np.full((rows, cols), np.nan, dtype=np.float32)

    valid_pixels = 0

    for r in tqdm(range(rows), desc="Processing rows"):
        for c in range(cols):
            ts = stack[:, r, c]
            valid = ts[~np.isnan(ts)]
            if len(valid) < 12:
                continue

            if np.any(np.isnan(ts)):
                x = np.arange(n)
                mask = ~np.isnan(ts)
                ts = np.interp(x, x[mask], ts[mask]).astype(np.float32)

            try:
                result = mk.original_test(ts, alpha=ALPHA)
                z_score[r, c]    = result.z
                p_value[r, c]    = result.p
                trend_flag[r, c] = (1 if result.trend == "increasing" 
                                    else -1 if result.trend == "decreasing" 
                                    else 0)
                sens_slope[r, c] = result.slope
                valid_pixels = valid_pixels + 1
            except Exception:
                continue

    return z_score, p_value, trend_flag, sens_slope

def save_raster(array: np.ndarray, profile: dict, path: Path):
    out_profile = profile.copy()
    out_profile.update({"count": 1, "dtype": "float32", "nodata": NODATA, "compress": "lzw"})
    out_arr = array.copy()
    out_arr[np.isnan(out_arr)] = NODATA

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(out_arr.astype(np.float32), 1)

def run_variable_pipeline(var_name: str, pattern: str):
    print(f"\nProcessing trends for: {var_name}")
    stack, profile = load_stack(DATA_DIR, pattern, TIME_STEPS)
    z, p, flag, slope = pixelwise_mk_sens(stack)
    
    save_raster(z, profile, OUT_DIR / f"{var_name.lower()}_MK_zscore.tif")
    save_raster(p, profile, OUT_DIR / f"{var_name.lower()}_MK_pvalue.tif")
    save_raster(flag, profile, OUT_DIR / f"{var_name.lower()}_MK_trend.tif")
    save_raster(slope, profile, OUT_DIR / f"{var_name.lower()}_sens_slope.tif")
    print(f"Completed {var_name} features")

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_variable_pipeline("NDVI", NDVI_PATTERN)
    run_variable_pipeline("LST", LST_PATTERN)
