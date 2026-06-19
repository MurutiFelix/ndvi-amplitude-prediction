# Required Libraries
import os
import logging
import cdsapi
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s-%(levelname)s-%(message)s")
log = logging.getLogger(__name__)

# Configuration
# Bounding Box upper ewaso nyiro river basin
BBOX = {
    "North": 1.1,
    "West": 36.20,
    "South": -0.4,
    "East": 37.9
}

# CDS Area extent format: [North, West, South, East]
AREA = [BBOX['North'], BBOX['West'], BBOX['South'], BBOX['East']]

# Decade Chunks
PERIODS = [
    {"start": 2006, "end": 2014, "version": "version_3_1"},
    {"start": 2015, "end": 2022, "version": "version_3_1"},
    {"start": 2023, "end": 2025, "version": "version_4_0"},
]

MONTHS = [f"{m:02d}" for m in range(1, 13)]

# Output Directories
# Output Directories
SCRIPT_DIR = Path(__file__).resolve().parent  
ERA5_DIR = SCRIPT_DIR.parent / "data" / "raw"
# API endpoints
ERA5_URL = "https://cds.climate.copernicus.eu/api"


# Reading API keys from environment variables
def get_api_keys():
    key = os.getenv("CDS_API_KEY")
    if not key:
        raise EnvironmentError("CDS_API_KEY not found in environment variables.")
    log.info("CDS API key successfully retrieved from environment variables.")
    return key


# Creating output directories
def create_directories():
    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Directory created or already exists: {ERA5_DIR}")


# Building year list for periods
def year_list(start: int, end: int):
    return [str(year) for year in range(start, end + 1)]


# Downloading ERA5 data
def download_era5(client: cdsapi.Client):
    log.info("Starting ERA5-Land downloads")
 
    for period in PERIODS:
        start = period["start"]
        end = period["end"]
 
        # THE FIX: Explicitly appending the .nc extension to output targets
        out_path = ERA5_DIR / f"era5_{start}-{end}.nc"
 
        # Skip if file already exists
        if out_path.exists():
            log.info("SKIP already exists: %s", out_path)
            continue
 
        log.info("Downloading ERA5-Land %d-%d", start, end)
 
        client.retrieve(
            "reanalysis-era5-land-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": [
                    "total_precipitation",
                    "2m_temperature",
                ],
                "year": year_list(start, end),
                "month": MONTHS,
                "time": ["00:00"],  # Explicit wrapper list format
                "area": AREA,
                "data_format": "netcdf",
                "download_format": "unarchived"
            },
            str(out_path),
        )
 
        log.info("Saved: %s", out_path)
 
    log.info("ERA5-Land downloads complete")


# Main
def main():
    log.info("Ewaso Nyiro Basin — Data Downloader ")
    
    # get API key
    key = get_api_keys()
 
    # create directories
    create_directories()
 
    # create ERA5 client and download
    log.info("Connecting to ERA5 CDS server")
    era5_client = cdsapi.Client(url=ERA5_URL, key=key)
    download_era5(era5_client)
 
    log.info("All downloads complete")
    log.info("Check data/raw/ for your files")
     

# THE FIX: Using explicit double underscores on both sides of main
if __name__ == "__main__":
    main()