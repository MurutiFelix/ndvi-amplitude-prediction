# A mini project based on the paper "Forecasting vegetation dynamics in an open ecosystem by integrating deep 
# learning and environmental variables " by Yue Ma et.al,(2022)

**Paper:** Yue Ma et.al,(2022) — ndvi-amplitude-prediction: A torch spatio-temporal models comparison
**Task:**  forecast vegetation state (predict ndvi)
**Hardware target:** KENET HPC — 2× NVIDIA L40 (48 GB each), 96 cores, 355 GB RAM per node
---

## Folder Structure
Note: The Data folder isnt available online as its huge--details on how to get the datasets are provided

```

├── data/                              # ALL EXPERIMENT DATA MODALITIES 
│   ├── raw/                            # Dynamic spatiotemporal rasters (.tif) from MODIS & ERA5 (NDVI, LST, Precip), and population ratsers
│   ├── static/                         # Landscape-invariant rasters (.tif) dictating constraints (TWI, Soil DSMW)
│   └── processed/                      # Output matrix cache (tabular_dataset.csv, baseline scores, GLM regression reports)
├── src/                               # MAIN SOURCE CODE CORE 
│   ├── config.yaml                     # Centralized pipeline configuration (hyperparameters, paths, random seeds, lags)
│   ├── train.py                        # Root execution orchestrator that imports modules to run the entire end-to-end pipeline
│   ├── logs/                           # Automated cluster logs directory (captures stdout/stderr from Slurm execution runs)
│   ├── data/                           # Data Engineering
│   │   ├── dataset.py                  # Custom PyTorch Dataset/Loader streaming architectures for neural network training
│   │   └── raster_processor.py         # Heavy geospatial engine: handles raster alignment, coordinate mapping, and 3D-to-2D flattening
│   ├── models/                         # ----
│   │   ├── run_baselines.sh            # Slurm cluster shell script containing partition nodes, environments, and execution tasks
│   │   ├── baselines.py                # Pipeline class containing cyclic time mapping, interaction features, and baseline models (XGB/RF)
│   │   ├── train.py                    # Model training loops, tracking, and metric evaluation modules specifically for Deep Learning models
│   │   ├── spatio_temporal.py          # Custom Deep Learning neural architectures (CNN, LSTM, or Transformer encoders)
│   │   └── loss.py                     # Custom loss functions tailored for spatial optimization and amplitude variance penalties
│   └── utils/                          # Utilitiies
│       └── spatial.py                  # Spatial helper functions (coordinate conversions, windowing, spatial weight matrices)
└── requirements.txt                   # Project environment dependencies lockfile (pip packages like scikit-learn, xgboost, rasterio)   

```