# A spatiotemporal deep learning framework for predicting vegetation dynamics using biophysical, edaphic, anthropogenic, and geomorphological parameters

This project implements a comprehensive pipeline for predicting NDVI (Normalized Difference Vegetation Index) amplitude - a key indicator of vegetation productivity - using a multi-modal approach that integrates:

- Biophysical drivers: Land Surface Temperature (LST), Precipitation
- Edaphic factors: Soil type, Topographic Wetness Index (TWI)
- Anthropogenic influences: Population density
- Geomorphological constraints: Topography, drainage patterns

The pipeline supports both traditional machine learning baselines (OLS, GLM, Random Forest, XGBoost) and deep learning models built on the Torch Spatiotemporal Library (TSL), including Graph Neural Networks (GNNs) for capturing complex spatial dependencies.

This work is based on the methodology from:

**Ma, Y., et al. (2022). "Forecasting vegetation dynamics in an open ecosystem by integrating deep learning and environmental variables."**


---

## Folder Structure

> \[!NOTE] 
> * The Data folder isnt available online as its huge--details on how to get the datasets are provided

```

├── data/                              # ALL EXPERIMENT DATA MODALITIES 
│   ├── raw/                            # Dynamic spatiotemporal rasters (.tif) from MODIS & ERA5 (NDVI, LST, Precip), and population ratsers
│   ├── static/                         # Landscape-invariant rasters (.tif) dictating constraints (TWI, Soil DSMW)
│   └── processed/                      # Output matrix cache (tabular_dataset.csv, baseline scores, GLM regression reports)
│   │
├── src/                               # MAIN SOURCE CODE HERE 
│   ├── config.yaml                     # Centralized pipeline configuration (hyperparameters, paths, random seeds, lags)
│   ├── train.py                        # Root execution orchestrator that imports modules to run the entire end-to-end pipeline
│   ├── predict.py                      # Out-of-sample forward inference engine
│   ├── logs/                           # Automated cluster logs directory (captures stdout/stderr from Slurm execution runs)
│   │
│   ├── data/                           # Data Engineering
│   │   ├── dataset.py                  # Custom PyTorch Dataset/Loader streaming architectures for neural network training
│   │   ├── raster_processor.py         # Heavy geospatial engine: handles raster alignment, coordinate mapping, and 3D-to-2D flattening
│   │   └── analyze_and_tune.py         #Baseline training orchestrator, feature importance extraction, spatial residual mapping, and HT
│   │  
│   ├── models/                         # ----
│   │   ├── run_baselines.sh            # Slurm cluster shell script containing partition nodes, environments, and execution tasks
│   │   ├── run_dl.sh                   # Slurm cluster shell script for running the dl training
│   │   ├── baselines.py                # Pipeline class containing cyclic time mapping, interaction features, and baseline models (XGB/RF)
│   │   ├── train.py                    # Model training loops, tracking, and metric evaluation modules specifically for Deep Learning models
│   │   ├── spatio_temporal.py          # Custom Deep Learning neural architectures (CNN, LSTM, or Transformer encoders)
│   │   └── loss.py                     # Custom loss functions tailored for spatial optimization and amplitude variance penalties
│   │
│   └── utils/                          # Utilitiies
│       └── spatial.py                  # Spatial helper functions (coordinate conversions, windowing, spatial weight matrices)
│
└── requirements.txt                   # Project environment dependencies lockfile    

```

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HPC](https://img.shields.io/badge/HPC-KENET-brightgreen.svg)](https://kenet.or.ke/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---
Baselines: OLS, GLM, Random Forest, XGBoost (with hyperparameter tuning)

Deep Learning: 4 state-of-the-art spatiotemporal models from TSL
- STID (Spatial-Temporal Identity)
- DCRNN (Diffusion Convolutional RNN)
- GRUGCN (GRU + Graph Convolutional Network)
- GraphWaveNet (Learned adjacency + dilated convolutions)

### Spatial Graph Representation
Treats each pixel as a node in a graph (159×181 = 28,779 nodes).
 8-connectivity (queen contiguity) for spatial edges.
Enables models to learn spatial dependencies and propagation patterns.

## Step 1: Clone Project to Scratch
Ensure execution runs inside your cluster’s scratch space:

```bash
cd /scratch/lustre/users/$USER
git clone [https://github.com/YOUR_USERNAME/ndvi-amplitude-prediction.git](https://github.com/YOUR_USERNAME/ndvi-amplitude-prediction.git)
cd ndvi-amplitude-prediction
```
---

## Step 2 — Load the GPU Python module

```bash
module load applications/eng/gpu/python/conda-26.1.0-python-3.14
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Step 4 — Upload your data

Place GeoTIFF files under scratch:

```
/scratch/lustre/users/$USER/project/data/

data/                                 # ALL EXPERIMENT DATA MODALITIES 
    ├── raw/                            # Dynamic spatiotemporal rasters (.tif) from MODIS & ERA5 (NDVI, LST, Precip), and population ratsers
    ├── static/                         # Landscape-invariant rasters (.tif) dictating constraints (TWI, Soil DSMW)
    └── processed/                      # Output matrix cache (tabular_dataset.csv, baseline scores, GLM regression reports)

```

---

## Step 5 — Run a debug job first 

```bash
mkdir -p /scratch/lustre/users/$USER/project/logs
sbatch debug.slurm
```

Monitor:
```bash
squeue -u $USER
tail -f /scratch/lustre/users/$USER/project/logs/debug.*.out
```

Expected output if pipeline is healthy:
```
GPU check:  NVIDIA L40  47.999... GB
Building patches ...
Debug training (1 GPU, 1000 iters) ...
  iter      100 | loss 0.54321 | lr 1.00e-04 | ...
  ...
  iter     1000 | loss 0.38712 | ...
Debug complete.
```

---

## Step 6 — Submit the full training job

```bash
sbatch src/models/run_baselines.sh
```
That trains te baseline models
For deep learning use:
```bash
sbatch src/models/run_dl.sh
```
You'll receive:
```
Submitted batch job xyz
```

The Slurm script runs all 5 phases sequentially in one 72-hour job:

- **Phase 5** — Evaluate all models, save CSV + plots

---

## Step 7 — Monitor training

```bash
# Check job status
squeue -u $USER

# Live log tail
tail -f /scratch/lustre/users/$USER/yourprojectname/logs/train.xyz.out

# GPU utilisation (inside interactive job)
srun --gres=gpu:1 --partition=gpu1 --time=00:10:00 --pty bash -i
nvidia-smi -l 2

# Memory usage report (KENET tool)
jobmem
```

---

## Step 8 — Run inference on .



## Resource configuration rationale

| Resource | Value | Reason |
|---|---|---|
| `--gres=gpu:1` | One L40s |  |
| `--mem=300000` | 300 GB | Patch loading + DataLoader prefetch buffers |
| `--cpus-per-task=48` | 48 cores | 16 DataLoader workers × 2 GPU processes + headroom |
| `--time=16:00:00` | 72 h |Current runs use 16hrs on the gpu
| `batch_size=64` | 32/GPU | Fills 48 GB L40 VRAM for 128px patches |
| `workers=16` | 8/GPU | Keeps GPU feed-starved minimised |

---

