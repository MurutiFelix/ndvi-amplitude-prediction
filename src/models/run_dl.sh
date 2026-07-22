#!/bin/bash
#SBATCH --job-name=dl2_ndv
#SBATCH --partition=gpulong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/dl_%j.out
#SBATCH --error=logs/dl_%j.err

# --- Thread & Environment Isolation ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONNOUSERSITE=1

# --- Load Base Conda ---
module purge
module load applications/eng/gpu/python/conda-26.1.0-python-3.12-vLLM
source /scratch/lustre/apps/eng/gpu/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lustre/users/$USER/envs/ndvi_env

# --- Target Local Packages First ---
MY_ENV_PACKAGES="/scratch/lustre/users/$USER/envs/ndvi_env/lib/python3.12/site-packages"
export PYTHONPATH="${MY_ENV_PACKAGES}:/scratch/lustre/users/$USER/ndvi-amplitude-prediction:${PYTHONPATH}"
export LD_LIBRARY_PATH="${MY_ENV_PACKAGES}/torch/lib:${MY_ENV_PACKAGES}/torch_scatter:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# --- Working Directory Setup ---
cd /scratch/lustre/users/$USER/ndvi-amplitude-prediction
mkdir -p logs

# --- GPU & Extension Diagnostics ---
echo "Job started on: $(date)"
echo "Using python: $(which python)"
echo "Slurm CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

python -c "
import torch
import torch_scatter
import torch_sparse
print('='*40)
print('PyTorch Location:', torch.__file__)
print('CUDA Available:', torch.cuda.is_available())
print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('torch_scatter loaded successfully!')
print('torch_sparse loaded successfully!')
print('='*40)
"

# --- Run DL Training ---
python -m src.models.train

echo "Job finished on: $(date)"