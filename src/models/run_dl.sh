#!/bin/bash
#SBATCH --job-name=ndvi_dl
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=logs/dl_%j.out
#SBATCH --error=logs/dl_%j.err

# --- Environment Setup ---
module purge
module load applications/eng/gpu/python/conda-26.1.0-python-3.12-vLLM
source /scratch/lustre/apps/eng/gpu/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lustre/users/$USER/envs/ndvi_env

# --- Working Directory Setup ---
cd /scratch/lustre/users/$USER/ndvi-amplitude-prediction
mkdir -p logs
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}"

# --- GPU Diagnostics ---
echo "Job started on: $(date)"
echo "Running from: $(pwd)"
echo "Using python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU names: $(python -c 'import torch; [print(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]')"

# --- Run DL Training ---
python -m src.models.train

echo "Job finished on: $(date)"