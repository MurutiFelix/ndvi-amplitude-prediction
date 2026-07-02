#!/bin/bash
#SBATCH --job-name=ndvi_baselines
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=04:00:00
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err

# --- Environment Setup ---
module purge
module load applications/eng/gpu/python/conda-26.1.0-python-3.12-vLLM
source /scratch/lustre/apps/eng/gpu/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lustre/users/$USER/envs/ndvi_env

# --- Working Directory Setup ---
cd /scratch/lustre/users/$USER/ndvi-amplitude-prediction
mkdir -p logs
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}"

# --- Diagnostics ---
echo "Job started on: $(date)"
echo "Running from: $(pwd)"
echo "Using python: $(which python)"

# --- Run ---
python -m src.models.train
echo "Job finished on: $(date)"
