#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=24:00:00
#SBATCH --job-name=totalspineseg
#SBATCH -o logs/totalspineseg_%j.out
#SBATCH -e logs/totalspineseg_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

MODE=${MODE:-prod}
SERIES=${SERIES:-both}  # sagittal, axial, or both

echo "================================================================"
echo "TOTALSPINESEG SEGMENTATION"
echo "Mode: $MODE | Series: $SERIES"
echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "================================================================"

# --- Environment ---
# Load TotalSegmentator environment
# Adjust this based on your setup
conda activate totalsegmentator || source activate totalsegmentator

# --- Paths ---
PROJECT_DIR="$(pwd)"
NIFTI_DIR="${PROJECT_DIR}/results/nifti"
OUTPUT_DIR="${PROJECT_DIR}/results/totalspineseg"

mkdir -p logs "${OUTPUT_DIR}"

# --- Run ---
python scripts/03_run_totalspineseg.py \
    --nifti_dir "${NIFTI_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --series "${SERIES}" \
    --mode "${MODE}"

echo "================================================================"
echo "TotalSpineSeg complete | End: $(date)"
echo "================================================================"
