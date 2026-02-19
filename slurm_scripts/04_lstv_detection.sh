#!/bin/bash
#SBATCH -q standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=lstv_detect
#SBATCH -o logs/lstv_detect_%j.out
#SBATCH -e logs/lstv_detect_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "================================================================"
echo "LSTV DETECTION"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================================================"

# --- Paths ---
PROJECT_DIR="$(pwd)"
SPINEPS_DIR="${PROJECT_DIR}/results/spineps"
TOTALSPINE_DIR="${PROJECT_DIR}/results/totalspineseg"
OUTPUT_DIR="${PROJECT_DIR}/results/lstv_detection"

mkdir -p logs "${OUTPUT_DIR}"

# --- Run ---
python scripts/04_detect_lstv.py \
    --spineps_dir "${SPINEPS_DIR}" \
    --totalspine_dir "${TOTALSPINE_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "================================================================"
echo "LSTV detection complete | End: $(date)"
echo "================================================================"
