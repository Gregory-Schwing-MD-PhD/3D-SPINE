#!/bin/bash
#SBATCH -q standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --job-name=dicom_nifti
#SBATCH -o logs/dicom_nifti_%j.out
#SBATCH -e logs/dicom_nifti_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

MODE=${MODE:-prod}

echo "================================================================"
echo "DICOM TO NIFTI CONVERSION"
echo "Mode: $MODE"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================================================"

# Environment
module load dcm2niix || echo "dcm2niix not in modules, assuming in PATH"

# Paths
PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
OUTPUT_DIR="${PROJECT_DIR}/results/nifti"

mkdir -p logs "${OUTPUT_DIR}"

# Run
python scripts/01_dicom_to_nifti.py \
    --input_dir "${DATA_DIR}" \
    --series_csv "${SERIES_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --mode "${MODE}"

echo "================================================================"
echo "Conversion complete | End: $(date)"
echo "================================================================"
