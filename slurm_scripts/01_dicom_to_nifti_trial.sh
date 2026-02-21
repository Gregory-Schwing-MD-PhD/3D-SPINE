#!/bin/bash
#SBATCH -q primary
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

MODE=${MODE:-trial}

echo "================================================================"
echo "DICOM TO NIFTI CONVERSION"
echo "Mode: $MODE"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || echo "WARNING: singularity not found"
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p $XDG_RUNTIME_DIR $NXF_SINGULARITY_CACHEDIR
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
OUTPUT_DIR="${PROJECT_DIR}/results/nifti"
MODELS_DIR="${PROJECT_DIR}/models"

mkdir -p logs "$OUTPUT_DIR"

# --- Preflight checks ---
if [[ ! -f "${MODELS_DIR}/valid_id.npy" ]]; then
    echo "ERROR: valid_id.npy not found at ${MODELS_DIR}/valid_id.npy"
    echo "Run first: sbatch slurm_scripts/00_download_all.sh"
    exit 1
fi

# --- Container ---
CONTAINER="docker://go2432/spineps-segmentation:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-segmentation.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Run ---
singularity exec \
    --bind "${PROJECT_DIR}:/work" \
    --bind "${DATA_DIR}:/data/input" \
    --bind "${OUTPUT_DIR}:/data/output" \
    --bind "$(dirname $SERIES_CSV):/data/raw" \
    --bind "${MODELS_DIR}:/app/models" \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/01_dicom_to_nifti.py \
        --input_dir  /data/input \
        --series_csv /data/raw/train_series_descriptions.csv \
        --output_dir /data/output \
        --valid_ids  /app/models/valid_id.npy \
        --mode "$MODE"

echo "================================================================"
echo "Conversion complete | End: $(date)"
echo "================================================================"
