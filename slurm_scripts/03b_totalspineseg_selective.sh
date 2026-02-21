#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=36:00:00
#SBATCH --job-name=totalspineseg_selective
#SBATCH -o logs/totalspineseg_selective_%j.out
#SBATCH -e logs/totalspineseg_selective_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Configuration — edit these two lines to change behaviour ─────────────────
TOP_N=1                   # studies from each end (top 1 highest + bottom 1 lowest)
RANK_BY=l5_s1_confidence  # column in lstv_uncertainty_metrics.csv to rank by
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "TOTALSPINESEG SELECTIVE SEGMENTATION"
echo "Top/Bottom N: $TOP_N"
echo "Rank by:      $RANK_BY"
echo "Job ID:       $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start:        $(date)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || echo "WARNING: singularity not found"
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"

SCRATCH_DIR="/wsu/tmp/${USER}/totalspineseg_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"
export SINGULARITY_TMPDIR="$SCRATCH_DIR"

mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"
echo "Scratch dir: $SCRATCH_DIR"

export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
UNCERTAINTY_CSV="${PROJECT_DIR}/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv"
NIFTI_DIR="${PROJECT_DIR}/results/nifti"
OUTPUT_DIR="${PROJECT_DIR}/results/totalspineseg"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
MODELS_DIR="${PROJECT_DIR}/models"
TOTALSPINESEG_MODELS="${PROJECT_DIR}/models/totalspineseg_models"
NNUNET_TRAINER_DIR="${PROJECT_DIR}/models/nnunetv2_trainer"

mkdir -p logs "$OUTPUT_DIR" "$TOTALSPINESEG_MODELS" "$NNUNET_TRAINER_DIR"

# --- Preflight ---
if [[ ! -f "$UNCERTAINTY_CSV" ]]; then
    echo "ERROR: Uncertainty CSV not found: $UNCERTAINTY_CSV"
    echo "Run sbatch slurm_scripts/00_ian_pan_inference.sh first"
    exit 1
fi

if [[ ! -f "${MODELS_DIR}/valid_id.npy" ]]; then
    echo "ERROR: valid_id.npy not found at ${MODELS_DIR}/valid_id.npy"
    exit 1
fi

echo "Studies in uncertainty CSV: $(( $(wc -l < "$UNCERTAINTY_CSV") - 1 ))"

# --- Container ---
CONTAINER="docker://go2432/totalspineseg:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/totalspineseg.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling TotalSpineSeg container (first time only)..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# Extract nnUNetTrainer directory from container if not already done (one-time setup)
if [[ -z "$(ls -A $NNUNET_TRAINER_DIR 2>/dev/null)" ]]; then
    echo "Extracting nnUNetTrainer directory from container (one-time setup)..."
    singularity exec "$IMG_PATH" \
        cp -r /opt/conda/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/. \
        "$NNUNET_TRAINER_DIR/"
    echo "Extracted $(ls $NNUNET_TRAINER_DIR | wc -l) files"
fi

# --- Run ---
singularity exec --nv \
    --bind "${PROJECT_DIR}":/work \
    --bind "${NIFTI_DIR}":/work/results/nifti \
    --bind "${OUTPUT_DIR}":/work/results/totalspineseg \
    --bind "${MODELS_DIR}":/app/models \
    --bind "${TOTALSPINESEG_MODELS}":/app/totalspineseg_models \
    --bind "${NNUNET_TRAINER_DIR}":/opt/conda/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer \
    --env TOTALSPINESEG_DATA=/app/totalspineseg_models \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/03b_totalspineseg_selective.py \
        --uncertainty_csv /work/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv \
        --nifti_dir       /work/results/nifti \
        --output_dir      /work/results/totalspineseg \
        --series_csv      /work/data/raw/train_series_descriptions.csv \
        --valid_ids       /app/models/valid_id.npy \
        --top_n           "$TOP_N" \
        --rank_by         "$RANK_BY"

echo "================================================================"
echo "Selective TotalSpineSeg complete | End: $(date)"
echo "To segment more: edit TOP_N in this script and resubmit."
echo "Already-done studies are skipped automatically."
echo "================================================================"

rm -rf "$SCRATCH_DIR"
echo "Scratch cleanup complete."
