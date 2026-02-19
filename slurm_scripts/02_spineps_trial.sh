#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=36:00:00
#SBATCH --job-name=spineps_run
#SBATCH -o logs/spineps_%j.out
#SBATCH -e logs/spineps_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

MODE=${MODE:-trial}
RETRY_FAILED=${RETRY_FAILED:-false}

echo "================================================================"
echo "SPINEPS SEGMENTATION"
echo "Mode: $MODE"
if [[ "$RETRY_FAILED" == "true" ]]; then
    echo "Retry Failed: YES"
fi
echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
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
NIFTI_DIR="${PROJECT_DIR}/results/nifti"
OUTPUT_DIR="${PROJECT_DIR}/results/spineps"
MODELS_CACHE="${PROJECT_DIR}/models/spineps_cache"
SPINEPS_PKG_MODELS="${PROJECT_DIR}/models/spineps_pkg_models"

mkdir -p logs "${OUTPUT_DIR}" "${MODELS_CACHE}" "${SPINEPS_PKG_MODELS}"

# --- Container ---
CONTAINER="docker://go2432/spineps-segmentation:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-segmentation.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Run ---
RETRY_ARG=""
if [[ "$RETRY_FAILED" == "true" ]]; then
    RETRY_ARG="--retry-failed"
fi

singularity exec --nv \
    --bind "$PROJECT_DIR":/work \
    --bind "$NIFTI_DIR":/work/results/nifti \
    --bind "$OUTPUT_DIR":/work/results/spineps \
    --bind "$MODELS_CACHE":/app/models \
    --bind "$SPINEPS_PKG_MODELS":/opt/conda/lib/python3.10/site-packages/spineps/models \
    --env SPINEPS_SEGMENTOR_MODELS=/app/models \
    --env SPINEPS_ENVIRONMENT_DIR=/app/models \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/02_run_spineps.py \
        --nifti_dir /work/results/nifti \
        --output_dir /work/results/spineps \
        --mode "$MODE" \
        $RETRY_ARG

echo "================================================================"
echo "SPINEPS complete | End: $(date)"
echo "================================================================"
