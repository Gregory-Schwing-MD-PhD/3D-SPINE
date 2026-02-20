#!/bin/bash
#SBATCH -q standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=register_axial
#SBATCH -o logs/register_%j.out
#SBATCH -e logs/register_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

MODE="${MODE:-prod}"
RETRY_FAILED="${RETRY_FAILED:-false}"
STUDY_ID="${STUDY_ID:-}"

echo "================================================================"
echo "REGISTER SAG -> AXIAL SPACE"
echo "Mode:  $MODE"
echo "Job:   $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || { echo "ERROR: singularity not found"; exit 1; }

export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
NIFTI_DIR="${PROJECT_DIR}/results/nifti"
SPINEPS_DIR="${PROJECT_DIR}/results/spineps"
TOTALSPINE_DIR="${PROJECT_DIR}/results/totalspineseg"
REGISTERED_DIR="${PROJECT_DIR}/results/registered"

mkdir -p logs "$REGISTERED_DIR"

# --- Container ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container image..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Build optional args ---
EXTRA_ARGS=""
if [[ "$RETRY_FAILED" == "true" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --retry-failed"
fi
if [[ -n "$STUDY_ID" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --study_id ${STUDY_ID}"
    echo "Single-study mode: $STUDY_ID"
fi

# --- Run ---
singularity exec \
    --bind "${PROJECT_DIR}":/work \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/03b_register_to_axial.py \
        --nifti_dir      /work/results/nifti \
        --spineps_dir    /work/results/spineps \
        --totalspine_dir /work/results/totalspineseg \
        --registered_dir /work/results/registered \
        --mode           "$MODE" \
        $EXTRA_ARGS

echo "================================================================"
echo "Registration complete | End: $(date)"
echo "================================================================"
