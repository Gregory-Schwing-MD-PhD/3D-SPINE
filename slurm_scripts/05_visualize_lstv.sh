#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=lstv_viz
#SBATCH -o logs/lstv_viz_%j.out
#SBATCH -e logs/lstv_viz_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

STUDY_ID="${STUDY_ID:-}"

echo "================================================================"
echo "LSTV VISUALIZATION (registered space)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start:  $(date)"
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
REGISTERED_DIR="${PROJECT_DIR}/results/registered"
LSTV_DIR="${PROJECT_DIR}/results/lstv_detection"
OUTPUT_DIR="${PROJECT_DIR}/results/lstv_viz"

mkdir -p logs "$OUTPUT_DIR"

# --- Container ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container image..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Optional args ---
LSTV_JSON="${LSTV_DIR}/lstv_results.json"
LSTV_JSON_ARG=""
if [[ -f "$LSTV_JSON" ]]; then
    LSTV_JSON_ARG="--lstv_json /work/results/lstv_detection/lstv_results.json"
    echo "Detection results found — summary panels will be annotated."
else
    echo "WARNING: lstv_results.json not found. Run 04_lstv_detection.sh first."
fi

STUDY_ID_ARG=""
if [[ -n "$STUDY_ID" ]]; then
    STUDY_ID_ARG="--study_id ${STUDY_ID}"
    echo "Single-study mode: $STUDY_ID"
else
    echo "Batch mode"
fi

# --- Run ---
singularity exec \
    --bind "${PROJECT_DIR}":/work \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/05_visualize_overlay.py \
        --registered_dir  /work/results/registered \
        --nifti_dir       /work/results/nifti \
        --spineps_dir     /work/results/spineps \
        --totalspine_dir  /work/results/totalspineseg \
        --output_dir      /work/results/lstv_viz \
        ${STUDY_ID_ARG} \
        ${LSTV_JSON_ARG}

echo "================================================================"
echo "Visualization complete | PNGs -> ${OUTPUT_DIR}"
echo "End: $(date)"
echo "================================================================"
