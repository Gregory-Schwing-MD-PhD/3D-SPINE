#!/bin/bash
#SBATCH -q primary 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
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
SPINEPS_DIR="${PROJECT_DIR}/results/spineps"
TOTALSPINE_DIR="${PROJECT_DIR}/results/totalspineseg"
OUTPUT_DIR="${PROJECT_DIR}/results/lstv_detection"

mkdir -p logs "$OUTPUT_DIR"

# --- Container (reuse preprocessing container which has numpy/nibabel) ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Run ---
singularity exec \
    --bind "$PROJECT_DIR":/work \
    --bind "$SPINEPS_DIR":/data/spineps \
    --bind "$TOTALSPINE_DIR":/data/totalspine \
    --bind "$OUTPUT_DIR":/data/output \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/04_detect_lstv.py \
        --spineps_dir    /data/spineps \
        --totalspine_dir /data/totalspine \
        --output_dir     /data/output

echo "================================================================"
echo "LSTV detection complete | End: $(date)"
echo "================================================================"
