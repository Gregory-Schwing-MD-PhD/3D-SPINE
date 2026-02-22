#!/bin/bash
#SBATCH -q primary
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

# ── Configuration — edit these to change behaviour ────────────────────────────
TOP_N=1                    # studies from each end — must match 02b + 03b settings
RANK_BY=l5_s1_confidence   # column to rank by — must match 02b + 03b settings
ALL=false                  # set to true to register every study in nifti_dir
RETRY_FAILED=false
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "REGISTER SAG -> AXIAL SPACE (Hybrid Pipeline)"
echo "TOP_N=$TOP_N  RANK_BY=$RANK_BY  ALL=$ALL"
echo "Job: $SLURM_JOB_ID  |  Start: $(date)"
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
UNCERTAINTY_CSV="${PROJECT_DIR}/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv"
MODELS_DIR="${PROJECT_DIR}/models"

mkdir -p logs results/registered

# --- Preflight ---
if [[ "$ALL" != "true" ]]; then
    if [[ ! -f "$UNCERTAINTY_CSV" ]]; then
        echo "ERROR: Uncertainty CSV not found: $UNCERTAINTY_CSV"
        echo "Run 00_ian_pan_inference.sh first, or set ALL=true"
        exit 1
    fi
    if [[ ! -f "${MODELS_DIR}/valid_id.npy" ]]; then
        echo "ERROR: valid_id.npy not found at ${MODELS_DIR}/valid_id.npy"
        exit 1
    fi
fi

# --- Container ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container image..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Build selection args (bash array avoids multiline string splitting bugs) ---
SELECTION_ARGS=()
if [[ "$ALL" == "true" ]]; then
    SELECTION_ARGS+=( "--all" )
    echo "ALL mode: registering every study in results/nifti/"
else
    SELECTION_ARGS+=(
        "--uncertainty_csv" "/work/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv"
        "--valid_ids"       "/app/models/valid_id.npy"
        "--top_n"           "$TOP_N"
        "--rank_by"         "$RANK_BY"
    )
    echo "Selective mode: top/bottom $TOP_N by $RANK_BY"
fi

[[ "$RETRY_FAILED" == "true" ]] && SELECTION_ARGS+=( "--retry-failed" )

# --- Run ---
singularity exec \
    --bind "${PROJECT_DIR}":/work \
    --bind "${MODELS_DIR}":/app/models \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/03b_register_to_axial.py \
        --nifti_dir      /work/results/nifti \
        --spineps_dir    /work/results/spineps \
        --registered_dir /work/results/registered \
        "${SELECTION_ARGS[@]}"

echo "================================================================"
echo "Registration complete | End: $(date)"
echo "================================================================"
