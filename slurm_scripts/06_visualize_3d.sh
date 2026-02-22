#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=lstv_3d
#SBATCH -o logs/lstv_3d_%j.out
#SBATCH -e logs/lstv_3d_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Configuration — edit these to change behaviour ────────────────────────────
STUDY_ID=""                # single study ID — leave empty to use ALL or TOP_N mode
TOP_N=1                    # studies from each end — must match 02b + 03b + register + detect settings
RANK_BY=l5_s1_confidence   # column to rank by — must match all upstream settings
ALL=false                  # set to true to render every study with SPINEPS segmentation
STEP=2                     # marching cubes subsampling step (1=full res, slower; 3=faster batch)
SMOOTH=1.5                 # Gaussian pre-smoothing sigma for marching cubes surfaces
SHOW_CANAL=false           # set to true to render spinal canal (label 61, translucent mint)
SHOW_CORD=false            # set to true to render spinal cord (label 60)
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "LSTV 3D VISUALIZER (Interactive HTML — Marching Cubes)"
echo "STUDY_ID=${STUDY_ID:-<selective/all>}  TOP_N=$TOP_N  RANK_BY=$RANK_BY"
echo "ALL=$ALL  STEP=$STEP  SMOOTH=$SMOOTH"
echo "SHOW_CANAL=$SHOW_CANAL  SHOW_CORD=$SHOW_CORD"
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
LSTV_JSON="${PROJECT_DIR}/results/lstv_detection/lstv_results.json"

mkdir -p logs results/lstv_3d

# --- Preflight ---
if [[ ! -d "${PROJECT_DIR}/results/spineps/segmentations" ]]; then
    echo "ERROR: SPINEPS segmentations not found. Run 02b_spineps_selective.sh first"
    exit 1
fi

if [[ ! -d "${PROJECT_DIR}/results/totalspineseg" ]]; then
    echo "ERROR: TotalSpineSeg results not found. Run 03b_totalspineseg_selective.sh first"
    exit 1
fi

# --- Container ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container image..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Build selection args ---
SELECTION_ARGS=()

if [[ -n "$STUDY_ID" ]]; then
    # Single study — highest priority, skip all other selection logic
    SELECTION_ARGS+=( "--study_id" "$STUDY_ID" )
    echo "Single-study mode: $STUDY_ID"

elif [[ "$ALL" == "true" ]]; then
    SELECTION_ARGS+=( "--all" )
    echo "ALL mode: rendering every study with SPINEPS segmentations"
    echo "Note: Phase 2 axial panels will be empty for studies missing TSS or registration."

else
    # Selective mode — mirrors 05_visualize_lstv.sh exactly
    if [[ ! -f "$UNCERTAINTY_CSV" ]]; then
        echo "ERROR: Uncertainty CSV not found: $UNCERTAINTY_CSV"
        echo "Run 00_ian_pan_inference.sh first, or set ALL=true or STUDY_ID"
        exit 1
    fi
    if [[ ! -f "${MODELS_DIR}/valid_id.npy" ]]; then
        echo "ERROR: valid_id.npy not found at ${MODELS_DIR}/valid_id.npy"
        exit 1
    fi
    SELECTION_ARGS+=(
        "--uncertainty_csv" "/work/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv"
        "--valid_ids"       "/app/models/valid_id.npy"
        "--top_n"           "$TOP_N"
        "--rank_by"         "$RANK_BY"
    )
    echo "Selective mode: top/bottom $TOP_N by $RANK_BY"
fi

# --- Optional rendering flags ---
if [[ "$SHOW_CANAL" == "true" ]]; then
    SELECTION_ARGS+=( "--show_canal" )
    echo "Spinal canal rendering enabled (label 61)"
fi

if [[ "$SHOW_CORD" == "true" ]]; then
    SELECTION_ARGS+=( "--show_cord" )
    echo "Spinal cord rendering enabled (label 60)"
fi

# --- Annotation from detection results (optional but recommended) ---
if [[ -f "$LSTV_JSON" ]]; then
    SELECTION_ARGS+=( "--lstv_json" "/work/results/lstv_detection/lstv_results.json" )
    echo "Detection results found — HTML headers will be annotated with Castellvi classification."
else
    echo "WARNING: lstv_results.json not found — run 04_lstv_detection.sh first for annotations."
fi

# --- Bind models dir only when valid_ids is needed (selective mode) ---
BIND_ARGS="--bind ${PROJECT_DIR}:/work"
if [[ -d "$MODELS_DIR" ]]; then
    BIND_ARGS="${BIND_ARGS} --bind ${MODELS_DIR}:/app/models"
fi

# --- Run ---
singularity exec \
    $BIND_ARGS \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/06_visualize_3d.py \
        --spineps_dir    /work/results/spineps \
        --totalspine_dir /work/results/totalspineseg \
        --output_dir     /work/results/lstv_3d \
        --step           "$STEP" \
        --smooth         "$SMOOTH" \
        "${SELECTION_ARGS[@]}"

echo "================================================================"
echo "3D visualization complete | HTMLs -> results/lstv_3d/"
echo "End: $(date)"
echo "================================================================"
