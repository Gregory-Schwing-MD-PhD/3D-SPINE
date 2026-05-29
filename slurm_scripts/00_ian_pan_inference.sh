#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name=ian_pan_inference
#SBATCH -o logs/ian_pan_%j.out
#SBATCH -e logs/ian_pan_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

MODE=${MODE:-prod}
RETRY_FAILED=${RETRY_FAILED:-false}

echo "================================================================"
echo "IAN PAN EPISTEMIC UNCERTAINTY INFERENCE"
echo "Mode: $MODE"
echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "================================================================"

nvidia-smi

# --- Singularity temp setup (matches working trial script) ---
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="$SINGULARITY_TMPDIR/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
# Point Singularity's OWN cache at the same dir (NXF_* is only read by Nextflow)
export SINGULARITY_CACHEDIR="${NXF_SINGULARITY_CACHEDIR}"
mkdir -p "$SINGULARITY_TMPDIR" "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
DICOM_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
OUTPUT_DIR="${PROJECT_DIR}/results/epistemic_uncertainty"
MODELS_DIR="${PROJECT_DIR}/models"

mkdir -p logs "$OUTPUT_DIR"

# --- Preflight ---
if [[ ! -d "$DICOM_DIR" ]]; then
    echo "ERROR: DICOM directory not found: $DICOM_DIR"
    exit 1
fi

if [[ ! -f "${MODELS_DIR}/valid_id.npy" ]]; then
    echo "ERROR: valid_id.npy not found at ${MODELS_DIR}/valid_id.npy"
    exit 1
fi

if [[ ! -f "${MODELS_DIR}/point_net_checkpoint.pth" ]]; then
    echo "ERROR: point_net_checkpoint.pth not found at ${MODELS_DIR}/point_net_checkpoint.pth"
    exit 1
fi

N_STUDIES=$(ls -d "${DICOM_DIR}"/*/ 2>/dev/null | wc -l)
echo "Studies found in DICOM dir: $N_STUDIES"

# --- Container (same one that worked in trial) ---
CONTAINER="docker://go2432/lstv-uncertainty:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/lstv-uncertainty.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    TMP_SIF="${IMG_PATH}.tmp.${SLURM_JOB_ID}"
    rm -f "$TMP_SIF"
    # Pull to a temp path so a failed/partial pull never leaves a corrupt .sif.
    # On a stale-cache descriptor error, wipe the cache and retry once.
    if ! singularity pull "$TMP_SIF" "$CONTAINER"; then
        echo "Pull failed; cleaning Singularity cache and retrying once..."
        singularity cache clean -f || true
        rm -rf "${SINGULARITY_CACHEDIR}"/{cache,oci-tmp,oci-blob,blob,library,net,oras,shub} 2>/dev/null || true
        rm -f "$TMP_SIF"
        singularity pull "$TMP_SIF" "$CONTAINER"
    fi
    mv -f "$TMP_SIF" "$IMG_PATH"
fi
echo "Container ready: $IMG_PATH"

# --- Args ---
RETRY_ARG=""
if [[ "$RETRY_FAILED" == "true" ]]; then
    RETRY_ARG="--retry_failed"
fi

echo "================================================================"
echo "Starting inference"
echo "DICOM root: $DICOM_DIR"
echo "Output:     $OUTPUT_DIR"
echo "================================================================"

# --- Run ---
singularity exec --nv \
    --bind "${PROJECT_DIR}:/work" \
    --bind "${DICOM_DIR}:/data/input" \
    --bind "${OUTPUT_DIR}:/data/output" \
    --bind "${MODELS_DIR}:/app/models" \
    --bind "$(dirname $SERIES_CSV):/data/raw" \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/inference_dicom.py \
        --input_dir  /data/input \
        --series_csv /data/raw/train_series_descriptions.csv \
        --output_dir /data/output \
        --checkpoint /app/models/point_net_checkpoint.pth \
        --valid_ids  /app/models/valid_id.npy \
        --mode       "$MODE" \
        --trial_size "${TRIAL_SIZE:-3}" \
        $RETRY_ARG

echo "================================================================"
echo "Inference complete | End: $(date)"
echo "Output: $OUTPUT_DIR"
echo ""
echo "CSV:    ${OUTPUT_DIR}/lstv_uncertainty_metrics.csv"
echo "Debug:  ${OUTPUT_DIR}/debug_visualizations/"
echo ""
echo "Next: TOP_N=10 sbatch slurm_scripts/02b_spineps_selective.sh"
echo "================================================================"
