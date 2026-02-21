#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=02:00:00
#SBATCH --job-name=lstv_trial
#SBATCH -o logs/trial_%j.out
#SBATCH -e logs/trial_%j.err

set -euo pipefail
set -x

echo "================================================================"
echo "LSTV Uncertainty Detection - TRIAL MODE (10 studies)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

nvidia-smi

# Singularity temp setup
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="$SINGULARITY_TMPDIR/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$SINGULARITY_TMPDIR" "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"

trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

# Environment
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# Project paths
PROJECT_DIR="$(pwd)"
NIFTI_DIR="${PROJECT_DIR}/results/nifti"          # ← NIfTI root (not raw DICOMs)
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
OUTPUT_DIR="${PROJECT_DIR}/data/output/trial"
MODELS_DIR="${PROJECT_DIR}/models"

mkdir -p "$OUTPUT_DIR/logs" "$MODELS_DIR"

# Sanity checks
if [[ ! -d "$NIFTI_DIR" ]]; then
    echo "ERROR: NIfTI directory not found: $NIFTI_DIR"
    echo "Run DICOM conversion first: sbatch slurm_scripts/01_dicom_to_nifti.sh"
    exit 1
fi

if [[ ! -f "${MODELS_DIR}/valid_id.npy" ]]; then
    echo "ERROR: valid_id.npy not found at ${MODELS_DIR}/valid_id.npy"
    echo "Download it first: sbatch slurm_scripts/00_download_all.sh"
    exit 1
fi

# Container
CONTAINER="docker://go2432/lstv-uncertainty:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/lstv-uncertainty.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi
echo "Container ready: $IMG_PATH"

if [[ ! -f "${MODELS_DIR}/point_net_checkpoint.pth" ]]; then
    echo "================================================================"
    echo "WARNING: Model checkpoint not found — running in MOCK mode"
    echo "================================================================"
fi

echo "================================================================"
echo "Starting LSTV Uncertainty Inference - TRIAL MODE"
echo "NIfTI root:  $NIFTI_DIR"
echo "Series CSV:  $SERIES_CSV"
echo "Output:      $OUTPUT_DIR"
echo "Models:      $MODELS_DIR"
echo "================================================================"

singularity exec --nv \
    --bind "${PROJECT_DIR}:/work" \
    --bind "${NIFTI_DIR}:/data/nifti" \
    --bind "${OUTPUT_DIR}:/data/output" \
    --bind "${MODELS_DIR}:/app/models" \
    --bind "$(dirname $SERIES_CSV):/data/raw" \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/inference.py \
        --input_dir  /data/nifti \
        --series_csv /data/raw/train_series_descriptions.csv \
        --output_dir /data/output \
        --checkpoint /app/models/point_net_checkpoint.pth \
        --valid_ids  /app/models/valid_id.npy \
        --mode trial

inference_exit=$?

if [ $inference_exit -ne 0 ]; then
    echo "ERROR: Inference failed"
    exit $inference_exit
fi

echo ""
echo "================================================================"
echo "Generating HTML Report..."
echo "================================================================"

singularity exec \
    --bind "${PROJECT_DIR}:/work" \
    --bind "${OUTPUT_DIR}:/data/output" \
    --bind "${NIFTI_DIR}:/data/nifti" \
    --bind "$(dirname $SERIES_CSV):/data/raw" \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/generate_report.py \
        --csv       /data/output/lstv_uncertainty_metrics.csv \
        --output    /data/output/report.html \
        --data_dir  /data/nifti \
        --series_csv /data/raw/train_series_descriptions.csv \
        --debug_dir /data/output/debug_visualizations

echo "================================================================"
echo "Complete! End time: $(date)"
echo "================================================================"
echo ""
echo "RESULTS:"
echo "  CSV:    ${OUTPUT_DIR}/lstv_uncertainty_metrics.csv"
echo "  Report: ${OUTPUT_DIR}/report.html"
echo ""
echo "Next: sbatch slurm_scripts/03_prod_inference.sh"
echo "================================================================"
