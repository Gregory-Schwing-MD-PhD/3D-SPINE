#!/bin/bash
# =============================================================================
# spineps_single.sh — Run SPINEPS on ONE arbitrary NIfTI, outside the RSNA
#                     orchestrator (no valid_id.npy / series CSV needed).
#
# Usage:
#   INPUT=/abs/path/to/img.nii.gz sbatch slurm_scripts/spineps_single.sh
#
# Optional overrides (env vars):
#   OUTDIR=/abs/path/out            # where to copy the collected masks
#                                   # (default: <input_dir>/spineps_single)
#   MODEL_SEMANTIC=t2w              # semantic model key (t2w|t1w|vibe|...)
#   MODEL_INSTANCE=instance
#   MODEL_LABELING=t2w_labeling
#
# NOTE: SPINEPS semantic models are MRI-trained. Running the t2w model on a CT
#       volume will likely give poor/empty results. Swap MODEL_SEMANTIC if a CT
#       model is available (see the "Available models" listing this script prints).
# =============================================================================
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=spineps_single
#SBATCH -o logs/spineps_single_%j.out
#SBATCH -e logs/spineps_single_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
shopt -s globstar nullglob   # let **/ recurse and unmatched globs vanish

# ── Inputs ──────────────────────────────────────────────────────────────────
INPUT="${INPUT:?Set INPUT=/abs/path/to/img.nii.gz}"
if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: INPUT not found: $INPUT"
    exit 1
fi
INPUT="$(readlink -f "$INPUT")"
INPUT_DIR="$(dirname "$INPUT")"
INPUT_NAME="$(basename "$INPUT")"

MODEL_SEMANTIC="${MODEL_SEMANTIC:-t2w}"
MODEL_INSTANCE="${MODEL_INSTANCE:-instance}"
MODEL_LABELING="${MODEL_LABELING:-t2w_labeling}"
OUTDIR="${OUTDIR:-${INPUT_DIR}/spineps_single}"

echo "================================================================"
echo "SPINEPS — single file"
echo "Input:     $INPUT"
echo "Out dir:   $OUTDIR"
echo "Models:    semantic=$MODEL_SEMANTIC instance=$MODEL_INSTANCE labeling=$MODEL_LABELING"
echo "Job ID:    $SLURM_JOB_ID | GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "Start:     $(date)"
echo "================================================================"

# --- Environment (matches 02b_spineps_selective.sh) ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || echo "WARNING: singularity not found"
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE
export NXF_SINGULARITY_HOME_MOUNT=true

# --- Model paths from the project (same mounts the pipeline uses) ---
PROJECT_DIR="$(pwd)"
MODELS_DIR="${PROJECT_DIR}/models"
SPINEPS_PKG_MODELS="${PROJECT_DIR}/models/spineps_pkg_models"
mkdir -p logs "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR" "$SPINEPS_PKG_MODELS" "$OUTDIR"

# --- Container ---
CONTAINER="docker://go2432/spineps-segmentation:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-segmentation.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

echo "Available semantic models in ${MODELS_DIR}:"
ls -1 "$MODELS_DIR" 2>/dev/null | sed 's/^/  /' || true
echo "----------------------------------------------------------------"

# --- Run SPINEPS on the single file ---
# SPINEPS writes its results to <input_dir>/derivatives_seg/ inside the bind.
singularity exec --nv \
    --bind "${INPUT_DIR}":/indata \
    --bind "${MODELS_DIR}":/app/models \
    --bind "${SPINEPS_PKG_MODELS}":/opt/conda/lib/python3.10/site-packages/spineps/models \
    --env SPINEPS_SEGMENTOR_MODELS=/app/models \
    --env SPINEPS_ENVIRONMENT_DIR=/app/models \
    --env PYTHONUNBUFFERED=1 \
    --pwd /indata \
    "$IMG_PATH" \
    python -m spineps.entrypoint sample \
        -i "/indata/${INPUT_NAME}" \
        -model_semantic  "$MODEL_SEMANTIC" \
        -model_instance  "$MODEL_INSTANCE" \
        -model_labeling  "$MODEL_LABELING" \
        -save_softmax_logits \
        -override_semantic \
        -override_instance \
        -override_ctd

# --- Collect outputs ---
DERIV="${INPUT_DIR}/derivatives_seg"
echo "----------------------------------------------------------------"
if [[ -d "$DERIV" ]]; then
    echo "SPINEPS derivatives at: $DERIV"
    find "$DERIV" -maxdepth 3 -name '*.nii.gz' -o -name '*_ctd.json' 2>/dev/null | sed 's/^/  /'
    # copy the key masks into OUTDIR for convenience
    cp -v "$DERIV"/**/*_seg-vert_msk.nii.gz   "$OUTDIR"/ 2>/dev/null || true
    cp -v "$DERIV"/**/*_seg-spine_msk.nii.gz  "$OUTDIR"/ 2>/dev/null || true
    cp -v "$DERIV"/**/*_seg-subreg_msk.nii.gz "$OUTDIR"/ 2>/dev/null || true
    cp -v "$DERIV"/**/*_ctd.json              "$OUTDIR"/ 2>/dev/null || true
else
    echo "WARNING: no derivatives_seg produced — segmentation likely failed."
    echo "         (Expected for a CT volume with the MRI t2w model.)"
fi

echo "================================================================"
echo "Done | End: $(date)"
echo "Raw outputs:  $DERIV"
echo "Collected:    $OUTDIR"
echo "================================================================"
