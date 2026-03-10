#!/bin/bash
# =============================================================================
# 00b_ian_pan_disc_coords.sh  —  Ian Pan Disc Coordinate Extractor
# =============================================================================
#
# PURPOSE
# -------
# Extends Ian Pan epistemic uncertainty inference to extract 3D disc-level
# world coordinates from the sagittal T2 mid-slice heatmaps.
#
# For each study, per disc level (L1/L2 → L5/S1):
#   • Peak probability (confidence that disc is localised)
#   • Heatmap entropy (uncertainty)
#   • 2D argmax → original DICOM pixel coordinates (reverse the 160×160 resize)
#   • DICOM pixel → LPS world coords (ImagePositionPatient + IOP transform)
#   • LPS → RAS (NIfTI-compatible frame, for direct comparison with TSS/SPINEPS)
#   • Distance: Ian Pan disc peak ↔ TSS disc label centroid (if segs available)
#
# OUTPUTS
# -------
# results/ian_pan_disc_coords/
#   ian_pan_disc_coords.json         — per-study records  (primary output)
#   ian_pan_disc_coords.csv          — wide CSV (one row per study)
#   ian_pan_disc_per_level.csv       — long CSV (one row per study × level)
#   progress_coords.json             — resume checkpoint
#
# DEPENDENCIES
# ------------
# • Same Singularity container as 00_ian_pan_inference.sh
# • models/point_net_checkpoint.pth
# • models/valid_id.npy
# • data/raw/train_series_descriptions.csv
# • data/raw/train_images/  (DICOM root)
# • results/totalspineseg/  (optional — for TSS distance metrics)
#   Recommend running AFTER step 3 (TotalSpineSeg) to get the TSS distances,
#   but can also run standalone without them (just skips distance computation).
#
# PIPELINE POSITION
# -----------------
# Can run in parallel with everything else — only needs DICOMs + weights.
# The output JSON is consumed by:
#   04_detect_lstv.py       — disc-level confidence as alignment tiebreaker
#   06_visualize_3d.py      — overlay Ian Pan disc peaks on 3D renders
#
# USAGE
# -----
# Standalone (trial):
#   sbatch slurm_scripts/00b_ian_pan_disc_coords.sh
#
# After segmentations (with TSS distances):
#   MODE=prod sbatch --dependency=afterok:<TSS_JOB_ID> \
#       slurm_scripts/00b_ian_pan_disc_coords.sh
#
# Retry failures:
#   RETRY_FAILED=true MODE=prod sbatch slurm_scripts/00b_ian_pan_disc_coords.sh
#
# With segmentation comparison enabled (recommended for final run):
#   WITH_SEGS=true MODE=prod sbatch slurm_scripts/00b_ian_pan_disc_coords.sh
#
# =============================================================================
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --job-name=ip_disc_coords
#SBATCH -o logs/ip_disc_coords_%j.out
#SBATCH -e logs/ip_disc_coords_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Runtime configuration ─────────────────────────────────────────────────────
MODE=${MODE:-prod}                # trial | debug | prod
RETRY_FAILED=${RETRY_FAILED:-false}
TRIAL_SIZE=${TRIAL_SIZE:-3}
WITH_SEGS=${WITH_SEGS:-true}      # true = pass --spineps_dir / --totalspine_dir
                                   # (requires step 2+3 to have run already)
DEBUG_STUDY_ID=${DEBUG_STUDY_ID:-}
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "IAN PAN DISC COORDINATE EXTRACTOR"
echo "Mode:         $MODE"
echo "Retry failed: $RETRY_FAILED"
echo "With segs:    $WITH_SEGS  (TSS distance metrics)"
echo "Job ID:       $SLURM_JOB_ID"
echo "GPU:          ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start:        $(date)"
echo "================================================================"

nvidia-smi

# ── Singularity temp setup ────────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="$SINGULARITY_TMPDIR/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$SINGULARITY_TMPDIR" "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

# ── Environment ───────────────────────────────────────────────────────────────
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(pwd)"
DICOM_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
OUTPUT_DIR="${PROJECT_DIR}/results/ian_pan_disc_coords"
MODELS_DIR="${PROJECT_DIR}/models"
SPINEPS_DIR="${PROJECT_DIR}/results/spineps"
TSS_DIR="${PROJECT_DIR}/results/totalspineseg"

mkdir -p logs "$OUTPUT_DIR"

# ── Preflight checks ──────────────────────────────────────────────────────────
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
if [[ ! -f "$SERIES_CSV" ]]; then
    echo "ERROR: series CSV not found: $SERIES_CSV"
    exit 1
fi

N_STUDIES=$(ls -d "${DICOM_DIR}"/*/ 2>/dev/null | wc -l)
echo "Studies in DICOM dir: $N_STUDIES"

# ── Container ─────────────────────────────────────────────────────────────────
CONTAINER="docker://go2432/lstv-uncertainty:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/lstv-uncertainty.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi
echo "Container: $IMG_PATH"

# ── Argument assembly ─────────────────────────────────────────────────────────
ARGS=(
    "--input_dir"   "/data/input"
    "--series_csv"  "/data/raw/train_series_descriptions.csv"
    "--output_dir"  "/data/output"
    "--checkpoint"  "/app/models/point_net_checkpoint.pth"
    "--valid_ids"   "/app/models/valid_id.npy"
    "--mode"        "$MODE"
    "--trial_size"  "${TRIAL_SIZE}"
)

if [[ "$RETRY_FAILED" == "true" ]]; then
    ARGS+=("--retry_failed")
fi

if [[ -n "$DEBUG_STUDY_ID" ]]; then
    ARGS+=("--debug_study_id" "$DEBUG_STUDY_ID")
fi

# Add segmentation dirs if WITH_SEGS=true and they exist
if [[ "$WITH_SEGS" == "true" ]]; then
    if [[ -d "$SPINEPS_DIR" ]]; then
        ARGS+=("--spineps_dir" "/work/results/spineps")
        echo "Segmentation comparison: SPINEPS enabled"
    else
        echo "WARNING: WITH_SEGS=true but SPINEPS dir not found: $SPINEPS_DIR"
    fi
    if [[ -d "$TSS_DIR" ]]; then
        ARGS+=("--totalspine_dir" "/work/results/totalspineseg")
        echo "Segmentation comparison: TotalSpineSeg enabled"
    else
        echo "WARNING: WITH_SEGS=true but TotalSpineSeg dir not found: $TSS_DIR"
    fi
fi

echo "================================================================"
echo "Arguments: ${ARGS[*]}"
echo "================================================================"

# ── Run ───────────────────────────────────────────────────────────────────────
singularity exec --nv \
    --bind "${PROJECT_DIR}:/work" \
    --bind "${DICOM_DIR}:/data/input" \
    --bind "${OUTPUT_DIR}:/data/output" \
    --bind "${MODELS_DIR}:/app/models" \
    --bind "$(dirname $SERIES_CSV):/data/raw" \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/ian_pan_disc_coords.py \
        "${ARGS[@]}"

EXIT_CODE=$?

echo "================================================================"
echo "Disc coordinate extraction complete"
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"
echo ""
echo "Outputs:"
echo "  ${OUTPUT_DIR}/ian_pan_disc_coords.json     <- per-study disc coords"
echo "  ${OUTPUT_DIR}/ian_pan_disc_coords.csv      <- wide CSV"
echo "  ${OUTPUT_DIR}/ian_pan_disc_per_level.csv   <- long CSV"
echo ""
echo "Next steps:"
echo "  1. Inspect distances: column dist_to_tss_mm in ian_pan_disc_per_level.csv"
echo "     Distances > 15mm at l4_l5 or l5_s1 signal labeling disagreement."
echo ""
echo "  2. Pass to LSTV detection:"
echo "     In 04_detect_lstv.py, load via:"
echo "       from ian_pan_disc_coords import load_ian_pan_disc_coords"
echo "       ip = load_ian_pan_disc_coords('results/ian_pan_disc_coords/ian_pan_disc_coords.json')"
echo ""
echo "  3. Add to full pipeline orchestrator:"
echo "     JOB_IP=\$(sbatch --parsable --dependency=afterok:\$JOB3 \\"
echo "         --export=ALL,MODE=prod,WITH_SEGS=true \\"
echo "         slurm_scripts/00b_ian_pan_disc_coords.sh)"
echo "     Then pass to step 4: --dependency=afterok:\${JOB2}:\${JOB3}:\${JOB_IP}"
echo "================================================================"

exit $EXIT_CODE
