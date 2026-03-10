#!/bin/bash
# =============================================================================
# 06_visualize_3d.sh  —  LSTV 3D Visualizer with Pathology-Based Ranking (v3.2)
# =============================================================================
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

# ── Configuration ──────────────────────────────────────────────────────────────
STUDY_ID=""          # single study — leave empty for batch
RANK_BY=lstv         # "lstv" = rank by LSTV pathology score (requires lstv_results.json)
                     # "all"  = render every study (may be very slow for 283 spines)
TOP_N=10             # N most-pathologic studies to render
TOP_NORMAL=1         # N most-normal studies to render (score=0)
SMOOTH=2.0           # Gaussian sigma for marching cubes surfaces
NO_TSS=false         # skip TotalSpineSeg label rendering
IAN_PAN_COORDS=""    # path to ian_pan_disc_coords.json — leave empty to auto-detect
                     # e.g. results/ian_pan_disc_coords/ian_pan_disc_coords.json
                     # When provided, all 5 Ian Pan disc-level peaks are rendered
                     # as coloured diamond markers on each 3D HTML (focused view).

# ── Pathology score breakdown (lstv_engine.compute_lstv_pathology_score) ───────
#
#  Castellvi IV=5  III=4  II=3  I=1
#  Phenotype (sacralization/lumbarization, high confidence) +3
#  Lumbar count anomaly (≠5)                                +2
#  Disc below TV severely reduced (<50% DHI)               +2
#  TV body sacral-like morphology                          +2
#  Vertebral angles: delta ≤8.5°                           +3
#  Vertebral angles: C ≤35.5°                              +1.5
#  Rib anomaly                                             +1
#
# TOP_N=10 renders 10 highest-scoring cases (most pathologic LSTV)
# TOP_NORMAL=1 renders 1 lowest-scoring (score=0) for comparison
# ─────────────────────────────────────────────────────────────────────────────

echo "=============================================================="
echo "LSTV 3D VISUALIZATION (v3.2 — Ian Pan disc markers)"
echo "RANK_BY=$RANK_BY  TOP_N=$TOP_N  TOP_NORMAL=$TOP_NORMAL  SMOOTH=$SMOOTH"
echo "IAN_PAN_COORDS=${IAN_PAN_COORDS:-<auto-detect>}"
echo "Job: $SLURM_JOB_ID  |  Start: $(date)"
echo "=============================================================="

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR" logs results/lstv_3d

PROJECT_DIR="$(pwd)"
LSTV_JSON="${PROJECT_DIR}/results/lstv_detection/lstv_results.json"

CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"
[[ ! -f "$IMG_PATH" ]] && singularity pull "$IMG_PATH" "$CONTAINER"

# ── Auto-detect Ian Pan coords if not set ─────────────────────────────────────
_DEFAULT_IAN_PAN="${PROJECT_DIR}/results/ian_pan_disc_coords/ian_pan_disc_coords.json"
if [[ -z "$IAN_PAN_COORDS" && -f "$_DEFAULT_IAN_PAN" ]]; then
    IAN_PAN_COORDS="$_DEFAULT_IAN_PAN"
    echo "Auto-detected Ian Pan coords: $IAN_PAN_COORDS"
fi

# ── Pre-flight checks ──────────────────────────────────────────────────────────
if [[ -z "$STUDY_ID" && "$RANK_BY" == "lstv" && ! -f "$LSTV_JSON" ]]; then
    echo "ERROR: --rank_by lstv requires results/lstv_detection/lstv_results.json"
    echo "  Run 04_lstv_detection.sh first"
    exit 1
fi

# ── Argument assembly ──────────────────────────────────────────────────────────
# Fixed args always passed
ARGS=(
    "--spineps_dir"    "/work/results/spineps"
    "--totalspine_dir" "/work/results/totalspineseg"
    "--output_dir"     "/work/results/lstv_3d"
    "--smooth"         "$SMOOTH"
)

# Mode-specific args
if [[ -n "$STUDY_ID" ]]; then
    ARGS+=("--study_id" "$STUDY_ID")
    # Still pass lstv_json if available so the single study gets its result data
    [[ -f "$LSTV_JSON" ]] && ARGS+=("--lstv_json" "/work/results/lstv_detection/lstv_results.json")
elif [[ "$RANK_BY" == "all" ]]; then
    ARGS+=("--all")
    [[ -f "$LSTV_JSON" ]] && ARGS+=("--lstv_json" "/work/results/lstv_detection/lstv_results.json")
else
    # Default: rank by lstv pathology score
    ARGS+=(
        "--rank_by"    "lstv"
        "--top_n"      "$TOP_N"
        "--top_normal" "$TOP_NORMAL"
        "--lstv_json"  "/work/results/lstv_detection/lstv_results.json"
    )
fi

[[ "$NO_TSS" == "true" ]] && ARGS+=("--no_tss")

# Ian Pan disc coords — pass container-relative path
[[ -n "$IAN_PAN_COORDS" ]] && ARGS+=("--ian_pan_coords" "/work/results/ian_pan_disc_coords/ian_pan_disc_coords.json")

# ── Run ────────────────────────────────────────────────────────────────────────
singularity exec \
    --bind "${PROJECT_DIR}:/work" \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/06_visualize_3d.py "${ARGS[@]}"

EXIT_CODE=$?

echo "=============================================================="
echo "3D visualization complete | End: $(date)"
echo ""
echo "Outputs: results/lstv_3d/*_lstv_3d.html"
echo ""
echo "Each HTML shows:"
echo "  • Phenotype banner (SACRALIZATION / LUMBARIZATION / TRANSITIONAL / NORMAL)"
echo "  • Castellvi type + TP height rulers"
echo "  • TV body H/AP ratio with shape classification"
echo "  • Adjacent disc DHI (above / below TV)"
echo "  • Lumbar count (4 / 5 / 6) anomaly flag"
echo "  • Vertebral angles A/B/C/D/D1/δ (Seilanian Toosi 2025)"
echo "  • Rib anomaly flag"
echo "  • Classification rationale panel"
echo "  • Bayesian probability model"
echo "  • Surgical relevance / wrong-level risk"
echo "  • Dynamic clinical narrative"
[[ -n "$IAN_PAN_COORDS" ]] && echo "  • Ian Pan disc-level peak markers (all 5 levels, coloured by confidence)"
[[ -n "$IAN_PAN_COORDS" ]] && echo "  • Ian Pan sequence-vote summary badge (H0/H1/neutral)"
echo "=============================================================="

exit $EXIT_CODE
