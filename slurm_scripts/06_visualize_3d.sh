#!/bin/bash
# =============================================================================
# 06_visualize_3d.sh  —  LSTV 3D Visualizer with Pathology-Based Ranking
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
TOP_N=5              # N most-pathologic studies to render
TOP_NORMAL=2         # N most-normal studies to render
SMOOTH=2.0           # Gaussian sigma for marching cubes surfaces
NO_TSS=false         # skip TotalSpineSeg label rendering

# ── Pathology score breakdown (lstv_engine.compute_lstv_pathology_score) ───────
#
#  Castellvi IV=5  III=4  II=3  I=1
#  Phenotype (sacralization/lumbarization, high confidence) +3
#  Lumbar count anomaly (≠5)                                +2
#  Disc below TV severely reduced (<50% DHI)               +2
#  TV body sacral-like morphology                          +2
#  Rib anomaly                                             +1
#
# TOP_N=5 renders 5 highest-scoring cases (most pathologic LSTV)
# TOP_NORMAL=2 renders 2 lowest-scoring (score=0) for comparison
# ─────────────────────────────────────────────────────────────────────────────

echo "=============================================================="
echo "LSTV 3D VISUALIZATION"
echo "RANK_BY=$RANK_BY  TOP_N=$TOP_N  TOP_NORMAL=$TOP_NORMAL  SMOOTH=$SMOOTH"
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

if [[ "$RANK_BY" == "lstv" && ! -f "$LSTV_JSON" ]]; then
    echo "ERROR: --rank_by lstv requires results/lstv_detection/lstv_results.json"
    echo "  Run 04_lstv_detection.sh first"
    exit 1
fi

ARGS=("--smooth" "$SMOOTH")

if [[ -n "$STUDY_ID" ]]; then
    ARGS+=("--study_id" "$STUDY_ID")
elif [[ "$RANK_BY" == "all" ]]; then
    ARGS+=("--all")
else
    ARGS+=(
        "--rank_by"    "$RANK_BY"
        "--top_n"      "$TOP_N"
        "--top_normal" "$TOP_NORMAL"
        "--lstv_json"  "/work/results/lstv_detection/lstv_results.json"
    )
fi

[[ "$NO_TSS" == "true" ]] && ARGS+=("--no_tss")

singularity exec \
    --bind "${PROJECT_DIR}:/work" \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/06_visualize_3d.py \
        --spineps_dir    /work/results/spineps \
        --totalspine_dir /work/results/totalspineseg \
        --output_dir     /work/results/lstv_3d \
        "${ARGS[@]}"

echo "=============================================================="
echo "3D visualization complete | End: $(date)"
echo ""
echo "Outputs: results/lstv_3d/*_lstv_3d.html"
echo ""
echo "Each HTML shows:"
echo "  • Phenotype banner (SACRALIZATION / LUMBARIZATION / TRANSITIONAL / NORMAL)"
echo "  • Castellvi type + TP height rulers (global craniocaudal extent)"
echo "  • TV body H/AP ratio with shape classification"
echo "  • Adjacent disc DHI (above / below TV)"
echo "  • Lumbar count (4 / 5 / 6) anomaly flag"
echo "  • Rib anomaly flag"
echo "  • Classification rationale panel"
echo "=============================================================="
