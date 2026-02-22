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
TOP_N=1                    # studies from each end — must match 02b + 03b + detect settings
RANK_BY=l5_s1_confidence   # column to rank by — must match all upstream settings
ALL=false                  # set to true to render every study with SPINEPS segmentation
SMOOTH=3                   # Gaussian pre-smoothing sigma for marching cubes surfaces
NO_TSS=false               # set to true to skip TotalSpineSeg label rendering

# ── Morphometric output flags ─────────────────────────────────────────────────
# Save all computed morphometrics (DHI, canal stenosis, facet tropism, vertebral
# height ratios, spondylolisthesis, ligamentum flavum, Baastrup, foraminal volume,
# cord CSA/MSCC) to a single CSV file at results/lstv_3d/morphometrics_all_studies.csv
# This CSV can be used for downstream statistical analysis or ML feature extraction.
SAVE_MORPHOMETRICS_CSV=true

# ── Morphometric thresholds reference (informational — not editable here) ─────
#
# VERTEBRAL BODY (Genant/QM):
#   Compression Hm/Ha or Hm/Hp < 0.80 → biconcave fracture
#   Wedge       Ha/Hp             < 0.80 → anterior wedge fracture
#   Crush       Hp/Ha             < 0.80 → posterior height loss
#   Intervention threshold        < 0.75 → moderate/severe fracture
#
# DISC HEIGHT INDEX (DHI, Farfan method):
#   DHI = (Ha+Hp)/(Ds+Di) × 100
#   < 50% → Severe (Pfirrmann V equivalent, intervention threshold)
#   < 70% → Moderate
#   < 85% → Mild
#
# CENTRAL CANAL STENOSIS:
#   DSCA: Normal >100mm²  Relative 75-100mm²  Absolute <70-75mm²
#   AP:   Normal >12mm    Relative 10-12mm    Absolute <7-10mm
#   Critical threshold (Indian population): 11.13mm
#
# LATERAL RECESS:
#   LRD (Lateral Recess Depth) ≤ 3mm → stenosis
#   Lateral Recess Height      ≤ 2mm → stenosis
#
# NEURAL FORAMINAL (Lee grade):
#   Grade 0=Normal  Grade 1=Mild  Grade 2=Moderate  Grade 3=Severe (intervention)
#   Volume norms: L1/L2~580mm³  L2/L3~700mm³  L3/L4~770mm³
#                 L4/L5~800mm³  L5/S1~824mm³
#
# LIGAMENTUM FLAVUM:
#   Normal LFT baseline:    3.5mm at L4-L5
#   Hypertrophy threshold:  ≥ 4.0mm (varies by study)
#   Significant encroachment: > 5mm
#   LFA optimal canal stenosis predictor cutoff: 105.90mm²
#
# BAASTRUP DISEASE (Kissing Spine):
#   Inter-spinous gap ≤ 0mm → contact (sclerosis/bursitis risk)
#   Inter-spinous gap ≤ 2mm → Baastrup risk zone
#   Incidence in symptomatic >80 yrs: 81%
#
# FACET TROPISM (Ko et al. grade):
#   Grade 0: ≤ 7°    (normal asymmetry)
#   Grade 1: 7–10°   (moderate — disc prolapse risk)
#   Grade 2: ≥ 10°   (severe — spondylolisthesis risk)
#   Literature thresholds vary: 7°, 8°, 10°
#
# SPONDYLOLISTHESIS:
#   Sagittal translation ≥ 3mm → degenerative spondylolisthesis
#   Normal: < 2–4mm
#
# CORD METRICS (cervical / thoracic):
#   MSCC proxy: Cord AP / Canal AP  (normalized compression index)
#   CSA: cord cross-sectional area in mm²
#   K-line negative: anterior contact → surgical planning indicator
#
# PFIRRMANN GRADING (IVD, T2-weighted):
#   Grade I   = Clear nucleus/annulus, hyperintense, normal height
#   Grade III = Unclear distinction, intermediate signal
#   Grade V   = Lost distinction, hypointense (black), collapsed
#
# MODIC CHANGE BURDEN (MCG / Modic-Udby):
#   Grade A (Minor): < 25% vertebral body height/volume
#   Grade B (Major): 25–50%
#   Grade C (High):  > 50% — strongest disability predictor
#
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "LSTV 3D VISUALIZER + COMPREHENSIVE MORPHOMETRICS"
echo "STUDY_ID=${STUDY_ID:-<selective/all>}  TOP_N=$TOP_N  RANK_BY=$RANK_BY"
echo "ALL=$ALL  SMOOTH=$SMOOTH  NO_TSS=$NO_TSS"
echo "SAVE_MORPHOMETRICS_CSV=$SAVE_MORPHOMETRICS_CSV"
echo ""
echo "Morphometrics: Vertebral QM · DHI · DSCA · Canal shape · LF ·"
echo "               Spondylolisthesis · Baastrup · Facet tropism ·"
echo "               Foraminal volume · Cord CSA/MSCC · Lee grade"
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
    SELECTION_ARGS+=( "--study_id" "$STUDY_ID" )
    echo "Single-study mode: $STUDY_ID"

elif [[ "$ALL" == "true" ]]; then
    SELECTION_ARGS+=( "--all" )
    echo "ALL mode: rendering every study with SPINEPS segmentations"

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
if [[ "$NO_TSS" == "true" ]]; then
    SELECTION_ARGS+=( "--no_tss" )
    echo "TotalSpineSeg label rendering disabled"
fi

# --- Morphometrics CSV flag ---
if [[ "$SAVE_MORPHOMETRICS_CSV" == "true" ]]; then
    SELECTION_ARGS+=( "--save_morphometrics_csv" )
    echo "Morphometrics CSV export enabled → results/lstv_3d/morphometrics_all_studies.csv"
fi

# --- Annotation from detection results ---
if [[ -f "$LSTV_JSON" ]]; then
    SELECTION_ARGS+=( "--lstv_json" "/work/results/lstv_detection/lstv_results.json" )
    echo "Detection results found — 3D measurements + Castellvi annotations enabled."
else
    echo "WARNING: lstv_results.json not found — run 04_lstv_detection.sh first."
    echo "         3D measurement rulers will still be drawn from seg-spine_msk."
fi

# --- Bind models dir only when valid_ids is needed ---
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
        --smooth         "$SMOOTH" \
        "${SELECTION_ARGS[@]}"

echo "================================================================"
echo "3D visualization + morphometric analysis complete"
echo "HTMLs      → results/lstv_3d/"
echo "Morphometrics CSV → results/lstv_3d/morphometrics_all_studies.csv"
echo ""
echo "Morphometric parameters computed per study:"
echo "  Vertebral: Ha/Hm/Hp heights, Compression/Wedge/Crush ratios, Genant grade"
echo "  Disc:      DHI (Farfan + Method 2), Pfirrmann reference thresholds"
echo "  Canal:     AP diameter, DSCA (mm²), stenosis classification"
echo "  Cord:      CSA (mm²), MSCC proxy, canal occupation ratio"
echo "  LF:        LFT proxy, hypertrophy classification, LFA reference"
echo "  Baastrup:  inter-spinous gaps, contact detection"
echo "  Facet:     tropism angle, Ko grade (0/1/2)"
echo "  Foramen:   elliptical cylinder volume, normative % comparison, Lee equivalent"
echo "  Spondy:    sagittal translation per level vs 3mm threshold"
echo "End: $(date)"
echo "================================================================"
