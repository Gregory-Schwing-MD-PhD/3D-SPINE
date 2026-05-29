#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=168:00:00
#SBATCH --job-name=lstv_full_dataset
#SBATCH -o logs/lstv_full_dataset_%j.out
#SBATCH -e logs/lstv_full_dataset_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# LSTV FULL-DATASET PIPELINE — processes every study in valid_id.npy
#
# Step 0: Ian-Pan uncertainty inference   (GPU)      [no dependencies]
# Step 1: DICOM → NIfTI conversion        (CPU)      [no dependencies]
# Step 2: SPINEPS segmentation — ALL      (GPU)      [after NIfTI]
# Step 3: TotalSpineSeg — ALL             (GPU)      [after NIfTI, parallel with SPINEPS]
# Step 4: LSTV detection — ALL            (CPU)      [after SPINEPS + TotalSpineSeg]
# Step 5: Morphometrics — ALL             (CPU)      [after LSTV detection]
# Step 6: 3D visualization — pathologic   (CPU)      [after morphometrics]  ─┐ parallel
#          3D visualization — normal      (CPU)      [after morphometrics]  ─┘
# Step 7: Dataset summary HTML report     (CPU)      [after morphometrics]
#
# 3D visualization targets:
#   - Top N_VIZ_PATHOLOGIC most pathologic studies (ranked by pathology_burden)
#   - Top 1 most normal study (ranked by normality_score)
#
# Skips studies that are already done at each stage (safe to re-run).
# ══════════════════════════════════════════════════════════════════════════════

# ── Configuration ─────────────────────────────────────────────────────────────
N_VIZ_PATHOLOGIC=5   # number of most-pathologic studies to render in 3D
N_VIZ_NORMAL=1       # number of most-normal studies to render in 3D
# Pathology score weights: canal stenosis, cord MSCC, DHI per level,
# spondylolisthesis, vertebral wedge fracture, Baastrup, facet tropism,
# LFT hypertrophy, Castellvi grade.  See scripts/pathology_score.py.
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "LSTV FULL-DATASET PIPELINE"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "3D viz: top $N_VIZ_PATHOLOGIC pathologic + $N_VIZ_NORMAL normal (by pathology burden score)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME

PROJECT_DIR="$(pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

# ─── Preflight checks ─────────────────────────────────────────────────────────
if [[ ! -d "data/raw/train_images" ]] || \
   [[ $(ls -1 data/raw/train_images 2>/dev/null | wc -l) -eq 0 ]]; then
    echo "ERROR: data/raw/train_images is empty or missing."
    echo "  Download data first with: sbatch slurm_scripts/00_download_data.sh"
    exit 1
fi

if [[ ! -f "models/valid_id.npy" ]]; then
    echo "ERROR: models/valid_id.npy not found."
    exit 1
fi

if [[ ! -f "models/point_net_checkpoint.pth" ]]; then
    echo "ERROR: models/point_net_checkpoint.pth not found."
    exit 1
fi

# Count valid IDs without depending on numpy (the orchestrator runs in the
# `nextflow` conda env, which has no numpy — the real work runs in containers).
# Parse the .npy header directly; fall back to "?" so this can never abort the
# job under `set -euo pipefail`. N_VALID is informational only (echoed below).
N_VALID=$(python3 -c "
import ast, struct, sys
try:
    with open('models/valid_id.npy', 'rb') as f:
        assert f.read(6) == b'\x93NUMPY', 'not a .npy file'
        major = f.read(2)[0]
        hlen = struct.unpack('<I' if major >= 2 else '<H', f.read(4 if major >= 2 else 2))[0]
        hdr = ast.literal_eval(f.read(hlen).decode('latin1'))
        print(hdr['shape'][0])
except Exception as e:
    sys.stderr.write('WARN: could not count valid_id.npy (%s)\n' % e)
    print('?')
" 2>>logs/lstv_full_dataset_${SLURM_JOB_ID}.err || echo "?")
echo "Valid study IDs: $N_VALID"
echo ""
echo "Pipeline will submit:"
echo "  Step 0: Ian-Pan inference (GPU, ~1-2h)         [no dependencies]"
echo "  Step 1: DICOM → NIfTI (CPU, ~2-8h)             [no dependencies]"
echo "  Step 2: SPINEPS — ALL $N_VALID studies (GPU)   [after NIfTI]"
echo "  Step 3: TotalSpineSeg — ALL $N_VALID (GPU)     [after NIfTI, parallel with SPINEPS]"
echo "  Step 4: LSTV detection — ALL (CPU)              [after SPINEPS + TotalSpineSeg]"
echo "  Step 5: Morphometrics — ALL $N_VALID (CPU)     [after LSTV detection]"
echo "  Step 6: 3D viz — top $N_VIZ_PATHOLOGIC pathologic + $N_VIZ_NORMAL normal (CPU) [after morphometrics]"
echo "  Step 7: Dataset summary HTML report (CPU)       [after morphometrics]"
echo ""

# ─── Step 0: Ian-Pan inference (no dependency — runs immediately) ─────────────
echo "Submitting: Step 0 — Ian-Pan inference"
JOB0=$(sbatch --parsable slurm_scripts/00_ian_pan_inference.sh)
echo "  Job ID: $JOB0"

# ─── Step 1: DICOM → NIfTI (no dependency — runs immediately in parallel) ─────
echo ""
echo "Submitting: Step 1 — DICOM → NIfTI conversion"
JOB1=$(sbatch --parsable \
    --job-name=dicom_nifti \
    slurm_scripts/01_dicom_to_nifti.sh)
echo "  Job ID: $JOB1 (CPU, running in parallel with Ian-Pan)"

# ─── Step 2: SPINEPS — ALL (depends on NIfTI) ────────────────────────────────
echo ""
echo "Submitting: Step 2 — SPINEPS (ALL valid studies)"
JOB2=$(sbatch --parsable \
    --job-name=spineps_all \
    --time=96:00:00 \
    --dependency=afterok:$JOB1 \
    --export=ALL,MODE=all \
    slurm_scripts/02b_spineps_selective.sh)
echo "  Job ID: $JOB2 (GPU, after NIfTI)"

# ─── Step 3: TotalSpineSeg — ALL (depends on NIfTI, parallel with SPINEPS) ───
echo ""
echo "Submitting: Step 3 — TotalSpineSeg (ALL valid studies)"
JOB3=$(sbatch --parsable \
    --job-name=tss_all \
    --time=96:00:00 \
    --dependency=afterok:$JOB1 \
    --export=ALL,MODE=all \
    slurm_scripts/03b_totalspineseg_selective.sh)
echo "  Job ID: $JOB3 (GPU, after NIfTI, parallel with SPINEPS)"

# ─── Step 4: LSTV detection — ALL (depends on SPINEPS + TotalSpineSeg) ────────
echo ""
echo "Submitting: Step 4 — LSTV detection (ALL)"
JOB4=$(sbatch --parsable \
    --job-name=lstv_detect_all \
    --time=4:00:00 \
    --dependency=afterok:${JOB2}:${JOB3} \
    --export=ALL,ALL=true \
    slurm_scripts/04_lstv_detection.sh)
echo "  Job ID: $JOB4 (CPU, after SPINEPS + TotalSpineSeg)"

# ─── Step 5: Morphometrics — ALL (depends on LSTV detection) ──────────────────
# LSTV results are fed in so reports get Castellvi annotations.
echo ""
echo "Submitting: Step 5 — Morphometrics (ALL)"
JOB5=$(sbatch --parsable \
    --job-name=morpho_all \
    --time=24:00:00 \
    --dependency=afterok:${JOB4} \
    --export=ALL,ALL=true \
    slurm_scripts/05_morphometrics.sh)
echo "  Job ID: $JOB5 (CPU, after LSTV detection)"

# ─── Step 6: 3D visualization — pathologic + normal (depends on morphometrics) ─
#
# A single job using --rank_by morpho, which reads morphometrics_all.json and
# computes a pathology burden score for every study, then renders:
#   - top N_VIZ_PATHOLOGIC studies  (highest burden score)
#   - top N_VIZ_NORMAL studies      (lowest burden score, score >= 0)
#
# The pathology score is a weighted sum across independent pathology domains
# so the selected studies cover a range of pathology types rather than being
# dominated by one extreme metric.  See scripts/pathology_score.py for weights.
#
# The morphometrics_all.json and lstv_results.json are passed so the renderer
# annotates surfaces with Castellvi grade, measurement rulers, and the score.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "Submitting: Step 6 — 3D visualization (morpho-ranked: $N_VIZ_PATHOLOGIC pathologic + $N_VIZ_NORMAL normal)"
JOB6=$(sbatch --parsable \
    --job-name=viz3d_morpho \
    --time=6:00:00 \
    --dependency=afterok:${JOB5} \
    --export=ALL,ALL=false,RANK_BY=morpho,TOP_N=${N_VIZ_PATHOLOGIC},TOP_NORMAL=${N_VIZ_NORMAL},SMOOTH=3,NO_TSS=false \
    slurm_scripts/06_visualize_3d.sh)
echo "  Job ID: $JOB6 (CPU, morpho-ranked selection)"

# ─── Step 7: Dataset summary report (depends on morphometrics) ───────────────
echo ""
echo "Submitting: Step 7 — Dataset summary HTML report"
JOB7=$(sbatch --parsable \
    --job-name=dataset_report \
    --time=1:00:00 \
    --dependency=afterok:${JOB5} \
    slurm_scripts/06_html_report.sh)
echo "  Job ID: $JOB7 (CPU, after morphometrics)"

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "ALL JOBS SUBMITTED"
echo "================================================================"
echo ""
echo "Dependency chain:"
echo "  $JOB0  Ian-Pan inference        (GPU)  ─── (no blocking deps)"
echo "  $JOB1  DICOM → NIfTI            (CPU)  ─── (runs immediately in parallel)"
echo "    ├─→  $JOB2  SPINEPS ALL       (GPU)  ─── after NIfTI"
echo "    └─→  $JOB3  TotalSpineSeg ALL (GPU)  ─── after NIfTI"
echo "              ↓ (both GPU steps done)"
echo "         $JOB4  LSTV Detection ALL (CPU)"
echo "              ↓"
echo "         $JOB5  Morphometrics ALL  (CPU)"
echo "           ├──→ $JOB6   3D viz morpho-ranked (CPU, top $N_VIZ_PATHOLOGIC pathologic + $N_VIZ_NORMAL normal)"
echo "           └──→ $JOB7   Dataset report    (CPU)"
echo ""
echo "Monitor:"
echo "  squeue -u $USER"
echo "  watch -n 60 squeue -u $USER"
echo ""
echo "Logs:"
echo "  logs/lstv_full_dataset_${SLURM_JOB_ID}.out  ← this orchestrator"
echo "  logs/dicom_nifti_${JOB1}.out"
echo "  logs/spineps_all_${JOB2}.out"
echo "  logs/totalspineseg_selective_${JOB3}.out"
echo "  logs/lstv_detect_${JOB4}.out"
echo "  logs/spine_morpho_${JOB5}.out"
echo "  logs/lstv_3d_${JOB6}.out  ← morpho-ranked viz"
echo "  logs/lstv_report_${JOB7}.out"
echo ""
echo "Expected runtime: 48-96h (depending on GPU queue)"
echo ""
echo "Final outputs:"
echo "  results/lstv_detection/lstv_results.json"
echo "  results/morphometrics/morphometrics_all.json"
echo "  results/morphometrics/morphometrics_all.csv"
echo "  results/morphometrics/morphometrics_summary.json"
echo "  results/morphometrics/reports/*.html          ← per-study clinical reports"
echo "  results/lstv_3d/                              ← 3D renders (pathologic + normal)"
echo "  results/dataset_morphometrics_report.html     ← dataset summary"
echo "  results/lstv_report.html                      ← LSTV classification"
echo "================================================================"

# ─── Wait and monitor ─────────────────────────────────────────────────────────
echo ""
echo "Master job now monitoring progress..."
echo ""

ALL_JOBS=("$JOB0" "$JOB1" "$JOB2" "$JOB3" "$JOB4" "$JOB5" "$JOB6" "$JOB7")
JOB_NAMES=("Ian-Pan" "DICOM-NIfTI" "SPINEPS-ALL" "TotalSpineSeg-ALL" "LSTV-Detection" "Morphometrics-ALL" "3D-Viz-Morpho" "Dataset-Report")

for i in "${!ALL_JOBS[@]}"; do
    job_id="${ALL_JOBS[$i]}"
    name="${JOB_NAMES[$i]}"
    echo "Waiting for job $job_id ($name)..."

    while squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"; do
        sleep 120
    done

    STATE=$(sacct -j "$job_id" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$STATE" == "COMPLETED" ]]; then
        echo "  ✓ $job_id ($name) — COMPLETED"
    else
        echo "  ✗ $job_id ($name) — $STATE"
        echo ""
        echo "Check logs:"
        echo "  ls -ltr logs/ | grep $job_id"
        echo "  cat logs/*${job_id}*.err"
        exit 1
    fi
done

echo ""
echo "================================================================"
echo "FULL DATASET PIPELINE COMPLETE!"
echo "$(date)"
echo "================================================================"
echo ""
echo "Key outputs:"
echo "  results/lstv_detection/lstv_results.json"
echo "  results/morphometrics/morphometrics_all.csv"
echo "  results/lstv_3d/                    ← 3D renders"
echo "  results/dataset_morphometrics_report.html"
echo "  results/lstv_report.html"
