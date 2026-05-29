#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Submit the download + full pipeline as one dependency chain.
#
# Run this from the repo root on the cluster login node (it only submits jobs):
#   bash slurm_scripts/00_download_and_run.sh
#
#   Step A: 00_download_all.sh   (DICOM + model checkpoint + valid_id.npy)
#   Step B: 00_run_full_dataset.sh  runs only after the download SUCCEEDS,
#           then fans out Ian-Pan / NIfTI / SPINEPS / TSS / detection /
#           morphometrics / viz / report internally.
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root, regardless of where this is invoked from

JOB_DL=$(sbatch --parsable slurm_scripts/00_download_all.sh)
echo "Download job:          $JOB_DL"

JOB_RUN=$(sbatch --parsable --dependency=afterok:$JOB_DL slurm_scripts/00_run_full_dataset.sh)
echo "Pipeline orchestrator: $JOB_RUN  (starts after $JOB_DL completes OK)"

echo ""
echo "Chain submitted:"
echo "  $JOB_DL  download_all"
echo "    ↓ afterok"
echo "  $JOB_RUN  run_full_dataset → (Ian-Pan, NIfTI, SPINEPS, TSS, detection, morpho, viz, report)"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ls -ltr logs/"
