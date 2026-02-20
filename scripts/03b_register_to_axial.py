#!/usr/bin/env python3
"""
03b_register_to_axial.py
========================
Register sagittal T2w → axial T2w, then resample SPINEPS and TotalSpineSeg
segmentations into axial space using the recovered transform.

Why
---
SPINEPS runs on sagittal T2w and produces TP labels in sagittal voxel space.
TotalSpineSeg also runs on sagittal T2w. The LSTV detector needs to measure
distances between SPINEPS TP labels and the TSS sacrum, and the visualizer
needs to show TPs in the axial plane (where they are anatomically meaningful).

Bringing everything into a single axial reference frame:
  - eliminates inter-volume coordinate errors
  - allows proper axial-plane TP visualization
  - enables world-space distance measurements without affine hacks

Pipeline
--------
  For each study:
  1. Load axial T2w (reference / fixed image)
  2. Load sagittal T2w (moving image)
  3. Rigid registration: moving → fixed  (Mattes MI, 3-level pyramid)
  4. Resample SPINEPS seg with the transform (nearest-neighbour)
  5. Resample TSS labeled seg with the transform (nearest-neighbour)
  6. Resample TSS uncertainty map with the transform (linear)
  7. Save transform (.tfm), resampled segs, and a preview NIfTI for QC

Inputs
------
NIfTI layout (produced by 01_preprocess.py):
  nifti_dir/{study_id}/{series_id}/sub-{study_id}_acq-sag_T2w.nii.gz
  nifti_dir/{study_id}/{series_id}/sub-{study_id}_acq-ax_T2w.nii.gz

SPINEPS output:
  spineps_dir/segmentations/{study_id}/{study_id}_seg-spine_msk.nii.gz
  spineps_dir/segmentations/{study_id}/{study_id}_unc.nii.gz

TotalSpineSeg output:
  totalspine_dir/{study_id}/sagittal/{study_id}_sagittal_labeled.nii.gz
  totalspine_dir/{study_id}/sagittal/{study_id}_sagittal_unc.nii.gz

Outputs (all in axial reference space)
-------
  registered_dir/{study_id}/{study_id}_spineps_reg.nii.gz
  registered_dir/{study_id}/{study_id}_tss_reg.nii.gz
  registered_dir/{study_id}/{study_id}_tss_unc_reg.nii.gz
  registered_dir/{study_id}/{study_id}_spineps_unc_reg.nii.gz
  registered_dir/{study_id}/{study_id}_sag_preview.nii.gz
  registered_dir/{study_id}/{study_id}_transform.tfm

Usage
-----
  python 03b_register_to_axial.py \
      --nifti_dir      results/nifti \
      --spineps_dir    results/spineps \
      --totalspine_dir results/totalspineseg \
      --registered_dir results/registered \
      [--study_id      100206310]          # single-study mode
      [--mode          trial|prod]          # trial = first 5 studies
      [--retry-failed]
"""

import argparse
import json
import logging
import traceback
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROGRESS_FILE = 'progress_registration.json'


# ============================================================================
# REGISTRATION CORE
# ============================================================================

def register_sag_to_axial(
    axial_path: Path,
    sag_path: Path,
    transform_path: Path,
) -> sitk.Transform:
    """
    Rigid 3-D registration: sagittal T2w (moving) -> axial T2w (fixed).

    Strategy
    --------
    - Metric   : Mattes Mutual Information (multi-modality safe even though
                 both are T2w, because FOV / contrast differ significantly)
    - Init     : centre-of-geometry alignment (robust to large translations)
    - Optimizer: gradient descent, 3-level pyramid (4x → 2x → 1x)
    - Sampling : 10 % random per iteration (fast, sufficient for MI)
    """
    fixed  = sitk.ReadImage(str(axial_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(sag_path),   sitk.sitkFloat32)

    # Normalise intensities to [0, 1000] — speeds up MI convergence
    fixed  = sitk.RescaleIntensity(fixed,  0, 1000)
    moving = sitk.RescaleIntensity(moving, 0, 1000)

    # Geometry-based initialisation
    init_tx = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetInitialTransform(init_tx, inPlace=False)
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.10)
    reg.SetInterpolator(sitk.sitkLinear)

    # Coarse → medium → fine
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=300,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    transform = reg.Execute(fixed, moving)
    logger.info(
        f"    Registration: metric={reg.GetMetricValue():.5f} | "
        f"{reg.GetOptimizerStopConditionDescription()}"
    )

    sitk.WriteTransform(transform, str(transform_path))
    return transform


def resample_label(
    label_path: Path,
    reference_path: Path,
    transform: sitk.Transform,
    output_path: Path,
) -> bool:
    """Resample a label map (nearest-neighbour; preserves integer labels)."""
    if not label_path.exists():
        logger.warning(f"    Not found, skipping: {label_path.name}")
        return False
    label = sitk.ReadImage(str(label_path), sitk.sitkUInt16)
    ref   = sitk.ReadImage(str(reference_path), sitk.sitkUInt16)
    out   = sitk.Resample(label, ref, transform,
                          sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(output_path))
    return True


def resample_float(
    image_path: Path,
    reference_path: Path,
    transform: sitk.Transform,
    output_path: Path,
) -> bool:
    """Resample a float image (linear interpolation; for T2w / uncertainty)."""
    if not image_path.exists():
        logger.warning(f"    Not found, skipping: {image_path.name}")
        return False
    img = sitk.ReadImage(str(image_path),    sitk.sitkFloat32)
    ref = sitk.ReadImage(str(reference_path), sitk.sitkFloat32)
    out = sitk.Resample(img, ref, transform,
                        sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(output_path))
    return True


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def find_t2w(nifti_dir: Path, study_id: str, acq: str) -> Path | None:
    """
    Scan all series subdirectories for sub-{study_id}_acq-{acq}_T2w.nii.gz.
    Returns first match or None.
    """
    study_dir = nifti_dir / study_id
    if not study_dir.exists():
        return None
    for series_dir in sorted(study_dir.iterdir()):
        p = series_dir / f"sub-{study_id}_acq-{acq}_T2w.nii.gz"
        if p.exists():
            return p
    return None


def find_spineps_seg(spineps_dir: Path, study_id: str) -> Path | None:
    p = spineps_dir / 'segmentations' / study_id / f"{study_id}_seg-spine_msk.nii.gz"
    return p if p.exists() else None


def find_spineps_unc(spineps_dir: Path, study_id: str) -> Path | None:
    p = spineps_dir / 'segmentations' / study_id / f"{study_id}_unc.nii.gz"
    return p if p.exists() else None


def find_tss_seg(totalspine_dir: Path, study_id: str) -> Path | None:
    p = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"
    return p if p.exists() else None


def find_tss_unc(totalspine_dir: Path, study_id: str) -> Path | None:
    p = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_unc.nii.gz"
    return p if p.exists() else None


# ============================================================================
# PER-STUDY
# ============================================================================

def process_study(
    study_id: str,
    nifti_dir: Path,
    spineps_dir: Path,
    totalspine_dir: Path,
    registered_dir: Path,
) -> bool:
    out = registered_dir / study_id
    out.mkdir(parents=True, exist_ok=True)

    # Locate T2w volumes
    axial_path = find_t2w(nifti_dir, study_id, 'ax')
    sag_path   = find_t2w(nifti_dir, study_id, 'sag')

    if axial_path is None:
        raise FileNotFoundError(f"Axial T2w not found for {study_id}")
    if sag_path is None:
        raise FileNotFoundError(f"Sagittal T2w not found for {study_id}")

    logger.info(f"  [{study_id}] axial : {axial_path.parent.name}")
    logger.info(f"  [{study_id}] sagitt: {sag_path.parent.name}")

    # --- Registration --------------------------------------------------------
    transform_path = out / f"{study_id}_transform.tfm"
    logger.info(f"  [{study_id}] Registering sag -> ax ...")
    transform = register_sag_to_axial(axial_path, sag_path, transform_path)

    # --- Resample label maps -------------------------------------------------
    spineps_seg = find_spineps_seg(spineps_dir, study_id)
    if spineps_seg:
        ok = resample_label(
            spineps_seg, axial_path, transform,
            out / f"{study_id}_spineps_reg.nii.gz",
        )
        logger.info(f"  [{study_id}] SPINEPS seg  -> {'OK' if ok else 'SKIPPED'}")

    tss_seg = find_tss_seg(totalspine_dir, study_id)
    if tss_seg:
        ok = resample_label(
            tss_seg, axial_path, transform,
            out / f"{study_id}_tss_reg.nii.gz",
        )
        logger.info(f"  [{study_id}] TSS seg      -> {'OK' if ok else 'SKIPPED'}")

    # --- Resample uncertainty maps -------------------------------------------
    spineps_unc = find_spineps_unc(spineps_dir, study_id)
    if spineps_unc:
        resample_float(
            spineps_unc, axial_path, transform,
            out / f"{study_id}_spineps_unc_reg.nii.gz",
        )

    tss_unc = find_tss_unc(totalspine_dir, study_id)
    if tss_unc:
        resample_float(
            tss_unc, axial_path, transform,
            out / f"{study_id}_tss_unc_reg.nii.gz",
        )

    # --- Preview (registered sagittal T2w for QC) ----------------------------
    resample_float(
        sag_path, axial_path, transform,
        out / f"{study_id}_sag_preview.nii.gz",
    )

    return True


# ============================================================================
# PROGRESS BOOKKEEPING
# ============================================================================

def load_progress(registered_dir: Path):
    pf = registered_dir / PROGRESS_FILE
    if pf.exists():
        with open(pf) as f:
            return json.load(f), pf
    return {'processed': [], 'failed': []}, pf


def save_progress(progress: dict, pf: Path):
    with open(pf, 'w') as f:
        json.dump(progress, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Register sagittal segmentations to axial T2w space'
    )
    parser.add_argument('--nifti_dir',      required=True)
    parser.add_argument('--spineps_dir',    required=True)
    parser.add_argument('--totalspine_dir', required=True)
    parser.add_argument('--registered_dir', required=True)
    parser.add_argument('--study_id',       default=None,
                        help='Process a single study (skips batch logic)')
    parser.add_argument('--mode',           default='prod',
                        choices=['trial', 'prod'],
                        help='trial = first 5 studies only')
    parser.add_argument('--retry-failed',   action='store_true')
    args = parser.parse_args()

    nifti_dir      = Path(args.nifti_dir)
    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    registered_dir = Path(args.registered_dir)
    registered_dir.mkdir(parents=True, exist_ok=True)

    progress, progress_file = load_progress(registered_dir)

    # Build study list
    if args.study_id:
        study_ids = [args.study_id]
    else:
        study_ids = sorted(d.name for d in nifti_dir.iterdir() if d.is_dir())
        if args.mode == 'trial':
            study_ids = study_ids[:5]
        skip = set(progress['processed'])
        if not args.retry_failed:
            skip -= set(progress.get('failed', []))
        study_ids = [s for s in study_ids if s not in skip]

    logger.info(f"Studies to register: {len(study_ids)}")

    success = failed = 0
    for sid in tqdm(study_ids, desc='Registering', unit='study'):
        logger.info(f"\n[{sid}]")
        try:
            process_study(sid, nifti_dir, spineps_dir, totalspine_dir, registered_dir)
            success += 1
            if sid not in progress['processed']:
                progress['processed'].append(sid)
            if sid in progress.get('failed', []):
                progress['failed'].remove(sid)
        except Exception as e:
            logger.error(f"  [{sid}] FAILED: {e}")
            logger.debug(traceback.format_exc())
            failed += 1
            progress.setdefault('failed', []).append(sid)
        save_progress(progress, progress_file)

    logger.info(f"\nDone — success: {success}  failed: {failed}")


if __name__ == '__main__':
    main()
