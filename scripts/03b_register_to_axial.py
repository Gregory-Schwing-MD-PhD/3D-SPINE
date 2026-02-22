#!/usr/bin/env python3
"""
03b_register_to_axial.py — Hybrid Pipeline v2
==============================================
Register sagittal T2w → axial T2w and resample ONLY the SPINEPS
segmentation into axial space.

Study selection mirrors 02b_spineps_selective.py exactly:
  Reads --uncertainty_csv, sorts by --rank_by descending,
  takes top_n highest + top_n lowest, deduplicates.
  Pass --all to process every study in nifti_dir instead.

WHY ONLY SPINEPS?
-----------------
TotalSpineSeg has already been run on the NATIVE axial series, so we have
a native axial TSS segmentation that does not need registration.  SPINEPS
runs only on sagittal T2w, so its TP labels (43/44) live in sagittal space
and must be beamed into the axial coordinate grid via registration.

Usage
-----
  python 03b_register_to_axial.py \
      --nifti_dir       results/nifti \
      --spineps_dir     results/spineps \
      --registered_dir  results/registered \
      --uncertainty_csv results/epistemic_uncertainty/lstv_uncertainty_metrics.csv \
      --valid_ids       models/valid_id.npy \
      --top_n           1 \
      --rank_by         l5_s1_confidence \
      [--all]  [--retry-failed]
"""

import argparse
import json
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROGRESS_FILE = 'progress_registration.json'


# ============================================================================
# STUDY SELECTION  — identical logic to 02b_spineps_selective.py
# ============================================================================

def select_studies(csv_path: Path, top_n: int, rank_by: str,
                   valid_ids: set | None) -> list[str]:
    """
    Sort uncertainty CSV by rank_by descending.
    Return top_n highest + top_n lowest study IDs, deduplicated (top first).
    Mirrors 02b_spineps_selective.py select_studies() exactly.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Uncertainty CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)

    if valid_ids is not None:
        before = len(df)
        df = df[df['study_id'].isin(valid_ids)]
        logger.info(f"Filtered to {len(df)} studies via valid_ids "
                    f"({before - len(df)} excluded)")

    if rank_by not in df.columns:
        raise ValueError(f"Column '{rank_by}' not in CSV. "
                         f"Available: {', '.join(df.columns)}")

    df_sorted  = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    top_ids    = df_sorted.head(top_n)['study_id'].tolist()
    bottom_ids = df_sorted.tail(top_n)['study_id'].tolist()

    seen, selected = set(), []
    for sid in top_ids + bottom_ids:
        if sid not in seen:
            selected.append(sid)
            seen.add(sid)

    logger.info(f"Rank by:          {rank_by}")
    logger.info(f"Top {top_n}:           {top_ids}")
    logger.info(f"Bottom {top_n}:        {bottom_ids}")
    logger.info(f"Total (deduped):  {len(selected)}")
    return selected


# ============================================================================
# REGISTRATION CORE
# ============================================================================

def register_sag_to_axial(axial_path: Path, sag_path: Path,
                           transform_path: Path) -> sitk.Transform:
    fixed  = sitk.ReadImage(str(axial_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(sag_path),   sitk.sitkFloat32)
    fixed  = sitk.RescaleIntensity(fixed,  0, 1000)
    moving = sitk.RescaleIntensity(moving, 0, 1000)

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
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=300,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    transform = reg.Execute(fixed, moving)
    logger.info(f"    metric={reg.GetMetricValue():.5f} | "
                f"{reg.GetOptimizerStopConditionDescription()}")
    sitk.WriteTransform(transform, str(transform_path))
    return transform


def resample_label(label_path, reference_path, transform, output_path) -> bool:
    if not label_path.exists():
        logger.warning(f"    Not found, skipping: {label_path.name}")
        return False
    label = sitk.ReadImage(str(label_path),     sitk.sitkUInt16)
    ref   = sitk.ReadImage(str(reference_path), sitk.sitkUInt16)
    out   = sitk.Resample(label, ref, transform,
                          sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(output_path))
    return True


def resample_float(image_path, reference_path, transform, output_path) -> bool:
    if not image_path.exists():
        logger.warning(f"    Not found, skipping: {image_path.name}")
        return False
    img = sitk.ReadImage(str(image_path),     sitk.sitkFloat32)
    ref = sitk.ReadImage(str(reference_path), sitk.sitkFloat32)
    out = sitk.Resample(img, ref, transform,
                        sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(output_path))
    return True


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def find_t2w(nifti_dir: Path, study_id: str, acq: str):
    study_dir = nifti_dir / study_id
    if not study_dir.exists():
        return None
    for series_dir in sorted(study_dir.iterdir()):
        p = series_dir / f"sub-{study_id}_acq-{acq}_T2w.nii.gz"
        if p.exists():
            return p
    return None


def find_spineps_seg(spineps_dir: Path, study_id: str):
    p = spineps_dir / 'segmentations' / study_id / f"{study_id}_seg-spine_msk.nii.gz"
    return p if p.exists() else None


def find_spineps_unc(spineps_dir: Path, study_id: str):
    p = spineps_dir / 'segmentations' / study_id / f"{study_id}_unc.nii.gz"
    return p if p.exists() else None


# ============================================================================
# PER-STUDY
# ============================================================================

def already_registered(study_id: str, registered_dir: Path) -> bool:
    return (registered_dir / study_id / f"{study_id}_transform.tfm").exists()


def process_study(study_id: str, nifti_dir: Path,
                  spineps_dir: Path, registered_dir: Path) -> bool:
    out = registered_dir / study_id
    out.mkdir(parents=True, exist_ok=True)

    axial_path = find_t2w(nifti_dir, study_id, 'ax')
    sag_path   = find_t2w(nifti_dir, study_id, 'sag')

    if axial_path is None:
        raise FileNotFoundError(f"Axial T2w not found for {study_id}")
    if sag_path is None:
        raise FileNotFoundError(f"Sagittal T2w not found for {study_id}")

    logger.info(f"  axial : {axial_path.parent.name}")
    logger.info(f"  sagitt: {sag_path.parent.name}")

    transform_path = out / f"{study_id}_transform.tfm"
    logger.info(f"  Registering sag → ax ...")
    transform = register_sag_to_axial(axial_path, sag_path, transform_path)

    spineps_seg = find_spineps_seg(spineps_dir, study_id)
    if spineps_seg:
        ok = resample_label(spineps_seg, axial_path, transform,
                            out / f"{study_id}_spineps_reg.nii.gz")
        logger.info(f"  SPINEPS seg → {'OK' if ok else 'SKIPPED'}")
    else:
        logger.warning(f"  SPINEPS seg not found — skipping "
                       f"(study may not have been selected in 02b)")

    spineps_unc = find_spineps_unc(spineps_dir, study_id)
    if spineps_unc:
        resample_float(spineps_unc, axial_path, transform,
                       out / f"{study_id}_spineps_unc_reg.nii.gz")

    resample_float(sag_path, axial_path, transform,
                   out / f"{study_id}_sag_preview.nii.gz")

    logger.info(f"  Done")
    return True


# ============================================================================
# PROGRESS
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
        description='Register sag SPINEPS into axial space — selective or all'
    )
    parser.add_argument('--nifti_dir',       required=True)
    parser.add_argument('--spineps_dir',     required=True)
    parser.add_argument('--registered_dir',  required=True)

    # Selection (mirrors 02b exactly)
    parser.add_argument('--uncertainty_csv', default=None,
                        help='lstv_uncertainty_metrics.csv (required unless --all)')
    parser.add_argument('--valid_ids',       default=None,
                        help='models/valid_id.npy')
    parser.add_argument('--top_n',           type=int, default=None,
                        help='Studies from each end of ranking (required unless --all)')
    parser.add_argument('--rank_by',         default='l5_s1_confidence')
    parser.add_argument('--all',             action='store_true',
                        help='Process every study in nifti_dir, ignore uncertainty CSV')

    parser.add_argument('--retry-failed',    action='store_true')
    args = parser.parse_args()

    nifti_dir      = Path(args.nifti_dir)
    spineps_dir    = Path(args.spineps_dir)
    registered_dir = Path(args.registered_dir)
    registered_dir.mkdir(parents=True, exist_ok=True)

    progress, progress_file = load_progress(registered_dir)
    skip = set(progress['processed'])
    if args.retry_failed:
        skip -= set(progress.get('failed', []))

    # Build study list
    if args.all:
        SKIP_DIRS = {'metadata'}
        study_ids = sorted(d.name for d in nifti_dir.iterdir()
                           if d.is_dir() and d.name not in SKIP_DIRS)
        logger.info(f"ALL mode: {len(study_ids)} studies from {nifti_dir}")
    else:
        if not args.uncertainty_csv or args.top_n is None:
            parser.error("--uncertainty_csv and --top_n are required unless --all is set")

        valid_ids = None
        if args.valid_ids:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
            logger.info(f"Loaded {len(valid_ids)} valid study IDs")

        study_ids = select_studies(
            Path(args.uncertainty_csv), args.top_n, args.rank_by, valid_ids
        )
        logger.info(f"Selective mode: {len(study_ids)} studies selected")

    study_ids = [s for s in study_ids if s not in skip]
    logger.info(f"To register (excluding already done): {len(study_ids)}")

    success = failed = 0
    for sid in tqdm(study_ids, desc='Registering', unit='study'):
        logger.info(f"\n[{sid}]")
        try:
            process_study(sid, nifti_dir, spineps_dir, registered_dir)
            success += 1
            if sid not in progress['processed']:
                progress['processed'].append(sid)
            if sid in progress.get('failed', []):
                progress['failed'].remove(sid)
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            logger.debug(traceback.format_exc())
            failed += 1
            progress.setdefault('failed', []).append(sid)
        save_progress(progress, progress_file)

    logger.info(f"\nDone — success: {success}  failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
