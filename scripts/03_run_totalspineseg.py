#!/usr/bin/env python3
"""
TotalSpineSeg Wrapper - with Uncertainty Maps

Runs TotalSpineSeg on sagittal T2w and axial T2 NIfTI files.
After inference, computes per-voxel uncertainty from the nnU-Net softmax
probability outputs (saved via --save_probabilities), using the same
approach as 02_run_spineps.py:

    uncertainty = 1 - max(softmax_probabilities, axis=channel)

This gives a map where 0 = highly confident, 1 = maximally uncertain.
Shannon entropy is also computed as an alternative measure.

TotalSpineSeg is nnU-Net-backed, so --save_probabilities produces
.npz files (shape: C x H x W x D, channels = classes) in the
step1_raw/ directory alongside the segmentation.

NIfTI layout expected (from 01_dicom_to_nifti.py):
  results/nifti/{study_id}/{series_id}/sub-{study_id}_acq-sag_T2w.nii.gz
  results/nifti/{study_id}/{series_id}/sub-{study_id}_acq-ax_T2w.nii.gz

Usage:
    python 03_run_totalspineseg.py \
        --nifti_dir  results/nifti \
        --series_csv data/raw/train_series_descriptions.csv \
        --output_dir results/totalspineseg \
        --mode trial
"""

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SAGITTAL_T2_PATTERNS = [
    'Sagittal T2/STIR',
    'Sagittal T2',
    'SAG T2',
    'Sag T2',
]

AXIAL_T2_PATTERNS = [
    'Axial T2',
    'AXIAL T2',
    'Ax T2',
    'AX T2',
]


# ============================================================================
# SERIES SELECTION VIA CSV
# ============================================================================

def load_series_csv(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from series CSV")
        return df
    except Exception as e:
        logger.error(f"Failed to load series CSV: {e}")
        return None


def get_series_id(series_df: pd.DataFrame, study_id: str, patterns: list) -> str | None:
    """Return series_id (str) matching one of the patterns, or None."""
    if series_df is None:
        return None
    try:
        study_rows = series_df[series_df['study_id'] == int(study_id)]
        for pattern in patterns:
            match = study_rows[
                study_rows['series_description'].str.contains(pattern, case=False, na=False)
            ]
            if not match.empty:
                return str(match.iloc[0]['series_id'])
    except Exception as e:
        logger.warning(f"  Series CSV lookup failed for {study_id}: {e}")
    return None


# ============================================================================
# UNCERTAINTY FROM SOFTMAX
# ============================================================================

def compute_uncertainty_from_softmax(
    search_dir: Path,
    ref_nifti_path: Path,
    output_path: Path,
) -> bool:
    """
    Compute uncertainty maps from nnU-Net softmax .npz files saved by
    TotalSpineSeg's --save_probabilities flag.

    nnU-Net .npz layout:
        arr_0  — float32 array, shape (C, H, W, D)
                 where C = number of segmentation classes

    Two uncertainty metrics are saved in the same NIfTI as separate volumes:
        Volume 0 — max-prob uncertainty:  1 - max(softmax, axis=0)
        Volume 1 — Shannon entropy:       -sum(p * log2(p + eps), axis=0)

    The step1_raw/ directory is where TotalSpineSeg places raw nnU-Net output,
    so that is searched first.

    Returns True if an uncertainty map was successfully written.
    """
    if not HAS_NIBABEL:
        logger.warning("  nibabel not available; skipping uncertainty computation")
        return False

    # Locate .npz probability file — nnU-Net names them after the input file
    npz_files = (
        list(search_dir.glob("*.npz"))
        + list(search_dir.glob("**/*.npz"))
    )
    if not npz_files:
        logger.warning(f"  No .npz probability files found under {search_dir}")
        return False

    # If multiple files exist (step1 + step2) prefer step1 (lumbar region labels)
    npz_path = npz_files[0]
    if len(npz_files) > 1:
        step1_candidates = [f for f in npz_files if 'step1' in f.parts or 'step1' in f.name]
        if step1_candidates:
            npz_path = step1_candidates[0]
    logger.info(f"  Using softmax file: {npz_path.name}")

    try:
        softmax = np.load(npz_path)['arr_0']  # shape: (C, H, W, D)
    except Exception as e:
        logger.warning(f"  Failed to load softmax .npz: {e}")
        return False

    if softmax.ndim != 4:
        logger.warning(f"  Unexpected softmax shape {softmax.shape}; expected (C, H, W, D)")
        return False

    # ---- Metric 1: max-probability uncertainty (matches SPINEPS approach) ----
    max_prob    = softmax.max(axis=0)                            # (H, W, D)
    unc_maxprob = (1.0 - max_prob).astype(np.float32)

    # ---- Metric 2: Shannon entropy ----
    eps         = 1e-7
    log_softmax = np.log2(softmax + eps)
    entropy     = -(softmax * log_softmax).sum(axis=0)           # (H, W, D)
    # Normalise entropy to [0, 1] by dividing by log2(num_classes)
    n_classes   = softmax.shape[0]
    if n_classes > 1:
        entropy = (entropy / np.log2(n_classes)).astype(np.float32)

    # ---- Pack into 4D NIfTI (H, W, D, 2) ----
    # Load reference image to get affine / header
    try:
        ref_nii = nib.load(str(ref_nifti_path))
    except Exception as e:
        logger.warning(f"  Could not load reference NIfTI for affine: {e}")
        return False

    # Softmax volume might be in model space (1 mm isotropic); reference
    # image might differ.  We store at softmax resolution and note this.
    unc_4d = np.stack([unc_maxprob, entropy], axis=-1)           # (H, W, D, 2)

    # Build a minimal header from reference; update shape/dims
    header = ref_nii.header.copy()
    header.set_data_shape(unc_4d.shape)
    header.set_data_dtype(np.float32)

    unc_nii = nib.Nifti1Image(unc_4d, ref_nii.affine, header)
    nib.save(unc_nii, str(output_path))

    logger.info(
        f"  ✓ Uncertainty map saved: {output_path.name}  "
        f"(vol0=max-prob, vol1=entropy, shape={unc_4d.shape})"
    )
    return True


# ============================================================================
# TOTALSPINESEG
# ============================================================================

def run_totalspineseg(
    nifti_path: Path,
    study_output_dir: Path,
    study_id: str,
    acq: str,
) -> dict | None:
    """
    Run TotalSpineSeg on one NIfTI and return a dict of output paths,
    or None on failure.

    Passes --save_probabilities so nnU-Net writes softmax .npz files
    to step1_raw/, from which we derive uncertainty maps.

    acq: 'sagittal' or 'axial'
    """
    temp_output = study_output_dir / f"temp_{acq}"
    final_dir   = study_output_dir / acq
    final_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    try:
        cmd = [
            'totalspineseg',
            str(nifti_path),
            str(temp_output),
            '--step1',              # Step 1: vertebra labels + landmarks
            '--save_probabilities', # nnU-Net flag → softmax .npz in step1_raw/
        ]

        logger.info(f"  Running TotalSpineSeg ({acq})...")
        sys.stdout.flush()

        result = subprocess.run(
            cmd,
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,
        )
        sys.stdout.flush()

        if result.returncode != 0:
            logger.error(f"  TotalSpineSeg failed:\n{result.stderr[-2000:]}")
            return None

        # ── Labeled vertebrae (step1_output) ─────────────────────────────────
        step1_output_dir = temp_output / 'step1_output'
        if not step1_output_dir.exists():
            logger.error(f"  step1_output not found: {step1_output_dir}")
            return None

        output_files = sorted(step1_output_dir.glob("*.nii.gz"))
        if not output_files:
            logger.error(f"  No output files in {step1_output_dir}")
            return None

        labeled_dest = final_dir / f"{study_id}_{acq}_labeled.nii.gz"
        shutil.copy(output_files[0], labeled_dest)
        outputs['labeled'] = labeled_dest
        logger.info(f"  ✓ Labeled vertebrae: {labeled_dest.name}")

        # ── Disc level markers (step1_levels) ────────────────────────────────
        step1_levels_dir = temp_output / 'step1_levels'
        if step1_levels_dir.exists():
            level_files = sorted(step1_levels_dir.glob("*.nii.gz"))
            if level_files:
                levels_dest = final_dir / f"{study_id}_{acq}_levels.nii.gz"
                shutil.copy(level_files[0], levels_dest)
                outputs['levels'] = levels_dest
                logger.info(f"  ✓ Disc levels: {levels_dest.name}")

        # ── Spinal cord (step1_cord) ──────────────────────────────────────────
        step1_cord_dir = temp_output / 'step1_cord'
        if step1_cord_dir.exists():
            cord_files = sorted(step1_cord_dir.glob("*.nii.gz"))
            if cord_files:
                cord_dest = final_dir / f"{study_id}_{acq}_cord.nii.gz"
                shutil.copy(cord_files[0], cord_dest)
                outputs['cord'] = cord_dest
                logger.info(f"  ✓ Spinal cord: {cord_dest.name}")

        # ── Spinal canal (step1_canal) ────────────────────────────────────────
        step1_canal_dir = temp_output / 'step1_canal'
        if step1_canal_dir.exists():
            canal_files = sorted(step1_canal_dir.glob("*.nii.gz"))
            if canal_files:
                canal_dest = final_dir / f"{study_id}_{acq}_canal.nii.gz"
                shutil.copy(canal_files[0], canal_dest)
                outputs['canal'] = canal_dest
                logger.info(f"  ✓ Spinal canal: {canal_dest.name}")

        # ── Uncertainty map (from step1_raw softmax .npz) ────────────────────
        step1_raw_dir = temp_output / 'step1_raw'
        if step1_raw_dir.exists():
            unc_dest = final_dir / f"{study_id}_{acq}_unc.nii.gz"
            ok = compute_uncertainty_from_softmax(
                search_dir     = step1_raw_dir,
                ref_nifti_path = labeled_dest,   # use labeled output for affine
                output_path    = unc_dest,
            )
            if ok:
                outputs['uncertainty'] = unc_dest
        else:
            logger.warning(
                "  ⚠ step1_raw/ not found — TotalSpineSeg may not support "
                "--save_probabilities in this version; uncertainty map skipped"
            )

        return outputs

    except subprocess.TimeoutExpired:
        logger.error("  TotalSpineSeg timed out (>15 min)")
        sys.stdout.flush()
        return None
    except Exception as e:
        logger.error(f"  Error: {e}")
        logger.debug(traceback.format_exc())
        sys.stdout.flush()
        return None
    finally:
        if temp_output.exists():
            try:
                shutil.rmtree(temp_output)
            except Exception:
                pass


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                p = json.load(f)
            logger.info(f"Resuming: {len(p['success'])} done, {len(p['failed'])} failed")
            return p
        except Exception as e:
            logger.warning(f"Could not load progress: {e} — starting fresh")
    return {'processed': [], 'success': [], 'failed': []}


def save_progress(progress_file: Path, progress: dict) -> None:
    try:
        tmp = progress_file.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(progress, f, indent=2)
        tmp.replace(progress_file)
    except Exception as e:
        logger.warning(f"Could not save progress: {e}")


def mark_failed(progress: dict, study_id: str) -> None:
    if study_id not in progress['processed']:
        progress['processed'].append(study_id)
    if study_id not in progress['failed']:
        progress['failed'].append(study_id)


def mark_success(progress: dict, study_id: str) -> None:
    if study_id not in progress['processed']:
        progress['processed'].append(study_id)
    if study_id not in progress['success']:
        progress['success'].append(study_id)
    if study_id in progress['failed']:
        progress['failed'].remove(study_id)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description='TotalSpineSeg Segmentation Pipeline')
    parser.add_argument('--nifti_dir',      required=True,
                        help='Root NIfTI directory ({study_id}/{series_id}/sub-*_T2w.nii.gz)')
    parser.add_argument('--series_csv',     required=True,
                        help='CSV with study_id, series_id, series_description')
    parser.add_argument('--output_dir',     required=True,
                        help='Output directory for segmentations')
    parser.add_argument('--valid_ids',      default=None,
                        help='Optional .npy file of study IDs to process')
    parser.add_argument('--limit',          type=int, default=None)
    parser.add_argument('--mode',           choices=['trial', 'debug', 'prod'], default='prod')
    parser.add_argument('--retry-failed',   action='store_true')
    args = parser.parse_args()

    nifti_dir     = Path(args.nifti_dir)
    output_dir    = Path(args.output_dir)
    progress_file = output_dir / 'progress.json'

    output_dir.mkdir(parents=True, exist_ok=True)

    series_df = load_series_csv(Path(args.series_csv))
    if series_df is None:
        logger.error("Cannot proceed without series CSV")
        return 1

    progress = load_progress(progress_file)
    skip_ids = (
        set(progress.get('success', []))
        if args.retry_failed
        else set(progress['processed'])
    )

    study_dirs = sorted([d for d in nifti_dir.iterdir() if d.is_dir() and d.name != 'metadata'])

    if args.valid_ids:
        try:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
            study_dirs = [d for d in study_dirs if d.name in valid_ids]
            logger.info(f"Filtered to {len(study_dirs)} studies from valid_ids")
        except Exception as e:
            logger.error(f"Failed to load valid_ids: {e}")
            return 1

    if args.mode == 'debug':
        study_dirs = study_dirs[:1]
    elif args.mode == 'trial':
        study_dirs = study_dirs[:3]
    elif args.limit:
        study_dirs = study_dirs[:args.limit]

    study_dirs = [d for d in study_dirs if d.name not in skip_ids]

    logger.info("=" * 70)
    logger.info("TOTALSPINESEG SEGMENTATION (Full Pipeline + Uncertainty)")
    logger.info("=" * 70)
    logger.info(f"Mode:        {args.mode}")
    logger.info(f"To process:  {len(study_dirs)}")
    logger.info(f"NIfTI dir:   {nifti_dir}")
    logger.info(f"Output:      {output_dir}")
    logger.info("")
    logger.info("Outputs per study / orientation:")
    logger.info("  • {acq}_labeled.nii.gz  - Individual vertebra labels")
    logger.info("  • {acq}_levels.nii.gz   - Disc level markers")
    logger.info("  • {acq}_cord.nii.gz     - Spinal cord segmentation")
    logger.info("  • {acq}_canal.nii.gz    - Spinal canal segmentation")
    logger.info("  • {acq}_unc.nii.gz      - Uncertainty map (vol0=max-prob, vol1=entropy)")
    logger.info("=" * 70)
    sys.stdout.flush()

    success_count = len(progress['success'])
    error_count   = len(progress['failed'])

    for study_dir in tqdm(study_dirs, desc="Studies"):
        study_id         = study_dir.name
        study_output_dir = output_dir / study_id
        study_output_dir.mkdir(parents=True, exist_ok=True)
        any_success      = False

        logger.info(f"\n[{study_id}]")
        sys.stdout.flush()

        try:
            # ── Sagittal T2w ─────────────────────────────────────────────────
            sag_series_id = get_series_id(series_df, study_id, SAGITTAL_T2_PATTERNS)
            if sag_series_id is None:
                logger.warning("  ⚠ No sagittal T2w series in CSV")
            else:
                nifti_path = _resolve_nifti(study_dir, sag_series_id, study_id, 'sag')
                if nifti_path:
                    logger.info(f"  Series (sag): {sag_series_id}")
                    outputs = run_totalspineseg(nifti_path, study_output_dir, study_id, 'sagittal')
                    if outputs:
                        any_success = True
                        _log_outputs(outputs)

            # ── Axial T2 ─────────────────────────────────────────────────────
            ax_series_id = get_series_id(series_df, study_id, AXIAL_T2_PATTERNS)
            if ax_series_id is None:
                logger.warning("  ⚠ No axial T2 series in CSV")
            else:
                nifti_path = _resolve_nifti(study_dir, ax_series_id, study_id, 'ax')
                if nifti_path:
                    logger.info(f"  Series (ax):  {ax_series_id}")
                    outputs = run_totalspineseg(nifti_path, study_output_dir, study_id, 'axial')
                    if outputs:
                        any_success = True
                        _log_outputs(outputs)

            # ── Progress ──────────────────────────────────────────────────────
            if any_success:
                mark_success(progress, study_id)
                success_count += 1
                logger.info("  ✓ Done")
            else:
                logger.warning("  ✗ No series segmented successfully")
                mark_failed(progress, study_id)
                error_count += 1

            save_progress(progress_file, progress)
            sys.stdout.flush()

        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted — progress saved")
            save_progress(progress_file, progress)
            sys.stdout.flush()
            break
        except Exception as e:
            logger.error(f"  ✗ Unexpected error: {e}")
            logger.debug(traceback.format_exc())
            mark_failed(progress, study_id)
            save_progress(progress_file, progress)
            error_count += 1
            sys.stdout.flush()

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)
    logger.info(f"Success:  {success_count}")
    logger.info(f"Failed:   {error_count}")
    logger.info(f"Total:    {success_count + error_count}")
    if progress['failed']:
        logger.info(f"Failed IDs: {progress['failed']}")
    logger.info(f"Progress: {progress_file}")
    logger.info("")
    logger.info("Output structure per study:")
    logger.info(f"  {output_dir}/{{study_id}}/sagittal/  and  .../axial/")
    logger.info("    ├── *_labeled.nii.gz")
    logger.info("    ├── *_levels.nii.gz")
    logger.info("    ├── *_cord.nii.gz")
    logger.info("    ├── *_canal.nii.gz")
    logger.info("    └── *_unc.nii.gz      ← uncertainty (2 volumes)")
    logger.info("")
    logger.info("Next: sbatch slurm_scripts/04_lstv_detection.sh")
    sys.stdout.flush()

    return 0 if error_count == 0 else 1


# ============================================================================
# HELPERS
# ============================================================================

def _resolve_nifti(study_dir: Path, series_id: str, study_id: str, acq: str) -> Path | None:
    """Try standard name then dcm2niix _Eq_1 variant."""
    base   = study_dir / series_id / f"sub-{study_id}_acq-{acq}_T2w.nii.gz"
    eq_var = study_dir / series_id / f"sub-{study_id}_acq-{acq}_T2w_Eq_1.nii.gz"
    if base.exists():
        return base
    if eq_var.exists():
        return eq_var
    logger.warning(f"  ✗ NIfTI not found: {base}")
    return None


def _log_outputs(outputs: dict) -> None:
    for key, path in outputs.items():
        logger.info(f"    {key}: {Path(path).name}")


if __name__ == '__main__':
    sys.exit(main())
