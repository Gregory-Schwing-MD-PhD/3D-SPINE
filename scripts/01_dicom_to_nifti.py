#!/usr/bin/env python3
"""
DICOM to NIfTI Converter

Converts DICOM studies to NIfTI format using dcm2niix.
Converts both sagittal T2w and axial T2 series via the series descriptions CSV.

Filters to validation set only (via valid_id.npy) to prevent data leakage.

NIfTI output layout:
  results/nifti/{study_id}/{series_id}/sub-{study_id}_acq-sag_T2w.nii.gz
  results/nifti/{study_id}/{series_id}/sub-{study_id}_acq-ax_T2w.nii.gz

Usage:
    python 01_dicom_to_nifti.py \
        --input_dir  data/raw/train_images \
        --series_csv data/raw/train_series_descriptions.csv \
        --output_dir results/nifti \
        --valid_ids  models/valid_id.npy \
        --mode prod
"""

import argparse
import json
import subprocess
import shutil
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

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
# SERIES SELECTION
# ============================================================================

def load_series_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from series CSV")
        return df
    except Exception as e:
        logger.error(f"Failed to load series CSV: {e}")
        return None


def get_series_id(series_df: pd.DataFrame, study_id: str, patterns: list) -> str | None:
    """Return series_id (str) matching the first pattern found, or None."""
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
# DICOM -> NIFTI
# ============================================================================

def convert_dicom_to_nifti(dicom_dir: Path, out_dir: Path, bids_base: str) -> Path | None:
    """Convert a DICOM series to NIfTI using dcm2niix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = out_dir / f"{bids_base}.nii.gz"

    if expected.exists():
        logger.info(f"  NIfTI already exists, skipping: {expected.name}")
        return expected

    cmd = [
        '/usr/bin/dcm2niix',
        '-z', 'y',
        '-f', bids_base,
        '-o', str(out_dir),
        '-m', 'y',
        '-b', 'n',
        str(dicom_dir),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"  dcm2niix failed:\n{result.stderr}")
            return None

        if not expected.exists():
            candidates = sorted(out_dir.glob(f"{bids_base}*.nii.gz"))
            if not candidates:
                logger.error("  dcm2niix produced no output")
                logger.debug(result.stdout)
                return None
            shutil.move(str(candidates[0]), str(expected))

        logger.info(f"  Converted -> {expected.name}")
        return expected

    except subprocess.TimeoutExpired:
        logger.error("  dcm2niix timed out (>120s)")
        return None
    except FileNotFoundError:
        logger.error("  dcm2niix not found at /usr/bin/dcm2niix")
        return None
    except Exception as e:
        logger.error(f"  dcm2niix error: {e}")
        return None


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
            logger.warning(f"Could not load progress: {e} -- starting fresh")
    return {'processed': [], 'success': [], 'failed': []}


def save_progress(progress_file: Path, progress: dict):
    try:
        tmp = progress_file.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(progress, f, indent=2)
        tmp.replace(progress_file)
    except Exception as e:
        logger.warning(f"Could not save progress: {e}")


# ============================================================================
# METADATA
# ============================================================================

def save_metadata(study_id: str, conversions: dict, metadata_dir: Path):
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'study_id':    study_id,
        'conversions': conversions,
        'timestamp':   pd.Timestamp.now().isoformat(),
    }
    with open(metadata_dir / f"{study_id}_conversion.json", 'w') as f:
        json.dump(metadata, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DICOM -> NIfTI conversion (sagittal T2w + axial T2)'
    )
    parser.add_argument('--input_dir',  required=True,
                        help='Root DICOM directory (study_id/series_id/...)')
    parser.add_argument('--series_csv', required=True,
                        help='CSV with study_id, series_id, series_description')
    parser.add_argument('--output_dir', required=True,
                        help='Root NIfTI output directory')
    parser.add_argument('--valid_ids',  required=True,
                        help='Path to valid_id.npy — only these study IDs will be processed')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--trial_size', type=int, default=3,
                        help='Number of studies to process in trial mode (default: 3)')
    parser.add_argument('--mode', choices=['trial', 'debug', 'prod'], default='prod')
    args = parser.parse_args()

    input_dir     = Path(args.input_dir)
    output_dir    = Path(args.output_dir)
    metadata_dir  = output_dir / 'metadata'
    progress_file = output_dir / 'conversion_progress.json'

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and apply validation ID filter (prevents data leakage) ---
    valid_ids_path = Path(args.valid_ids)
    if not valid_ids_path.exists():
        logger.error(f"valid_ids file not found: {valid_ids_path}")
        logger.error("Download it first: sbatch slurm_scripts/00_download_all.sh")
        return 1

    valid_ids = set(str(v) for v in np.load(valid_ids_path))
    logger.info(f"✓ Loaded {len(valid_ids)} validation study IDs from {valid_ids_path}")
    logger.info("  VALIDATION SET ONLY — no data leakage")

    series_df = load_series_csv(Path(args.series_csv))
    if series_df is None:
        return 1

    progress          = load_progress(progress_file)
    already_processed = set(progress['processed'])

    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    # Filter to validation set
    n_total = len(study_dirs)
    study_dirs = [d for d in study_dirs if d.name in valid_ids]
    logger.info(f"After validation filter: {len(study_dirs)} / {n_total} studies retained")

    # Filter already-processed BEFORE applying mode limits
    study_dirs = [d for d in study_dirs if d.name not in already_processed]

    if args.mode == 'debug':
        study_dirs = study_dirs[:1]
    elif args.mode == 'trial':
        # Use first N studies in valid_id.npy order for reproducibility
        valid_ids_ordered = [str(v) for v in np.load(valid_ids_path)]
        study_dirs_map = {d.name: d for d in study_dirs}
        study_dirs = [study_dirs_map[v] for v in valid_ids_ordered if v in study_dirs_map][:args.trial_size]
    elif args.limit:
        study_dirs = study_dirs[:args.limit]

    logger.info("=" * 70)
    logger.info("DICOM -> NIFTI CONVERSION  (sagittal T2w + axial T2)")
    logger.info("=" * 70)
    logger.info(f"Mode:       {args.mode}")
    logger.info(f"To process: {len(study_dirs)}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Layout:     {{study_id}}/{{series_id}}/sub-{{study_id}}_acq-[sag|ax]_T2w.nii.gz")
    logger.info("=" * 70)

    success_count = len(progress['success'])
    error_count   = len(progress['failed'])

    for study_dir in tqdm(study_dirs, desc="Converting"):
        study_id = study_dir.name
        logger.info(f"\n[{study_id}]")
        conversions = {}
        any_success = False

        try:
            # ── Sagittal T2w ─────────────────────────────────────────────────
            sag_series_id = get_series_id(series_df, study_id, SAGITTAL_T2_PATTERNS)
            if sag_series_id is None:
                logger.warning("  ⚠ No sagittal T2w series in CSV")
            else:
                dicom_dir = study_dir / sag_series_id
                if not dicom_dir.exists():
                    logger.warning(f"  ⚠ Sagittal DICOM dir not found: {dicom_dir}")
                else:
                    out_dir    = output_dir / study_id / sag_series_id
                    bids_base  = f"sub-{study_id}_acq-sag_T2w"
                    nifti_path = convert_dicom_to_nifti(dicom_dir, out_dir, bids_base)
                    if nifti_path:
                        conversions['sagittal_t2'] = {
                            'series_id':  sag_series_id,
                            'nifti_path': str(nifti_path),
                        }
                        any_success = True
                    else:
                        logger.warning("  ⚠ Sagittal conversion failed")

            # ── Axial T2 ─────────────────────────────────────────────────────
            ax_series_id = get_series_id(series_df, study_id, AXIAL_T2_PATTERNS)
            if ax_series_id is None:
                logger.warning("  ⚠ No axial T2 series in CSV")
            else:
                dicom_dir = study_dir / ax_series_id
                if not dicom_dir.exists():
                    logger.warning(f"  ⚠ Axial DICOM dir not found: {dicom_dir}")
                else:
                    out_dir    = output_dir / study_id / ax_series_id
                    bids_base  = f"sub-{study_id}_acq-ax_T2w"
                    nifti_path = convert_dicom_to_nifti(dicom_dir, out_dir, bids_base)
                    if nifti_path:
                        conversions['axial_t2'] = {
                            'series_id':  ax_series_id,
                            'nifti_path': str(nifti_path),
                        }
                        any_success = True
                    else:
                        logger.warning("  ⚠ Axial conversion failed")

            # ── Progress ──────────────────────────────────────────────────────
            if conversions:
                save_metadata(study_id, conversions, metadata_dir)

            if any_success:
                progress['processed'].append(study_id)
                progress['success'].append(study_id)
                save_progress(progress_file, progress)
                success_count += 1
                logger.info(f"  ✓ Done ({list(conversions.keys())})")
            else:
                logger.warning("  ✗ No series converted")
                progress['processed'].append(study_id)
                progress['failed'].append(study_id)
                save_progress(progress_file, progress)
                error_count += 1

        except KeyboardInterrupt:
            logger.warning("\nInterrupted -- progress saved")
            save_progress(progress_file, progress)
            break
        except Exception as e:
            logger.error(f"  ✗ Unexpected error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            progress['processed'].append(study_id)
            progress['failed'].append(study_id)
            save_progress(progress_file, progress)
            error_count += 1

    logger.info("\n" + "=" * 70)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed:  {error_count}")
    logger.info(f"Total:   {success_count + error_count}")
    if progress['failed']:
        logger.info(f"Failed IDs: {progress['failed']}")
    logger.info(f"\nNIfTI files: {output_dir}/{{study_id}}/{{series_id}}/sub-*_acq-[sag|ax]_T2w.nii.gz")
    logger.info("Next step: sbatch slurm_scripts/02_lstv_trial.sh")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
