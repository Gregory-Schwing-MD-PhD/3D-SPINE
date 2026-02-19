#!/usr/bin/env python3
"""
DICOM to NIfTI Converter

Converts DICOM studies to NIfTI format using dcm2niix.
Selects sagittal T2w series via the series descriptions CSV.

NIfTI output layout:
  results/nifti/{study_id}/{series_id}/sub-{study_id}_acq-sag_T2w.nii.gz

Usage:
    python 01_dicom_to_nifti.py \
        --input_dir  data/raw/train_images \
        --series_csv data/raw/train_series_descriptions.csv \
        --output_dir results/nifti \
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


def get_sagittal_t2_series(series_df: pd.DataFrame, study_id: str) -> str | None:
    """Return series_id (str) of best sagittal T2w series, or None."""
    try:
        study_rows = series_df[series_df['study_id'] == int(study_id)]
        for pattern in SAGITTAL_T2_PATTERNS:
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

def convert_dicom_to_nifti(dicom_dir: Path, out_dir: Path, study_id: str) -> Path | None:
    """Convert a DICOM series to NIfTI using dcm2niix."""
    out_dir.mkdir(parents=True, exist_ok=True)

    bids_base = f"sub-{study_id}_acq-sag_T2w"
    expected  = out_dir / f"{bids_base}.nii.gz"

    if expected.exists():
        logger.info("  NIfTI already exists, skipping conversion")
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
        logger.error("  dcm2niix not found on PATH")
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

def save_metadata(study_id: str, series_id: str, nifti_path: Path, metadata_dir: Path):
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'study_id':   study_id,
        'series_id':  series_id,
        'nifti_path': str(nifti_path),
        'timestamp':  pd.Timestamp.now().isoformat(),
    }
    with open(metadata_dir / f"{study_id}_conversion.json", 'w') as f:
        json.dump(metadata, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DICOM -> NIfTI conversion (sagittal T2w only)'
    )
    parser.add_argument('--input_dir',  required=True,
                        help='Root DICOM directory (study_id/series_id/...)')
    parser.add_argument('--series_csv', required=True,
                        help='CSV with study_id, series_id, series_description')
    parser.add_argument('--output_dir', required=True,
                        help='Root NIfTI output directory')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--mode', choices=['trial', 'debug', 'prod'], default='prod')
    args = parser.parse_args()

    input_dir     = Path(args.input_dir)
    output_dir    = Path(args.output_dir)
    metadata_dir  = output_dir / 'metadata'
    progress_file = output_dir / 'conversion_progress.json'

    output_dir.mkdir(parents=True, exist_ok=True)

    series_df = load_series_csv(Path(args.series_csv))
    if series_df is None:
        return 1

    progress          = load_progress(progress_file)
    already_processed = set(progress['processed'])

    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if args.mode == 'debug':
        study_dirs = study_dirs[:1]
    elif args.mode == 'trial':
        study_dirs = study_dirs[:3]
    elif args.limit:
        study_dirs = study_dirs[:args.limit]

    remaining = [d for d in study_dirs if d.name not in already_processed]

    logger.info("=" * 70)
    logger.info("DICOM -> NIFTI CONVERSION  (sagittal T2w)")
    logger.info("=" * 70)
    logger.info(f"Mode:         {args.mode}")
    logger.info(f"Total:        {len(study_dirs)}")
    logger.info(f"Already done: {len(study_dirs) - len(remaining)}")
    logger.info(f"To process:   {len(remaining)}")
    logger.info(f"Output:       {output_dir}")
    logger.info(f"Layout:       {'{study_id}'}/{'{series_id}'}/sub-{'{study_id}'}_acq-sag_T2w.nii.gz")
    logger.info("=" * 70)

    success_count = len(progress['success'])
    error_count   = len(progress['failed'])

    for study_dir in tqdm(remaining, desc="Converting"):
        study_id = study_dir.name
        logger.info(f"\n[{study_id}]")

        try:
            series_id = get_sagittal_t2_series(series_df, study_id)
            if series_id is None:
                logger.warning("  No sagittal T2w series found in CSV")
                progress['processed'].append(study_id)
                progress['failed'].append(study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            dicom_series_dir = study_dir / series_id
            if not dicom_series_dir.exists():
                logger.warning(f"  DICOM series dir not found: {dicom_series_dir}")
                progress['processed'].append(study_id)
                progress['failed'].append(study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            logger.info(f"  Series: {series_id}")

            out_dir    = output_dir / study_id / series_id
            nifti_path = convert_dicom_to_nifti(dicom_series_dir, out_dir, study_id)

            if nifti_path is None:
                logger.warning("  Conversion failed")
                progress['processed'].append(study_id)
                progress['failed'].append(study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            save_metadata(study_id, series_id, nifti_path, metadata_dir)
            progress['processed'].append(study_id)
            progress['success'].append(study_id)
            save_progress(progress_file, progress)
            success_count += 1

        except KeyboardInterrupt:
            logger.warning("\nInterrupted -- progress saved")
            save_progress(progress_file, progress)
            break
        except Exception as e:
            logger.error(f"  Unexpected error: {e}")
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
    logger.info("\nNext step: sbatch slurm_scripts/02_spineps.sh")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
