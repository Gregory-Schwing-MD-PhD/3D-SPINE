#!/usr/bin/env python3
"""
SPINEPS Segmentation Pipeline - Refactored

Converts DICOM → NIfTI (nested by study/series, sagittal T2w only via CSV),
then runs SPINEPS segmentation.

NIfTI layout:
  results/nifti/{study_id}/{series_id}/sub-{study_id}_acq-sag_T2w.nii.gz

Usage:
    python 02_run_spineps.py \
        --input_dir  data/raw/train_images \
        --series_csv data/raw/train_series_descriptions.csv \
        --output_dir results/spineps \
        --valid_ids  data/valid_ids.npy \
        --mode trial
"""

import argparse
import json
import subprocess
import shutil
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import logging
import traceback

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

# Series descriptions considered sagittal T2w (priority order)
SAGITTAL_T2_PATTERNS = [
    'Sagittal T2/STIR',
    'Sagittal T2',
    'SAG T2',
    'Sag T2',
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


def get_sagittal_t2_series(series_df: pd.DataFrame, study_id: str) -> str | None:
    """
    Return the series_id (as str) of the best sagittal T2w series for a study.
    Returns None if none found.
    """
    if series_df is None:
        return None
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
# DICOM → NIFTI  (nested: nifti_dir/{study_id}/{series_id}/sub-*_acq-sag_T2w.nii.gz)
# ============================================================================

def convert_dicom_to_nifti(
    dicom_dir: Path,
    nifti_study_dir: Path,
    study_id: str,
    series_id: str,
) -> Path | None:
    """
    Convert a single DICOM series to NIfTI.
    Output: nifti_study_dir/{series_id}/sub-{study_id}_acq-sag_T2w.nii.gz
    """
    out_dir = nifti_study_dir / series_id
    out_dir.mkdir(parents=True, exist_ok=True)

    bids_base = f"sub-{study_id}_acq-sag_T2w"
    expected  = out_dir / f"{bids_base}.nii.gz"

    if expected.exists():
        logger.info(f"  NIfTI already exists, skipping conversion")
        return expected

    cmd = [
        'dcm2niix',
        '-z', 'y',       # gzip
        '-f', bids_base,
        '-o', str(out_dir),
        '-m', 'y',        # merge slices
        '-b', 'n',        # no BIDS sidecar (we don't need it)
        str(dicom_dir),
    ]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120
        )
        if result.returncode != 0:
            logger.error(f"  dcm2niix failed:\n{result.stderr}")
            return None

        if not expected.exists():
            # dcm2niix sometimes appends extra suffixes — grab first match
            candidates = sorted(out_dir.glob(f"{bids_base}*.nii.gz"))
            if not candidates:
                logger.error("  dcm2niix produced no NIfTI output")
                return None
            shutil.move(str(candidates[0]), str(expected))

        logger.info(f"  ✓ NIfTI written: {expected.relative_to(expected.parents[3])}")
        return expected

    except subprocess.TimeoutExpired:
        logger.error("  dcm2niix timed out")
        return None
    except Exception as e:
        logger.error(f"  dcm2niix error: {e}")
        return None


# ============================================================================
# CENTROID COMPUTATION
# ============================================================================

def compute_all_centroids(
    instance_mask_path: Path,
    semantic_mask_path: Path,
    ctd_path: Path,
) -> dict:
    if not HAS_NIBABEL:
        return {}
    try:
        instance_data = nib.load(instance_mask_path).get_fdata().astype(int)
        semantic_data = nib.load(semantic_mask_path).get_fdata().astype(int)

        with open(ctd_path) as f:
            ctd_data = json.load(f)

        if len(ctd_data) < 2:
            return {}

        counts = {'vertebrae': 0, 'discs': 0, 'endplates': 0, 'subregions': 0}

        for label in np.unique(instance_data)[1:]:          # skip 0
            ls = str(label)
            if ls in ctd_data[1]:
                continue
            mask = instance_data == label
            ctd_data[1][ls] = {'50': list(center_of_mass(mask))}
            if label <= 28:
                counts['vertebrae'] += 1
            elif 119 <= label <= 126:
                counts['discs'] += 1
            elif label >= 200:
                counts['endplates'] += 1

        for label in np.unique(semantic_data)[1:]:
            ls = str(label)
            if ls in ctd_data[1]:
                continue
            mask = semantic_data == label
            ctd_data[1][ls] = {'50': list(center_of_mass(mask))}
            counts['subregions'] += 1

        with open(ctd_path, 'w') as f:
            json.dump(ctd_data, f, indent=2)

        return counts

    except Exception as e:
        logger.warning(f"  Centroid computation error: {e}")
        logger.debug(traceback.format_exc())
        return {}


# ============================================================================
# UNCERTAINTY MAP
# ============================================================================

def compute_uncertainty_from_softmax(
    derivatives_dir: Path,
    study_id: str,
    seg_dir: Path,
) -> bool:
    if not HAS_NIBABEL:
        return False
    try:
        logits_files = list(derivatives_dir.glob(f"**/*{study_id}*logit*.npz"))
        if not logits_files:
            return False

        softmax = np.load(logits_files[0])['arr_0']
        uncertainty = 1.0 - np.max(softmax, axis=-1)

        semantic_mask = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
        if not semantic_mask.exists():
            return False

        ref = nib.load(semantic_mask)
        nib.save(
            nib.Nifti1Image(uncertainty.astype(np.float32), ref.affine, ref.header),
            seg_dir / f"{study_id}_unc.nii.gz",
        )
        logger.info("  ✓ Uncertainty map saved")
        return True

    except Exception as e:
        logger.warning(f"  Uncertainty map failed: {e}")
        return False


# ============================================================================
# SPINEPS
# ============================================================================

def run_spineps(nifti_path: Path, seg_dir: Path, study_id: str) -> dict | None:
    import os
    seg_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['SPINEPS_SEGMENTOR_MODELS'] = '/app/models'
    env['SPINEPS_ENVIRONMENT_DIR']  = '/app/models'

    cmd = [
        'python', '-m', 'spineps.entrypoint', 'sample',
        '-i', str(nifti_path),
        '-model_semantic',  't2w',
        '-model_instance',  'instance',
        '-model_labeling',  't2w_labeling',
        '-save_softmax_logits',
        '-override_semantic',
        '-override_instance',
        '-override_ctd',
    ]

    logger.info("  Running SPINEPS...")
    try:
        result = subprocess.run(
            cmd, stderr=subprocess.PIPE, text=True, timeout=600, env=env
        )
    except subprocess.TimeoutExpired:
        logger.error("  SPINEPS timed out (>600s)")
        return None
    except Exception as e:
        logger.error(f"  SPINEPS error: {e}")
        return None

    if result.returncode != 0:
        logger.error(f"  SPINEPS non-zero exit:\n{result.stderr}")
        return None

    # Outputs land in derivatives_seg/ next to the NIfTI
    derivatives_base = nifti_path.parent / "derivatives_seg"
    if not derivatives_base.exists():
        logger.error(f"  derivatives_seg not found at: {derivatives_base}")
        return None

    def find_file(exact: str, glob: str) -> Path | None:
        f = derivatives_base / exact
        if f.exists():
            return f
        hits = list(derivatives_base.glob(glob))
        return hits[0] if hits else None

    outputs = {}

    # Instance mask
    f = find_file(f"sub-{study_id}_acq-sag_mod-T2w_seg-vert_msk.nii.gz", "**/*_seg-vert_msk.nii.gz")
    if f:
        dest = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
        shutil.copy(f, dest)
        outputs['instance_mask'] = dest
        logger.info("  ✓ Instance mask")
    else:
        logger.warning("  ⚠ Instance mask not found")

    # Semantic mask
    f = find_file(f"sub-{study_id}_acq-sag_mod-T2w_seg-spine_msk.nii.gz", "**/*_seg-spine_msk.nii.gz")
    if f:
        dest = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
        shutil.copy(f, dest)
        outputs['semantic_mask'] = dest
        logger.info("  ✓ Semantic mask")

    # Sub-region mask
    f = find_file(f"sub-{study_id}_acq-sag_mod-T2w_seg-subreg_msk.nii.gz", "**/*_seg-subreg_msk.nii.gz")
    if f:
        dest = seg_dir / f"{study_id}_seg-subreg_msk.nii.gz"
        shutil.copy(f, dest)
        outputs['subreg_mask'] = dest
        logger.info("  ✓ Sub-region mask")

    # Centroids JSON
    f = find_file(f"sub-{study_id}_acq-sag_mod-T2w_ctd.json", "**/*_ctd.json")
    if f:
        dest = seg_dir / f"{study_id}_ctd.json"
        shutil.copy(f, dest)
        outputs['centroid_json'] = dest
        logger.info("  ✓ Centroids JSON")

        if 'instance_mask' in outputs and 'semantic_mask' in outputs:
            counts = compute_all_centroids(
                outputs['instance_mask'], outputs['semantic_mask'], dest
            )
            if counts:
                total = sum(counts.values())
                logger.info(
                    f"  ✓ Added {total} centroids: "
                    f"{counts['discs']} discs, "
                    f"{counts['endplates']} endplates, "
                    f"{counts['subregions']} subregions"
                )

    # Uncertainty map
    if 'semantic_mask' in outputs:
        if compute_uncertainty_from_softmax(derivatives_base, study_id, seg_dir):
            outputs['uncertainty_map'] = seg_dir / f"{study_id}_unc.nii.gz"

    if 'instance_mask' not in outputs:
        logger.error("  Instance mask missing — treating as failure")
        return None

    return outputs


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                p = json.load(f)
            logger.info(
                f"Resuming: {len(p['success'])} done, {len(p['failed'])} failed"
            )
            return p
        except Exception as e:
            logger.warning(f"Could not load progress: {e} — starting fresh")
    return {'processed': [], 'success': [], 'failed': []}


def save_progress(progress_file: Path, progress: dict):
    try:
        tmp = progress_file.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(progress, f, indent=2)
        tmp.replace(progress_file)
    except Exception as e:
        logger.warning(f"Could not save progress: {e}")


def mark_failed(progress: dict, study_id: str):
    if study_id not in progress['processed']:
        progress['processed'].append(study_id)
    if study_id not in progress['failed']:
        progress['failed'].append(study_id)


def mark_success(progress: dict, study_id: str):
    if study_id not in progress['processed']:
        progress['processed'].append(study_id)
    if study_id not in progress['success']:
        progress['success'].append(study_id)
    # Remove from failed if retrying
    if study_id in progress['failed']:
        progress['failed'].remove(study_id)


# ============================================================================
# METADATA
# ============================================================================

def save_metadata(study_id: str, outputs: dict, metadata_dir: Path):
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'study_id':  study_id,
        'outputs':   {k: str(v) for k, v in outputs.items()},
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    with open(metadata_dir / f"{study_id}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SPINEPS Segmentation Pipeline')
    parser.add_argument('--input_dir',  required=True,
                        help='Root DICOM directory (study_id/series_id/...)')
    parser.add_argument('--series_csv', required=True,
                        help='CSV with study_id, series_id, series_description columns')
    parser.add_argument('--output_dir', required=True,
                        help='Root output directory')
    parser.add_argument('--valid_ids',  default=None,
                        help='Optional .npy file of study IDs to process')
    parser.add_argument('--limit',      type=int, default=None)
    parser.add_argument('--mode',       choices=['trial', 'debug', 'prod'], default='prod')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry previously failed studies')
    args = parser.parse_args()

    input_dir    = Path(args.input_dir)
    output_dir   = Path(args.output_dir)
    nifti_dir    = output_dir / 'nifti'          # nested: nifti/{study_id}/{series_id}/
    seg_dir      = output_dir / 'segmentations'
    metadata_dir = output_dir / 'metadata'
    progress_file = output_dir / 'progress.json'

    for d in [nifti_dir, seg_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    progress = load_progress(progress_file)
    skip_ids = (
        set(progress.get('success', []))
        if args.retry_failed
        else set(progress['processed'])
    )

    # Load CSV — mandatory
    series_df = load_series_csv(Path(args.series_csv))
    if series_df is None:
        logger.error("Cannot proceed without series CSV")
        return 1

    # Enumerate study directories
    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    # Filter to valid_ids if provided
    if args.valid_ids:
        try:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
            study_dirs = [d for d in study_dirs if d.name in valid_ids]
            logger.info(f"Filtered to {len(study_dirs)} studies from valid_ids")
        except Exception as e:
            logger.error(f"Failed to load valid_ids: {e}")
            return 1

    # Apply mode limits
    if args.mode == 'debug':
        study_dirs = study_dirs[:1]
    elif args.mode == 'trial':
        study_dirs = study_dirs[:3]
    elif args.limit:
        study_dirs = study_dirs[:args.limit]

    # Filter already-processed
    study_dirs = [d for d in study_dirs if d.name not in skip_ids]

    logger.info("=" * 70)
    logger.info("SPINEPS SEGMENTATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Mode:          {args.mode}")
    logger.info(f"To process:    {len(study_dirs)}")
    logger.info(f"Output root:   {output_dir}")
    logger.info(f"NIfTI layout:  nifti/{{study_id}}/{{series_id}}/sub-*_acq-sag_T2w.nii.gz")
    logger.info("=" * 70)

    success_count = len(progress['success'])
    error_count   = len(progress['failed'])

    for study_dir in tqdm(study_dirs, desc="Studies"):
        study_id = study_dir.name
        logger.info(f"\n[{study_id}]")

        try:
            # ── 1. Find the sagittal T2w series via CSV ──────────────────────
            series_id = get_sagittal_t2_series(series_df, study_id)
            if series_id is None:
                logger.warning("  ✗ No sagittal T2w series found in CSV — skipping")
                mark_failed(progress, study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            dicom_series_dir = study_dir / series_id
            if not dicom_series_dir.exists():
                logger.warning(f"  ✗ DICOM series dir not found: {dicom_series_dir}")
                mark_failed(progress, study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            logger.info(f"  Series: {series_id} (sagittal T2w)")

            # ── 2. DICOM → NIfTI (nested layout) ─────────────────────────────
            nifti_study_dir = nifti_dir / study_id
            nifti_path = nifti_study_dir / series_id / f"sub-{study_id}_acq-sag_T2w.nii.gz"

            if not nifti_path.exists():
                logger.info("  Converting DICOM → NIfTI...")
                nifti_path = convert_dicom_to_nifti(
                    dicom_series_dir, nifti_study_dir, study_id, series_id
                )
                if nifti_path is None:
                    logger.warning("  ✗ DICOM conversion failed")
                    mark_failed(progress, study_id)
                    save_progress(progress_file, progress)
                    error_count += 1
                    continue
            else:
                logger.info("  NIfTI already present, skipping conversion")

            # ── 3. SPINEPS ────────────────────────────────────────────────────
            study_seg_dir = seg_dir / study_id
            outputs = run_spineps(nifti_path, study_seg_dir, study_id)

            if outputs is None:
                logger.warning("  ✗ SPINEPS failed")
                mark_failed(progress, study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            save_metadata(study_id, outputs, metadata_dir)
            mark_success(progress, study_id)
            save_progress(progress_file, progress)
            success_count += 1
            logger.info(f"  ✓ Done ({len(outputs)} outputs)")

        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted — progress saved")
            save_progress(progress_file, progress)
            break
        except Exception as e:
            logger.error(f"  ✗ Unexpected error: {e}")
            logger.debug(traceback.format_exc())
            mark_failed(progress, study_id)
            save_progress(progress_file, progress)
            error_count += 1

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
    logger.info("Outputs per study (under segmentations/{study_id}/):")
    logger.info("  • {study_id}_seg-vert_msk.nii.gz   - Instance mask")
    logger.info("  • {study_id}_seg-spine_msk.nii.gz  - Semantic mask")
    logger.info("  • {study_id}_seg-subreg_msk.nii.gz - Sub-region mask")
    logger.info("  • {study_id}_ctd.json              - Centroids (all structures)")
    logger.info("  • {study_id}_unc.nii.gz            - Uncertainty map")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
