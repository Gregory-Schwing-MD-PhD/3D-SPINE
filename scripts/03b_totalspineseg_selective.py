#!/usr/bin/env python3
"""
03b_totalspineseg_selective.py — Run TotalSpineSeg on top-N / bottom-N studies
===============================================================================

Reads results/epistemic_uncertainty/lstv_uncertainty_metrics.csv, ranks studies
by the given column, selects top N (highest) and bottom N (lowest), then runs
TotalSpineSeg on those studies — skipping any already segmented.

NIfTI paths are identical to those used by 02b_spineps_selective.py:
  results/nifti/{study_id}/{series_id}/sub-{study_id}_acq-sag_T2w.nii.gz
  results/nifti/{study_id}/{series_id}/sub-{study_id}_acq-ax_T2w.nii.gz

Fully self-contained — no imports from other local scripts.

Usage:
    python scripts/03b_totalspineseg_selective.py \
        --uncertainty_csv results/epistemic_uncertainty/lstv_uncertainty_metrics.csv \
        --nifti_dir       results/nifti \
        --output_dir      results/totalspineseg \
        --series_csv      data/raw/train_series_descriptions.csv \
        --valid_ids       models/valid_id.npy \
        --top_n           1 \
        --rank_by         l5_s1_confidence \
        [--dry_run]
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

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
# SERIES CSV
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
# PROGRESS
# ============================================================================

def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                p = json.load(f)
            logger.info(f"Resuming: {len(p.get('success', []))} done, "
                        f"{len(p.get('failed', []))} failed")
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


def mark_success(progress: dict, study_id: str):
    if study_id not in progress['processed']:
        progress['processed'].append(study_id)
    if study_id not in progress['success']:
        progress['success'].append(study_id)
    if study_id in progress.get('failed', []):
        progress['failed'].remove(study_id)


def mark_failed(progress: dict, study_id: str):
    if study_id not in progress['processed']:
        progress['processed'].append(study_id)
    if study_id not in progress.get('failed', []):
        progress.setdefault('failed', []).append(study_id)


# ============================================================================
# NIFTI RESOLUTION  (same logic as 03_run_totalspineseg.py)
# ============================================================================

def resolve_nifti(study_dir: Path, series_id: str, study_id: str, acq: str) -> Path | None:
    """Try standard name then dcm2niix _Eq_1 variant."""
    base   = study_dir / series_id / f"sub-{study_id}_acq-{acq}_T2w.nii.gz"
    eq_var = study_dir / series_id / f"sub-{study_id}_acq-{acq}_T2w_Eq_1.nii.gz"
    if base.exists():
        return base
    if eq_var.exists():
        return eq_var
    logger.warning(f"  NIfTI not found: {base}")
    return None


# ============================================================================
# TOTALSPINESEG
# ============================================================================

def run_totalspineseg(nifti_path: Path, study_output_dir: Path,
                      study_id: str, acq: str) -> dict | None:
    temp_output = study_output_dir / f"temp_{acq}"
    final_dir   = study_output_dir / acq
    final_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    try:
        cmd = [
            'totalspineseg',
            str(nifti_path),
            str(temp_output),
            '--save-uncertainties',
        ]

        logger.info(f"  Running TotalSpineSeg ({acq}, full inference)...")
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

        # Labeled vertebrae + discs (Step 2)
        step2_output_dir = temp_output / 'step2_output'
        if not step2_output_dir.exists():
            logger.error(f"  step2_output not found: {step2_output_dir}")
            return None

        output_files = sorted(step2_output_dir.glob("*.nii.gz"))
        if not output_files:
            logger.error(f"  No output files in {step2_output_dir}")
            return None

        labeled_dest = final_dir / f"{study_id}_{acq}_labeled.nii.gz"
        shutil.copy(output_files[0], labeled_dest)
        outputs['labeled'] = labeled_dest
        logger.info(f"  ✓ Labeled (Step 2): {labeled_dest.name}")

        # Disc level markers
        step1_levels_dir = temp_output / 'step1_levels'
        if step1_levels_dir.exists():
            level_files = sorted(step1_levels_dir.glob("*.nii.gz"))
            if level_files:
                dest = final_dir / f"{study_id}_{acq}_levels.nii.gz"
                shutil.copy(level_files[0], dest)
                outputs['levels'] = dest
                logger.info(f"  ✓ Levels: {dest.name}")

        # Spinal cord
        step1_cord_dir = temp_output / 'step1_cord'
        if step1_cord_dir.exists():
            cord_files = sorted(step1_cord_dir.glob("*.nii.gz"))
            if cord_files:
                dest = final_dir / f"{study_id}_{acq}_cord.nii.gz"
                shutil.copy(cord_files[0], dest)
                outputs['cord'] = dest
                logger.info(f"  ✓ Cord: {dest.name}")

        # Spinal canal
        step1_canal_dir = temp_output / 'step1_canal'
        if step1_canal_dir.exists():
            canal_files = sorted(step1_canal_dir.glob("*.nii.gz"))
            if canal_files:
                dest = final_dir / f"{study_id}_{acq}_canal.nii.gz"
                shutil.copy(canal_files[0], dest)
                outputs['canal'] = dest
                logger.info(f"  ✓ Canal: {dest.name}")

        # Uncertainty map
        step2_uncertainties_dir = temp_output / 'step2_uncertainties'
        if step2_uncertainties_dir.exists():
            unc_files = sorted(step2_uncertainties_dir.glob("*.nii.gz"))
            if unc_files:
                dest = final_dir / f"{study_id}_{acq}_unc.nii.gz"
                shutil.copy(unc_files[0], dest)
                outputs['uncertainty'] = dest
                logger.info(f"  ✓ Uncertainty: {dest.name}")
            else:
                logger.warning("  step2_uncertainties/ exists but no .nii.gz files")
        else:
            logger.warning("  step2_uncertainties/ not found")

        return outputs if 'labeled' in outputs else None

    except subprocess.TimeoutExpired:
        logger.error("  TotalSpineSeg timed out (>15 min)")
        return None
    except Exception as e:
        logger.error(f"  Error: {e}")
        logger.debug(traceback.format_exc())
        return None
    finally:
        if temp_output.exists():
            try:
                shutil.rmtree(temp_output)
            except Exception:
                pass


# ============================================================================
# ALREADY DONE CHECK
# ============================================================================

def already_segmented(study_id: str, output_dir: Path) -> bool:
    """Complete if the sagittal labeled mask exists."""
    return (output_dir / study_id / 'sagittal' /
            f"{study_id}_sagittal_labeled.nii.gz").exists()


# ============================================================================
# STUDY SELECTION
# ============================================================================

def select_studies(csv_path: Path, top_n: int, rank_by: str,
                   valid_ids: set | None) -> tuple[list[str], pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Uncertainty CSV not found: {csv_path}\n"
            f"Run 00_ian_pan_inference.sh first."
        )

    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)

    if valid_ids is not None:
        before = len(df)
        df = df[df['study_id'].isin(valid_ids)]
        logger.info(f"Filtered CSV to {len(df)} studies via valid_ids "
                    f"({before - len(df)} excluded)")

    if rank_by not in df.columns:
        raise ValueError(
            f"Column '{rank_by}' not in CSV.\nAvailable: {', '.join(df.columns)}"
        )

    df_sorted  = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    top_ids    = df_sorted.head(top_n)['study_id'].tolist()
    bottom_ids = df_sorted.tail(top_n)['study_id'].tolist()

    seen, selected = set(), []
    for sid in top_ids + bottom_ids:
        if sid not in seen:
            selected.append(sid)
            seen.add(sid)

    logger.info(f"Rank by:         {rank_by}")
    logger.info(f"Top {top_n}:          {top_ids}")
    logger.info(f"Bottom {top_n}:       {bottom_ids}")
    logger.info(f"Total (deduped): {len(selected)}")

    return selected, df_sorted


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run TotalSpineSeg on top-N / bottom-N uncertainty studies'
    )
    parser.add_argument('--uncertainty_csv', required=True)
    parser.add_argument('--nifti_dir',       required=True)
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--series_csv',      required=True)
    parser.add_argument('--valid_ids',       default=None)
    parser.add_argument('--top_n',           type=int, required=True)
    parser.add_argument('--rank_by',         default='l5_s1_confidence')
    parser.add_argument('--dry_run',         action='store_true')
    args = parser.parse_args()

    uncertainty_csv = Path(args.uncertainty_csv)
    nifti_dir       = Path(args.nifti_dir)
    output_dir      = Path(args.output_dir)
    series_csv      = Path(args.series_csv)
    progress_file   = output_dir / 'progress_selective.json'

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"TOTALSPINESEG SELECTIVE — top/bottom {args.top_n} by {args.rank_by}")
    logger.info("=" * 70)

    # Load valid IDs
    valid_ids = None
    if args.valid_ids:
        try:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
            logger.info(f"Loaded {len(valid_ids)} valid study IDs")
        except Exception as e:
            logger.error(f"Failed to load valid_ids: {e}")
            return 1

    selected_ids, df_ranked = select_studies(
        uncertainty_csv, args.top_n, args.rank_by, valid_ids
    )

    skip_already_done = [s for s in selected_ids if already_segmented(s, output_dir)]
    to_run            = [s for s in selected_ids if not already_segmented(s, output_dir)]

    logger.info(f"\nSelected:       {len(selected_ids)}")
    logger.info(f"Already done:   {len(skip_already_done)} (skipping: {skip_already_done})")
    logger.info(f"To run:         {len(to_run)}")

    if args.dry_run:
        logger.info("\n--- DRY RUN ---")
        for sid in to_run:
            row   = df_ranked[df_ranked['study_id'] == sid]
            score = float(row[args.rank_by].iloc[0]) if not row.empty else float('nan')
            logger.info(f"  {sid}  {args.rank_by}={score:.4f}")
        logger.info("--- DRY RUN complete — nothing was run ---")
        return 0

    if not to_run:
        logger.info("\nAll selected studies already segmented.")
        logger.info("Edit TOP_N in the slurm script and resubmit to segment more.")
        return 0

    series_df = load_series_csv(series_csv)
    if series_df is None:
        logger.error("Cannot load series CSV — aborting")
        return 1

    progress      = load_progress(progress_file)
    success_count = 0
    error_count   = 0

    for study_id in tqdm(to_run, desc='TotalSpineSeg'):
        logger.info(f"\n[{study_id}]")
        sys.stdout.flush()

        if already_segmented(study_id, output_dir):
            logger.info("  Already segmented since last check — skipping")
            mark_success(progress, study_id)
            save_progress(progress_file, progress)
            continue

        try:
            study_dir        = nifti_dir / study_id
            study_output_dir = output_dir / study_id
            study_output_dir.mkdir(parents=True, exist_ok=True)
            any_success      = False

            # Sagittal T2w
            sag_series_id = get_series_id(series_df, study_id, SAGITTAL_T2_PATTERNS)
            if sag_series_id is None:
                logger.warning("  No sagittal T2w series in CSV")
            else:
                nifti_path = resolve_nifti(study_dir, sag_series_id, study_id, 'sag')
                if nifti_path:
                    outputs = run_totalspineseg(nifti_path, study_output_dir, study_id, 'sagittal')
                    if outputs:
                        any_success = True

            # Axial T2
            ax_series_id = get_series_id(series_df, study_id, AXIAL_T2_PATTERNS)
            if ax_series_id is None:
                logger.warning("  No axial T2 series in CSV")
            else:
                nifti_path = resolve_nifti(study_dir, ax_series_id, study_id, 'ax')
                if nifti_path:
                    outputs = run_totalspineseg(nifti_path, study_output_dir, study_id, 'axial')
                    if outputs:
                        any_success = True

            if any_success:
                mark_success(progress, study_id)
                success_count += 1
                row = df_ranked[df_ranked['study_id'] == study_id]
                if not row.empty:
                    score = float(row[args.rank_by].iloc[0])
                    logger.info(f"  ✓  {args.rank_by}={score:.4f}")
            else:
                logger.warning("  No series segmented successfully")
                mark_failed(progress, study_id)
                error_count += 1

            save_progress(progress_file, progress)
            sys.stdout.flush()

        except KeyboardInterrupt:
            logger.warning("\nInterrupted — progress saved")
            save_progress(progress_file, progress)
            break
        except Exception as e:
            logger.error(f"  Unexpected: {e}")
            logger.debug(traceback.format_exc())
            mark_failed(progress, study_id)
            save_progress(progress_file, progress)
            error_count += 1

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info(f"Success:      {success_count}")
    logger.info(f"Failed:       {error_count}")
    logger.info(f"Already done: {len(skip_already_done)}")
    if progress.get('failed'):
        logger.info(f"Failed IDs:   {progress['failed']}")
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
