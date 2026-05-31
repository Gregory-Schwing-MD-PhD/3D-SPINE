#!/usr/bin/env python3
"""
03b_totalspineseg_selective.py — Run TotalSpineSeg on top-N / bottom-N studies
                                  OR on the entire valid-ID dataset (--all flag)
===============================================================================

Reads results/epistemic_uncertainty/lstv_uncertainty_metrics.csv, ranks studies
by the given column, selects top N (highest) and bottom N (lowest), then runs
TotalSpineSeg on those studies — skipping any already segmented.

With --all: processes every study in valid_id.npy regardless of uncertainty CSV.

Failed studies from a previous run are retried first before moving on to new ones.

NIfTI paths are identical to those used by 02b_spineps_selective.py:
  results/nifti/{study_id}/{series_id}/sub-{study_id}_acq-sag_T2w.nii.gz
  results/nifti/{study_id}/{series_id}/sub-{study_id}_acq-ax_T2w.nii.gz

Fully self-contained — no imports from other local scripts.

Usage:
    # Selective (top/bottom N):
    python scripts/03b_totalspineseg_selective.py \
        --uncertainty_csv results/epistemic_uncertainty/lstv_uncertainty_metrics.csv \
        --nifti_dir       results/nifti \
        --output_dir      results/totalspineseg \
        --series_csv      data/raw/train_series_descriptions.csv \
        --valid_ids       models/valid_id.npy \
        --top_n           10 \
        --rank_by         l5_s1_confidence

    # Full dataset:
    python scripts/03b_totalspineseg_selective.py \
        --nifti_dir       results/nifti \
        --output_dir      results/totalspineseg \
        --series_csv      data/raw/train_series_descriptions.csv \
        --valid_ids       models/valid_id.npy \
        --all
"""

import argparse
import json
import logging
import os
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

# stderr signatures that mean the container env itself is wrecked (OOM / shm
# exhaustion / killed worker) rather than this one study having bad data. When
# we see these, every subsequent study fails instantly with the same message,
# so we abort the whole run instead of churning through and mislabeling them.
FATAL_ENV_MARKERS = (
    'Intel MKL FATAL ERROR',
    'BrokenProcessPool',
    'Cannot allocate memory',
    'std::bad_alloc',
)


class FatalEnvironmentError(RuntimeError):
    """Raised when TotalSpineSeg fails in a way that poisons the whole run."""


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
# NIFTI RESOLUTION
# ============================================================================

def resolve_nifti(study_dir: Path, series_id: str, study_id: str, acq: str) -> Path | None:
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

        # Cap the worker pool. TotalSpineSeg defaults to os.cpu_count() (the
        # whole node), which oversubscribes RAM under SLURM and triggers
        # OOM -> BrokenProcessPool -> MKL crash. TSS_MAX_WORKERS is set by the
        # slurm script to the allocated CPU count.
        max_workers = os.environ.get('TSS_MAX_WORKERS')
        if max_workers and max_workers.isdigit() and int(max_workers) > 0:
            n = int(max_workers)
            cmd += ['--max-workers', str(n),
                    '--max-workers-nnunet', str(min(n, 4))]

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
            stderr = result.stderr or ''
            logger.error(f"  TotalSpineSeg failed:\n{stderr[-2000:]}")
            if any(marker in stderr for marker in FATAL_ENV_MARKERS):
                raise FatalEnvironmentError(stderr[-500:])
            return None

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

        step1_levels_dir = temp_output / 'step1_levels'
        if step1_levels_dir.exists():
            level_files = sorted(step1_levels_dir.glob("*.nii.gz"))
            if level_files:
                dest = final_dir / f"{study_id}_{acq}_levels.nii.gz"
                shutil.copy(level_files[0], dest)
                outputs['levels'] = dest
                logger.info(f"  ✓ Levels: {dest.name}")

        step1_cord_dir = temp_output / 'step1_cord'
        if step1_cord_dir.exists():
            cord_files = sorted(step1_cord_dir.glob("*.nii.gz"))
            if cord_files:
                dest = final_dir / f"{study_id}_{acq}_cord.nii.gz"
                shutil.copy(cord_files[0], dest)
                outputs['cord'] = dest
                logger.info(f"  ✓ Cord: {dest.name}")

        step1_canal_dir = temp_output / 'step1_canal'
        if step1_canal_dir.exists():
            canal_files = sorted(step1_canal_dir.glob("*.nii.gz"))
            if canal_files:
                dest = final_dir / f"{study_id}_{acq}_canal.nii.gz"
                shutil.copy(canal_files[0], dest)
                outputs['canal'] = dest
                logger.info(f"  ✓ Canal: {dest.name}")

        step2_uncertainties_dir = temp_output / 'step2_uncertainties'
        if step2_uncertainties_dir.exists():
            unc_files = sorted(step2_uncertainties_dir.glob("*.nii.gz"))
            if unc_files:
                dest = final_dir / f"{study_id}_{acq}_unc.nii.gz"
                shutil.copy(unc_files[0], dest)
                outputs['uncertainty'] = dest
                logger.info(f"  ✓ Uncertainty: {dest.name}")
        else:
            logger.warning("  step2_uncertainties/ not found")

        return outputs if 'labeled' in outputs else None

    except subprocess.TimeoutExpired:
        logger.error("  TotalSpineSeg timed out (>15 min)")
        return None
    except FatalEnvironmentError:
        raise  # must escape to main()'s abort handler, not be swallowed below
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


def select_all_studies(nifti_dir: Path, valid_ids: set | None) -> list[str]:
    """Return all study IDs that have NIfTI data and are in valid_ids."""
    all_ids = sorted(d.name for d in nifti_dir.iterdir() if d.is_dir())
    if valid_ids is not None:
        all_ids = [s for s in all_ids if s in valid_ids]
        logger.info(f"Filtered to {len(all_ids)} studies via valid_ids")
    return all_ids


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run TotalSpineSeg on top-N / bottom-N uncertainty studies, or ALL valid studies'
    )
    parser.add_argument('--uncertainty_csv', default=None,
                        help='Path to uncertainty CSV (not required with --all)')
    parser.add_argument('--nifti_dir',       required=True)
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--series_csv',      required=True)
    parser.add_argument('--valid_ids',       default=None)
    parser.add_argument('--top_n',           type=int, default=None)
    parser.add_argument('--rank_by',         default='l5_s1_confidence')
    parser.add_argument('--all',             action='store_true',
                        help='Process every valid study regardless of uncertainty CSV')
    parser.add_argument('--max_studies',     type=int, default=None,
                        help='Process at most this many studies, then exit with code 3 '
                             '(more remain). Bounds process lifetime so a slow resource '
                             'leak cannot accumulate into an Intel MKL / fork crash. '
                             'The slurm wrapper auto-resubmits on exit 3.')
    parser.add_argument('--dry_run',         action='store_true')
    args = parser.parse_args()

    if not args.all and (args.uncertainty_csv is None or args.top_n is None):
        parser.error("--uncertainty_csv and --top_n are required unless --all is set")

    nifti_dir     = Path(args.nifti_dir)
    output_dir    = Path(args.output_dir)
    series_csv    = Path(args.series_csv)
    progress_file = output_dir / 'progress_selective.json'

    output_dir.mkdir(parents=True, exist_ok=True)

    mode_label = 'ALL VALID STUDIES' if args.all else f'top/bottom {args.top_n} by {args.rank_by}'
    logger.info("=" * 70)
    logger.info(f"TOTALSPINESEG SEGMENTATION — {mode_label}")
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

    # Select studies
    if args.all:
        selected_ids = select_all_studies(nifti_dir, valid_ids)
        df_ranked    = None
        logger.info(f"Total studies to process: {len(selected_ids)}")
    else:
        selected_ids, df_ranked = select_studies(
            Path(args.uncertainty_csv), args.top_n, args.rank_by, valid_ids
        )

    skip_already_done = [s for s in selected_ids if already_segmented(s, output_dir)]
    to_run            = [s for s in selected_ids if not already_segmented(s, output_dir)]

    # Load progress early so we can prioritize retrying previously-failed studies
    progress = load_progress(progress_file)

    previously_failed = [s for s in to_run if s in progress.get('failed', [])]
    new_studies       = [s for s in to_run if s not in progress.get('failed', [])]
    ordered_to_run    = previously_failed + new_studies

    logger.info(f"\nSelected:         {len(selected_ids)}")
    logger.info(f"Already done:     {len(skip_already_done)} (skipping)")
    logger.info(f"To run:           {len(ordered_to_run)}")
    if previously_failed:
        logger.info(f"  ↺ Retrying failed: {len(previously_failed)}")
        logger.info(f"  → New studies:     {len(new_studies)}")

    if args.dry_run:
        logger.info("\n--- DRY RUN ---")
        if previously_failed:
            logger.info(f"  [RETRY] {len(previously_failed)} previously-failed studies:")
            for sid in previously_failed:
                logger.info(f"    {sid}")
        logger.info(f"  [NEW]   {len(new_studies)} new studies:")
        for sid in new_studies:
            logger.info(f"    {sid}")
        logger.info(f"--- DRY RUN complete — {len(ordered_to_run)} would be run ---")
        return 0

    if not ordered_to_run:
        logger.info("\nAll selected studies already segmented.")
        return 0

    # Bound this process to a batch so a slow resource leak (semaphores / shm /
    # memory mappings) can't accumulate over hundreds of studies into the
    # "Intel MKL FATAL ERROR: Cannot load <mkl-loader>" / fork crash. The slurm
    # wrapper resubmits a fresh process (exit 3) until nothing remains.
    batch          = ordered_to_run
    leftover_count = 0
    if args.max_studies and args.max_studies > 0 and len(ordered_to_run) > args.max_studies:
        batch          = ordered_to_run[:args.max_studies]
        leftover_count = len(ordered_to_run) - args.max_studies
        logger.info(f"Batch limit:      {args.max_studies} this run "
                    f"({leftover_count} will remain for the next run)")

    series_df = load_series_csv(series_csv)
    if series_df is None:
        logger.error("Cannot load series CSV — aborting")
        return 1

    success_count = 0
    error_count   = 0
    aborted       = False

    for study_id in tqdm(batch, desc='TotalSpineSeg'):
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
                logger.info(f"  ✓  complete")
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
        except FatalEnvironmentError:
            logger.error("\n" + "=" * 70)
            logger.error("FATAL: container environment corrupted (Intel MKL / BrokenProcessPool).")
            logger.error("This is almost always host-RAM or /dev/shm exhaustion from too many")
            logger.error("TotalSpineSeg workers. Aborting now so the remaining studies are NOT")
            logger.error("marked failed in a dead environment.")
            logger.error("Resubmit the job — already-done studies are skipped and failed are")
            logger.error("retried first. Ensure TSS_MAX_WORKERS is set (see slurm script).")
            logger.error("=" * 70)
            mark_failed(progress, study_id)
            save_progress(progress_file, progress)
            error_count += 1
            aborted = True
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
    if aborted:
        logger.info("Status:       ABORTED (environment crash) — resubmit to continue")
        return 2
    if leftover_count:
        logger.info(f"Status:       BATCH DONE — {leftover_count} studies remain, resubmit to continue")
        return 3
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
