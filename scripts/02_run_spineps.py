#!/usr/bin/env python3
"""
SPINEPS Segmentation Pipeline - Refactored

Runs SPINEPS on pre-converted NIfTI files.
Computes centroids for ALL structures and generates uncertainty maps.

Usage:
    python 02_run_spineps.py \
        --nifti_dir results/nifti \
        --output_dir results/spineps \
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


# ============================================================================
# CENTROID COMPUTATION FOR ALL STRUCTURES
# ============================================================================


def compute_all_centroids(instance_mask_path: Path, semantic_mask_path: Path, 
                          ctd_path: Path) -> dict:
    """Compute centroids for ALL structures."""
    if not HAS_NIBABEL:
        logger.warning("nibabel not available - skipping centroid computation")
        return {}
    
    try:
        instance_nii = nib.load(instance_mask_path)
        instance_data = instance_nii.get_fdata().astype(int)
        
        semantic_nii = nib.load(semantic_mask_path)
        semantic_data = semantic_nii.get_fdata().astype(int)
        
        with open(ctd_path) as f:
            ctd_data = json.load(f)
        
        if len(ctd_data) < 2:
            logger.warning(f"Unexpected centroid JSON structure: {ctd_path}")
            return {}
        
        added_counts = {
            'vertebrae': 0, 'discs': 0, 'endplates': 0, 'subregions': 0
        }
        
        # Instance mask (vertebrae, discs, endplates)
        instance_labels = np.unique(instance_data)
        instance_labels = instance_labels[instance_labels > 0]
        
        for label in instance_labels:
            label_str = str(label)
            if label_str in ctd_data[1]:
                continue
            
            mask = (instance_data == label)
            if mask.sum() == 0:
                continue
            
            centroid = center_of_mass(mask)
            ctd_data[1][label_str] = {'50': list(centroid)}
            
            if label <= 28:
                added_counts['vertebrae'] += 1
            elif 119 <= label <= 126:
                added_counts['discs'] += 1
            elif label >= 200:
                added_counts['endplates'] += 1
        
        # Semantic mask (subregions)
        semantic_labels = np.unique(semantic_data)
        semantic_labels = semantic_labels[semantic_labels > 0]
        
        for label in semantic_labels:
            label_str = str(label)
            if label_str in ctd_data[1]:
                continue
            
            mask = (semantic_data == label)
            if mask.sum() == 0:
                continue
            
            centroid = center_of_mass(mask)
            ctd_data[1][label_str] = {'50': list(centroid)}
            added_counts['subregions'] += 1
        
        with open(ctd_path, 'w') as f:
            json.dump(ctd_data, f, indent=2)
        
        return added_counts
    
    except Exception as e:
        logger.warning(f"Error computing centroids: {e}")
        logger.debug(traceback.format_exc())
        return {}


# ============================================================================
# UNCERTAINTY MAP COMPUTATION
# ============================================================================


def compute_uncertainty_from_softmax(derivatives_dir: Path, study_id: str, 
                                     seg_dir: Path) -> bool:
    """Compute uncertainty map from softmax logits."""
    if not HAS_NIBABEL:
        return False
    
    try:
        logits_pattern = f"*{study_id}*logit*.npz"
        logits_files = list(derivatives_dir.glob(logits_pattern))
        
        if not logits_files:
            return False
        
        logits_data = np.load(logits_files[0])
        softmax = logits_data['arr_0']
        
        uncertainty = 1.0 - np.max(softmax, axis=-1)
        
        semantic_mask = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
        if not semantic_mask.exists():
            return False
        
        ref_nii = nib.load(semantic_mask)
        unc_nii = nib.Nifti1Image(
            uncertainty.astype(np.float32), 
            ref_nii.affine, 
            ref_nii.header
        )
        
        unc_path = seg_dir / f"{study_id}_unc.nii.gz"
        nib.save(unc_nii, unc_path)
        
        logger.info(f"  ✓ Uncertainty map saved")
        return True
    
    except Exception as e:
        logger.warning(f"Could not compute uncertainty map: {e}")
        logger.debug(traceback.format_exc())
        return False


# ============================================================================
# SPINEPS EXECUTION
# ============================================================================


def run_spineps(nifti_path: Path, seg_dir: Path, study_id: str) -> dict:
    """Run SPINEPS segmentation."""
    try:
        seg_dir.mkdir(parents=True, exist_ok=True)
        
        import os
        env = os.environ.copy()
        env['SPINEPS_SEGMENTOR_MODELS'] = '/app/models'
        env['SPINEPS_ENVIRONMENT_DIR'] = '/app/models'
        
        cmd = [
            'python', '-m', 'spineps.entrypoint', 'sample',
            '-i', str(nifti_path),
            '-model_semantic', 't2w',
            '-model_instance', 'instance',
            '-model_labeling', 't2w_labeling',
            '-save_softmax_logits',
            '-override_semantic',
            '-override_instance',
            '-override_ctd'
        ]
        
        logger.info("  Running SPINEPS...")
        sys.stdout.flush()
        result = subprocess.run(
            cmd, 
            stdout=None, 
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=600, 
            env=env
        )
        sys.stdout.flush()
        
        if result.returncode != 0:
            logger.error(f"  SPINEPS failed:\n{result.stderr}")
            return None
        
        derivatives_base = nifti_path.parent / "derivatives_seg"
        if not derivatives_base.exists():
            logger.error(f"  derivatives_seg not found at: {derivatives_base}")
            return None
        
        def find_file(exact_name: str, glob_pattern: str) -> Path:
            f = derivatives_base / exact_name
            if not f.exists():
                matches = list(derivatives_base.glob(glob_pattern))
                f = matches[0] if matches else None
            return f if (f and f.exists()) else None
        
        outputs = {}
        
        # Instance segmentation
        f = find_file(f"sub-{study_id}_mod-T2w_seg-vert_msk.nii.gz", "*_seg-vert_msk.nii.gz")
        if f:
            dest = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
            shutil.copy(f, dest)
            outputs['instance_mask'] = dest
            logger.info("  ✓ Instance mask (seg-vert)")
        else:
            logger.warning("  ⚠ Instance mask not found")
        
        # Semantic segmentation
        f = find_file(f"sub-{study_id}_mod-T2w_seg-spine_msk.nii.gz", "*_seg-spine_msk.nii.gz")
        if f:
            dest = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
            shutil.copy(f, dest)
            outputs['semantic_mask'] = dest
            logger.info("  ✓ Semantic mask (seg-spine)")
        
        # Sub-region segmentation
        f = find_file(f"sub-{study_id}_mod-T2w_seg-subreg_msk.nii.gz", "*_seg-subreg_msk.nii.gz")
        if f:
            dest = seg_dir / f"{study_id}_seg-subreg_msk.nii.gz"
            shutil.copy(f, dest)
            outputs['subreg_mask'] = dest
            logger.info("  ✓ Sub-region mask (seg-subreg)")
        
        # Centroids JSON
        f = find_file(f"sub-{study_id}_mod-T2w_ctd.json", "*_ctd.json")
        if f:
            dest = seg_dir / f"{study_id}_ctd.json"
            shutil.copy(f, dest)
            outputs['centroid_json'] = dest
            logger.info("  ✓ Centroids JSON (ctd)")
            
            # Add ALL centroids
            if 'instance_mask' in outputs and 'semantic_mask' in outputs:
                counts = compute_all_centroids(
                    outputs['instance_mask'],
                    outputs['semantic_mask'],
                    dest
                )
                if counts:
                    total = sum(counts.values())
                    logger.info(f"  ✓ Added {total} centroids: "
                              f"{counts.get('discs', 0)} discs, "
                              f"{counts.get('endplates', 0)} endplates, "
                              f"{counts.get('subregions', 0)} subregions")
        
        # Uncertainty map (from softmax logits)
        if 'semantic_mask' in outputs:
            unc_computed = compute_uncertainty_from_softmax(derivatives_base, study_id, seg_dir)
            if unc_computed:
                outputs['uncertainty_map'] = seg_dir / f"{study_id}_unc.nii.gz"
        
        if 'instance_mask' not in outputs:
            logger.error("  Instance mask missing — treating as failure")
            return None
        
        return outputs if outputs else None
    
    except subprocess.TimeoutExpired:
        logger.error("  SPINEPS timed out (>600s)")
        sys.stdout.flush()
        return None
    except Exception as e:
        logger.error(f"  SPINEPS error: {e}")
        logger.debug(traceback.format_exc())
        sys.stdout.flush()
        return None


# ============================================================================
# PROGRESS TRACKING
# ============================================================================


def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            logger.info(
                f"Resuming: {len(progress['success'])} done, "
                f"{len(progress['failed'])} failed previously"
            )
            return progress
        except Exception as e:
            logger.warning(f"Could not load progress file: {e} — starting fresh")
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


def save_metadata(study_id: str, outputs: dict, metadata_dir: Path):
    """Save metadata for completed study."""
    metadata = {
        'study_id':  study_id,
        'outputs':   {k: str(v) for k, v in outputs.items()},
        'timestamp': pd.Timestamp.now().isoformat()
    }
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / f"{study_id}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='SPINEPS Segmentation Pipeline'
    )
    parser.add_argument('--nifti_dir', required=True,
                       help='Directory with NIfTI files')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for segmentations')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--mode', choices=['trial', 'debug', 'prod'], 
                       default='prod')
    parser.add_argument('--retry-failed', action='store_true',
                       help='Retry previously failed studies')
    
    args = parser.parse_args()
    
    nifti_dir = Path(args.nifti_dir)
    output_dir = Path(args.output_dir)
    seg_dir = output_dir / 'segmentations'
    metadata_dir = output_dir / 'metadata'
    progress_file = output_dir / 'progress.json'
    
    seg_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    progress = load_progress(progress_file)
    
    # Determine which studies to skip
    if args.retry_failed:
        # Only skip successful ones
        already_processed = set(progress.get('success', []))
        logger.info(f"Retry mode: Will retry {len(progress.get('failed', []))} failed studies")
    else:
        # Skip both successful and failed
        already_processed = set(progress['processed'])
    
    # Find sagittal T2 NIfTI files in per-study subdirectories
    study_dirs = sorted([d for d in nifti_dir.iterdir() if d.is_dir()])
    
    # Filter to only those with sagittal T2
    nifti_files = []
    for study_dir in study_dirs:
        sag_t2 = study_dir / "sag_t2.nii.gz"
        if sag_t2.exists():
            nifti_files.append(sag_t2)
    
    if args.mode == 'debug':
        nifti_files = nifti_files[:1]
    elif args.mode == 'trial':
        nifti_files = nifti_files[:3]
    elif args.limit:
        nifti_files = nifti_files[:args.limit]
    
    # Extract study IDs and filter
    study_files = []
    for f in nifti_files:
        study_id = f.parent.name  # Parent directory is the study ID
        if study_id not in already_processed:
            study_files.append((study_id, f))
    
    remaining = len(study_files)
    skipped = len(nifti_files) - remaining
    
    logger.info("=" * 70)
    logger.info("SPINEPS SEGMENTATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Mode:             {args.mode}")
    logger.info(f"Total studies:    {len(nifti_files)}")
    logger.info(f"Already done:     {skipped}")
    logger.info(f"To process:       {remaining}")
    logger.info(f"Output root:      {output_dir}")
    logger.info("")
    logger.info("Features enabled:")
    logger.info("  ✓ All structure centroids (vertebrae, discs, endplates, subregions)")
    logger.info("  ✓ Uncertainty maps")
    logger.info("  ✓ DICOM-aligned masks for overlay")
    logger.info("=" * 70)
    sys.stdout.flush()
    
    success_count = len(progress['success'])
    error_count = len(progress['failed'])
    
    for study_id, nifti_path in tqdm(study_files, desc="Studies"):
        logger.info(f"\n[{study_id}]")
        sys.stdout.flush()
        
        try:
            outputs = run_spineps(nifti_path, seg_dir, study_id)
            
            if outputs:
                # Save metadata
                save_metadata(study_id, outputs, metadata_dir)
                
                # Remove from failed list if retrying
                if study_id in progress.get('failed', []):
                    progress['failed'].remove(study_id)
                
                if study_id not in progress['processed']:
                    progress['processed'].append(study_id)
                if study_id not in progress['success']:
                    progress['success'].append(study_id)
                
                save_progress(progress_file, progress)
                success_count += 1
                logger.info(f"  ✓ Done ({len(outputs)} outputs)")
                sys.stdout.flush()
            else:
                logger.warning(f"  ✗ SPINEPS failed")
                
                if study_id not in progress['processed']:
                    progress['processed'].append(study_id)
                if study_id not in progress['failed']:
                    progress['failed'].append(study_id)
                
                save_progress(progress_file, progress)
                error_count += 1
                sys.stdout.flush()
        
        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted — progress saved")
            sys.stdout.flush()
            break
        except Exception as e:
            logger.error(f"  ✗ Unexpected error: {e}")
            logger.debug(traceback.format_exc())
            
            if study_id not in progress['processed']:
                progress['processed'].append(study_id)
            if study_id not in progress['failed']:
                progress['failed'].append(study_id)
            
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
    logger.info("Directory structure:")
    logger.info(f"  Input:  {nifti_dir}/<study_id>/sag_t2.nii.gz")
    logger.info(f"          {nifti_dir}/<study_id>/derivatives_seg/  (SPINEPS temp)")
    logger.info("")
    logger.info(f"  Output: {seg_dir}/")
    logger.info("    ├── <study_id>_seg-vert_msk.nii.gz")
    logger.info("    ├── <study_id>_seg-spine_msk.nii.gz")
    logger.info("    ├── <study_id>_seg-subreg_msk.nii.gz")
    logger.info("    ├── <study_id>_ctd.json")
    logger.info("    └── <study_id>_unc.nii.gz")
    logger.info("")
    logger.info(f"  Metadata: {metadata_dir}/<study_id>_metadata.json")
    sys.stdout.flush()
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
