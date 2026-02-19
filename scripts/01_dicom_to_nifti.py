#!/usr/bin/env python3
"""
DICOM to NIfTI Converter - Pure Python Implementation

Converts DICOM studies to NIfTI format using pydicom + nibabel.
No external dependencies (dcm2niix not required).

Usage:
    python 01_dicom_to_nifti.py \
        --input_dir data/raw/train_images \
        --series_csv data/raw/train_series_descriptions.csv \
        --output_dir results/nifti \
        --mode prod
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

try:
    import pydicom
    import nibabel as nib
    HAS_DICOM = True
except ImportError:
    HAS_DICOM = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_series_descriptions(csv_path: Path) -> pd.DataFrame:
    """Load series descriptions CSV."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} series descriptions")
        return df
    except Exception as e:
        logger.error(f"Failed to load series CSV: {e}")
        return None


def select_best_series(study_dir: Path, series_df: pd.DataFrame, study_id: str, 
                       modality: str) -> Path:
    """
    Select best series for given modality.
    
    Args:
        modality: 'sagittal_t2' or 'axial_t2'
    """
    if series_df is None:
        return None
    
    try:
        study_series = series_df[series_df['study_id'] == int(study_id)]
        
        if len(study_series) == 0:
            return None
        
        # Define priorities for each modality
        if modality == 'sagittal_t2':
            priorities = ['Sagittal T2', 'Sagittal T2/STIR', 'SAG T2']
        elif modality == 'axial_t2':
            priorities = ['Axial T2', 'AXIAL T2', 'AXI T2']
        else:
            return None
        
        # Try each priority
        for priority in priorities:
            matching = study_series[
                study_series['series_description'].str.contains(
                    priority, case=False, na=False
                )
            ]
            if len(matching) > 0:
                series_id = str(matching.iloc[0]['series_id'])
                series_path = study_dir / series_id
                if series_path.exists():
                    return series_path
        
        return None
    
    except Exception as e:
        logger.debug(f"Error selecting series: {e}")
        return None


def load_dicom_series(dicom_dir: Path) -> tuple:
    """
    Load all DICOM files from a directory and sort by instance number.
    
    Returns:
        (slices, affine, header_info) or (None, None, None) on error
    """
    if not HAS_DICOM:
        logger.error("pydicom not available")
        return None, None, None
    
    try:
        # Find all DICOM files
        dcm_files = list(dicom_dir.glob("*.dcm"))
        if not dcm_files:
            # Try without extension
            dcm_files = [f for f in dicom_dir.iterdir() if f.is_file()]
        
        if not dcm_files:
            return None, None, None
        
        # Load all slices
        slices = []
        for dcm_file in dcm_files:
            try:
                ds = pydicom.dcmread(str(dcm_file), force=True)
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
            except Exception:
                continue
        
        if not slices:
            return None, None, None
        
        # Sort by instance number or position
        slices.sort(key=lambda x: float(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
        
        # Get first slice for metadata
        ref_ds = slices[0]
        
        # Build 3D volume
        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=-1)
        
        # Get voxel spacing
        pixel_spacing = ref_ds.PixelSpacing if hasattr(ref_ds, 'PixelSpacing') else [1.0, 1.0]
        slice_thickness = ref_ds.SliceThickness if hasattr(ref_ds, 'SliceThickness') else 1.0
        
        # Build affine matrix
        # This is simplified - for production use dcm2niix or more sophisticated conversion
        affine = np.eye(4)
        affine[0, 0] = float(pixel_spacing[1])  # Column spacing
        affine[1, 1] = float(pixel_spacing[0])  # Row spacing
        affine[2, 2] = float(slice_thickness)
        
        # Try to get position from ImagePositionPatient
        if hasattr(ref_ds, 'ImagePositionPatient'):
            pos = ref_ds.ImagePositionPatient
            affine[0, 3] = float(pos[0])
            affine[1, 3] = float(pos[1])
            affine[2, 3] = float(pos[2])
        
        header_info = {
            'StudyInstanceUID': str(ref_ds.StudyInstanceUID) if hasattr(ref_ds, 'StudyInstanceUID') else '',
            'SeriesInstanceUID': str(ref_ds.SeriesInstanceUID) if hasattr(ref_ds, 'SeriesInstanceUID') else '',
            'SeriesDescription': str(ref_ds.SeriesDescription) if hasattr(ref_ds, 'SeriesDescription') else '',
            'n_slices': len(slices),
            'pixel_spacing': pixel_spacing,
            'slice_thickness': slice_thickness
        }
        
        return volume, affine, header_info
    
    except Exception as e:
        logger.error(f"Error loading DICOM series: {e}")
        return None, None, None


def convert_dicom_to_nifti(dicom_dir: Path, output_path: Path, 
                           series_type: str) -> Path:
    """
    Convert DICOM series to NIfTI using pydicom + nibabel.
    
    Args:
        series_type: 'sag_t2' or 'axial_t2' for naming
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load DICOM series
        volume, affine, header_info = load_dicom_series(dicom_dir)
        
        if volume is None:
            logger.error(f"Failed to load DICOM series from {dicom_dir}")
            return None
        
        # Create NIfTI image
        nii = nib.Nifti1Image(volume, affine)
        
        # Save
        nib.save(nii, output_path)
        
        logger.info(f"  ✓ Converted: {volume.shape} → {output_path.name}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"DICOM conversion failed: {e}")
        return None


def save_metadata(study_id: str, conversions: dict, metadata_dir: Path):
    """Save conversion metadata."""
    metadata = {
        'study_id': study_id,
        'conversions': conversions,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / f"{study_id}_conversion.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def load_progress(progress_file: Path) -> dict:
    """Load progress from previous run."""
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {'processed': [], 'success': [], 'failed': []}


def save_progress(progress_file: Path, progress: dict):
    """Save progress."""
    try:
        tmp = progress_file.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(progress, f, indent=2)
        tmp.replace(progress_file)
    except Exception as e:
        logger.warning(f"Could not save progress: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert DICOM to NIfTI for LSTV detection pipeline'
    )
    parser.add_argument('--input_dir', required=True, 
                       help='DICOM input directory')
    parser.add_argument('--series_csv', required=True, 
                       help='Series descriptions CSV')
    parser.add_argument('--output_dir', required=True, 
                       help='NIfTI output directory')
    parser.add_argument('--limit', type=int, default=None, 
                       help='Limit number of studies')
    parser.add_argument('--mode', choices=['trial', 'debug', 'prod'], 
                       default='prod')
    
    args = parser.parse_args()
    
    if not HAS_DICOM:
        logger.error("pydicom and/or nibabel not installed!")
        logger.error("Install with: pip install pydicom nibabel")
        return 1
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metadata_dir = output_dir / 'metadata'
    progress_file = output_dir / 'conversion_progress.json'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    series_df = load_series_descriptions(Path(args.series_csv))
    progress = load_progress(progress_file)
    already_processed = set(progress['processed'])
    
    # Get study list
    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    if args.mode == 'debug':
        study_dirs = study_dirs[:1]
    elif args.mode == 'trial':
        study_dirs = study_dirs[:3]
    elif args.limit:
        study_dirs = study_dirs[:args.limit]
    
    remaining = [d for d in study_dirs if d.name not in already_processed]
    
    logger.info("=" * 70)
    logger.info("DICOM TO NIFTI CONVERSION")
    logger.info("=" * 70)
    logger.info(f"Mode:         {args.mode}")
    logger.info(f"Total:        {len(study_dirs)}")
    logger.info(f"Already done: {len(study_dirs) - len(remaining)}")
    logger.info(f"To process:   {len(remaining)}")
    logger.info(f"Output:       {output_dir}")
    logger.info("=" * 70)
    
    success_count = len(progress['success'])
    error_count = len(progress['failed'])
    
    for study_dir in tqdm(remaining, desc="Converting"):
        study_id = study_dir.name
        logger.info(f"\n[{study_id}]")
        
        # Create per-study NIfTI directory
        study_nifti_dir = output_dir / study_id
        study_nifti_dir.mkdir(parents=True, exist_ok=True)
        
        conversions = {}
        
        try:
            # Convert sagittal T2
            sag_series = select_best_series(
                study_dir, series_df, study_id, 'sagittal_t2'
            )
            
            if sag_series:
                logger.info(f"  Sagittal T2: {sag_series.name}")
                sag_output = study_nifti_dir / "sag_t2.nii.gz"
                sag_nifti = convert_dicom_to_nifti(
                    sag_series, sag_output, 'sag_t2'
                )
                if sag_nifti:
                    conversions['sagittal_t2'] = {
                        'series_id': sag_series.name,
                        'nifti_path': str(sag_nifti)
                    }
            else:
                logger.warning(f"  ⚠ No sagittal T2 series found")
            
            # Convert axial T2
            axial_series = select_best_series(
                study_dir, series_df, study_id, 'axial_t2'
            )
            
            if axial_series:
                logger.info(f"  Axial T2: {axial_series.name}")
                axial_output = study_nifti_dir / "axial_t2.nii.gz"
                axial_nifti = convert_dicom_to_nifti(
                    axial_series, axial_output, 'axial_t2'
                )
                if axial_nifti:
                    conversions['axial_t2'] = {
                        'series_id': axial_series.name,
                        'nifti_path': str(axial_nifti)
                    }
            else:
                logger.warning(f"  ⚠ No axial T2 series found")
            
            # Save metadata
            if conversions:
                save_metadata(study_id, conversions, metadata_dir)
                progress['processed'].append(study_id)
                progress['success'].append(study_id)
                save_progress(progress_file, progress)
                success_count += 1
                logger.info(f"  ✓ Done ({len(conversions)} series)")
            else:
                logger.warning(f"  ✗ No series converted")
                progress['processed'].append(study_id)
                progress['failed'].append(study_id)
                save_progress(progress_file, progress)
                error_count += 1
        
        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted - progress saved")
            break
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
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
    logger.info("")
    logger.info("Directory structure:")
    logger.info(f"  {output_dir}/")
    logger.info("    ├── <study_id>/")
    logger.info("    │   ├── sag_t2.nii.gz")
    logger.info("    │   └── axial_t2.nii.gz")
    logger.info("    └── metadata/")
    logger.info("        └── <study_id>_conversion.json")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. sbatch slurm_scripts/02_spineps.sh")
    logger.info("  2. sbatch slurm_scripts/03_totalspineseg.sh")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
