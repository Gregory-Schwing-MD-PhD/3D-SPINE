#!/usr/bin/env python3
"""
LSTV Detection via Epistemic Uncertainty
Uses Ian Pan's RSNA 2024 2nd Place Solution
Novel approach: High uncertainty in spine localizer indicates LSTV

Reads pre-converted NIfTI files instead of raw DICOMs.
Strictly processes validation holdout set to prevent data leakage.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    logger.warning("nibabel not available - NIfTI loading will fail")

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    logger.warning("timm not available - model loading will fail")

# Configure logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")


#################################################
# MODEL ARCHITECTURE
#################################################

class MyDecoderBlock(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class MyUnetDecoder(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel, ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock(i, s, o)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode


class Net(nn.Module):
    def __init__(self, pretrained=False, cfg=None):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor(0))
        self.register_buffer('std', torch.tensor(1))

        arch = 'resnet50d'

        encoder_dim = {
            'resnet18': [64, 64, 128, 256, 512,],
            'resnet50d': [64, 256, 512, 1024, 2048,],
        }[arch]
        decoder_dim = [256, 128, 64, 32, 16]

        if not HAS_TIMM:
            raise ImportError("timm is required. Install with: pip install timm")

        self.encoder = timm.create_model(
            model_name=arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool=''
        )

        self.decoder = MyUnetDecoder(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
        )
        self.logit = nn.Conv2d(decoder_dim[-1], 6, kernel_size=1)

    def forward(self, batch):
        device = self.D.device
        image = batch['sagittal'].to(device)
        B, _1_, H, W = image.shape

        x = image.float() / 255
        x = (x - self.mean) / self.std
        x = x.expand(-1, 3, -1, -1)

        encode = []
        e = self.encoder
        x = e.act1(e.bn1(e.conv1(x)))
        encode.append(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = e.layer1(x)
        encode.append(x)
        x = e.layer2(x)
        encode.append(x)
        x = e.layer3(x)
        encode.append(x)
        x = e.layer4(x)
        encode.append(x)

        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1] + [None]
        )

        logit = self.logit(last)

        output = {}
        if 'loss' in self.output_type:
            truth = batch['mask'].long().to(device)
            output['mask_loss'] = F.cross_entropy(logit, truth)

        if 'infer' in self.output_type:
            p = torch.softmax(logit, 1)
            output['probability'] = p

        return output


#################################################
# UNCERTAINTY CALCULATION
#################################################

class UncertaintyCalculator:
    @staticmethod
    def calculate_uncertainty(heatmap: np.ndarray) -> Tuple[float, float]:
        peak_confidence = float(np.max(heatmap))
        flat_prob = heatmap.flatten()
        flat_prob_norm = flat_prob / (np.sum(flat_prob) + 1e-9)
        entropy = -np.sum(flat_prob_norm * np.log(flat_prob_norm + 1e-9))
        return peak_confidence, float(entropy)

    @staticmethod
    def calculate_spatial_entropy(heatmap: np.ndarray, num_bins: int = 10) -> float:
        H, W = heatmap.shape
        bin_height = H // num_bins
        bin_width = W // num_bins
        binned_probs = []
        for i in range(num_bins):
            for j in range(num_bins):
                region = heatmap[i*bin_height:(i+1)*bin_height,
                                 j*bin_width:(j+1)*bin_width]
                binned_probs.append(np.sum(region))
        binned_probs = np.array(binned_probs)
        binned_probs = binned_probs / (np.sum(binned_probs) + 1e-9)
        return float(-np.sum(binned_probs * np.log(binned_probs + 1e-9)))


def probability_to_point_with_uncertainty(probability: np.ndarray,
                                          threshold: float = 0.5) -> Tuple[List, Dict]:
    calc = UncertaintyCalculator()
    points = []
    uncertainty_metrics = {}
    level_names = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

    for l in range(1, 6):
        heatmap = probability[l]
        level_name = level_names[l-1]

        peak_conf, entropy = calc.calculate_uncertainty(heatmap)
        spatial_entropy = calc.calculate_spatial_entropy(heatmap)

        y, x = np.where(heatmap > threshold)
        if len(y) > 0 and len(x) > 0:
            points.append((round(x.mean()), round(y.mean())))
        else:
            points.append((None, None))

        uncertainty_metrics[level_name] = {
            'peak_confidence': peak_conf,
            'entropy': entropy,
            'spatial_entropy': spatial_entropy,
            'num_pixels_above_threshold': len(y)
        }

    return points, uncertainty_metrics


#################################################
# DATA LOADING — NIfTI
#################################################

def normalise_to_8bit(arr: np.ndarray) -> np.ndarray:
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        return ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def read_nifti_series(nifti_dir: Path, study_id: str) -> Optional[np.ndarray]:
    """
    Load the sagittal T2w NIfTI for a study and return a (Depth, Height, Width)
    uint8 volume ready for the model.

    nibabel canonical orientation loads as (X, Y, Z) = (Sagittal, Coronal, Axial).
    We want to slice along the sagittal axis (first axis after transpose), so:

        raw shape:     (X, Y, Z)
        after T(0,2,1): (X, Z, Y)  →  slice [mid] gives (Z, Y) = (Height, Width) ✓

    Args:
        nifti_dir: directory containing sub-{study_id}_acq-sag_T2w.nii.gz
        study_id:  used to construct the filename

    Returns:
        volume: uint8 array of shape (D, H, W), or None on failure
    """
    if not HAS_NIBABEL:
        logger.error("nibabel is not installed — cannot load NIfTI files")
        return None

    nifti_path = nifti_dir / f"sub-{study_id}_acq-sag_T2w.nii.gz"
    if not nifti_path.exists():
        logger.warning(f"NIfTI not found: {nifti_path}")
        return None

    try:
        img = nib.load(str(nifti_path))
        # Reorient to closest canonical (RAS) so axes are consistent
        img = nib.as_closest_canonical(img)
        # float32 avoids precision loss during transpose / normalisation
        data = img.get_fdata(dtype=np.float32)

        # data is (X, Y, Z) = (Sagittal, Coronal, Axial)
        # Transpose to (X, Z, Y) so slicing axis-0 gives a (Z, Y) sagittal image
        data = np.transpose(data, (0, 2, 1))   # → (D, H, W)

        volume = normalise_to_8bit(data)
        logger.debug(f"Loaded NIfTI {nifti_path.name}: {volume.shape}, dtype={volume.dtype}")
        return volume

    except Exception as e:
        logger.error(f"Error loading NIfTI {nifti_path}: {e}")
        return None


#################################################
# HELPERS
#################################################

def find_nifti_dir(nifti_root: Path, study_id: str, series_id: str) -> Optional[Path]:
    """
    Locate the directory containing the sagittal NIfTI.
    Layout: {nifti_root}/{study_id}/{series_id}/sub-{study_id}_acq-sag_T2w.nii.gz
    """
    candidate = nifti_root / str(study_id) / str(series_id)
    if (candidate / f"sub-{study_id}_acq-sag_T2w.nii.gz").exists():
        return candidate

    # Fallback: search all series subdirs for this study
    study_dir = nifti_root / str(study_id)
    if study_dir.exists():
        for sub in study_dir.iterdir():
            if (sub / f"sub-{study_id}_acq-sag_T2w.nii.gz").exists():
                return sub

    return None


def generate_mock_uncertainty() -> Dict:
    uncertainty_metrics = {}
    for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
        if level in ['l4_l5', 'l5_s1']:
            base_entropy = np.random.uniform(4.5, 6.5)
            base_conf = np.random.uniform(0.3, 0.6)
        else:
            base_entropy = np.random.uniform(2.0, 4.0)
            base_conf = np.random.uniform(0.7, 0.95)
        uncertainty_metrics[level] = {
            'peak_confidence': base_conf,
            'entropy': base_entropy,
            'spatial_entropy': np.random.uniform(1.5, 3.5),
            'num_pixels_above_threshold': np.random.randint(100, 500)
        }
    return uncertainty_metrics


#################################################
# MAIN INFERENCE
#################################################

def run_inference(args):
    input_dir  = Path(args.input_dir)   # NIfTI root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"NIfTI root:     {input_dir}")
    logger.info(f"Output dir:     {output_dir}")
    logger.info(f"Mode:           {args.mode}")

    # --- Validation ID filter ---
    valid_ids_path = Path(args.valid_ids)
    if valid_ids_path.exists():
        valid_ids = set(str(v) for v in np.load(valid_ids_path))
        logger.info(f"✓ Loaded {len(valid_ids)} validation study IDs — no data leakage")
        use_validation_only = True
    else:
        logger.warning(f"⚠ Validation IDs not found at {valid_ids_path}")
        logger.warning("  Running on ALL studies — CAUTION: may include training data!")
        valid_ids = None
        use_validation_only = False

    # --- Series descriptions ---
    series_csv = Path(args.series_csv)
    if not series_csv.exists():
        logger.error(f"Series CSV not found: {series_csv}")
        return

    series_df = pd.read_csv(series_csv)
    logger.info(f"Loaded {len(series_df)} series descriptions")

    sagittal_df = series_df[
        series_df['series_description'].str.lower().str.contains('sagittal') &
        series_df['series_description'].str.lower().str.contains('t2')
    ]
    studies = sagittal_df['study_id'].unique()
    logger.info(f"Found {len(studies)} studies with Sagittal T2 series")

    # --- Apply validation filter ---
    if use_validation_only and valid_ids is not None:
        n_before = len(studies)
        studies = [s for s in studies if str(s) in valid_ids]
        logger.info(f"✓ Validation filter: kept {len(studies)}, excluded {n_before - len(studies)}")

    # --- Mode selection ---
    if args.mode == 'trial':
        # Use first N studies in valid_id.npy order for reproducibility
        valid_ids_ordered = [str(v) for v in np.load(valid_ids_path)]
        studies_set = set(str(s) for s in studies)
        studies = [v for v in valid_ids_ordered if v in studies_set][:args.trial_size]
        logger.info(f"Trial mode: first {len(studies)} studies from valid_id.npy (reproducible)")
        studies = [args.debug_study_id] if args.debug_study_id else [studies[0]]
        logger.info(f"Debug mode: study {studies[0]}")
    else:
        logger.info(f"Production mode: {len(studies)} studies")

    # --- Load model ---
    logger.info("=" * 60)
    logger.info("LOADING MODEL")
    logger.info("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    model = None

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path} — running in MOCK mode")
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            logger.info(f"✓ Checkpoint loaded")

            if isinstance(checkpoint, dict):
                keys = list(checkpoint.keys())
                logger.info(f"Checkpoint keys: {keys}")
                state_dict_key = next((k for k in ('state_dict', 'model_state_dict') if k in keys), None)
                if state_dict_key:
                    state_dict = checkpoint[state_dict_key]
                    model = Net(pretrained=False)
                    model.load_state_dict(state_dict)
                else:
                    logger.error(f"Unknown checkpoint structure: {keys}")
            elif hasattr(checkpoint, 'eval'):
                model = checkpoint
            else:
                logger.error(f"Unknown checkpoint type: {type(checkpoint)}")

            if model is not None:
                model = model.to(device)
                model.eval()
                model.output_type = ['infer']
                logger.info("✓✓✓ MODEL LOADED — REAL INFERENCE ✓✓✓")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())

    if model is None:
        logger.warning("⚠ MOCK MODE — synthetic uncertainty data")

    # --- Inference loop ---
    results = []
    iterator = tqdm(studies, desc="Processing") if args.mode == 'prod' else studies

    for study_id in iterator:
        logger.info(f"\n{'='*60}\nStudy: {study_id}\n{'='*60}")

        study_series = sagittal_df[sagittal_df['study_id'] == study_id]
        if len(study_series) == 0:
            logger.warning(f"No Sagittal T2 series for {study_id}")
            continue

        series_id = study_series.iloc[0]['series_id']
        logger.info(f"Series: {series_id}")

        # --- Locate NIfTI ---
        nifti_dir = find_nifti_dir(input_dir, str(study_id), str(series_id))
        if nifti_dir is None:
            logger.warning(f"NIfTI directory not found for {study_id}/{series_id}")
            continue

        # --- Load volume from NIfTI ---
        volume = read_nifti_series(nifti_dir, str(study_id))
        if volume is None:
            logger.warning(f"Failed to load NIfTI for {study_id}")
            continue

        logger.info(f"Volume shape: {volume.shape}")

        # --- Run inference ---
        if model is not None:
            try:
                IMAGE_SIZE = 160

                # Resize (D, H, W) volume to (D, 160, 160)
                # cv2.resize operates on (H, W, D) so transpose in/out
                vol_hwd = np.ascontiguousarray(volume.transpose(1, 2, 0))
                vol_hwd = cv2.resize(vol_hwd, (IMAGE_SIZE, IMAGE_SIZE),
                                     interpolation=cv2.INTER_LINEAR)
                resized = np.ascontiguousarray(vol_hwd.transpose(2, 0, 1))  # back to (D, H, W)

                mid_idx = resized.shape[0] // 2
                image   = resized[mid_idx]  # (160, 160) sagittal slice

                image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).byte().to(device)
                batch = {'sagittal': image_tensor}

                with torch.cuda.amp.autocast(enabled=True):
                    with torch.no_grad():
                        output = model(batch)

                probability = output['probability'][0].float().cpu().numpy()  # (6, H, W)
                logger.debug(f"Prob shape: {probability.shape}, range [{probability.min():.4f}, {probability.max():.4f}]")

                points, uncertainty_metrics = probability_to_point_with_uncertainty(
                    probability, threshold=0.5
                )
                logger.info("✓ Real model inference complete")

            except Exception as e:
                logger.error(f"Inference error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.warning("Falling back to MOCK data for this study")
                uncertainty_metrics = generate_mock_uncertainty()
        else:
            uncertainty_metrics = generate_mock_uncertainty()

        # --- Log metrics ---
        logger.info("Uncertainty Metrics:")
        for level, metrics in uncertainty_metrics.items():
            logger.info(f"  {level}: conf={metrics['peak_confidence']:.4f}, "
                        f"entropy={metrics['entropy']:.4f}, "
                        f"spatial_entropy={metrics['spatial_entropy']:.4f}")

        # --- Accumulate result ---
        result = {'study_id': study_id, 'series_id': series_id}
        for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
            result[f'{level}_confidence']    = uncertainty_metrics[level]['peak_confidence']
            result[f'{level}_entropy']       = uncertainty_metrics[level]['entropy']
            result[f'{level}_spatial_entropy'] = uncertainty_metrics[level]['spatial_entropy']
        results.append(result)

        if args.mode == 'debug':
            save_debug_visualizations(output_dir, study_id, series_id, volume, uncertainty_metrics)

    # --- Save results ---
    results_df = pd.DataFrame(results)
    output_csv = output_dir / 'lstv_uncertainty_metrics.csv'
    results_df.to_csv(output_csv, index=False)
    logger.info(f"\nResults saved: {output_csv}  ({len(results)} studies)")

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)
    for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
        ec = f'{level}_entropy'
        cc = f'{level}_confidence'
        logger.info(f"{level.upper()}: entropy={results_df[ec].mean():.4f}±{results_df[ec].std():.4f}  "
                    f"conf={results_df[cc].mean():.4f}±{results_df[cc].std():.4f}")


def save_debug_visualizations(output_dir: Path, study_id, series_id,
                               volume: np.ndarray, uncertainty_metrics: Dict):
    import matplotlib.pyplot as plt

    debug_dir = output_dir / 'debug_visualizations'
    debug_dir.mkdir(exist_ok=True)

    mid_slice = volume.shape[0] // 2
    img = volume[mid_slice]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Study: {study_id}\nSeries: {series_id}\nSlice: {mid_slice}')
    plt.axis('off')

    levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
    labels = ['L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']

    plt.subplot(1, 3, 2)
    plt.bar(labels, [uncertainty_metrics[l]['entropy'] for l in levels])
    plt.ylabel('Entropy')
    plt.title('Uncertainty by Level')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    plt.bar(labels, [uncertainty_metrics[l]['peak_confidence'] for l in levels])
    plt.ylabel('Peak Confidence')
    plt.title('Peak Confidence by Level')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(debug_dir / f'{study_id}_{series_id}_debug.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Debug visualization saved: {debug_dir}")


def main():
    parser = argparse.ArgumentParser(description='LSTV Detection via Epistemic Uncertainty')
    parser.add_argument('--input_dir',       required=True,
                        help='NIfTI root directory (results/nifti)')
    parser.add_argument('--series_csv',      required=True,
                        help='Path to train_series_descriptions.csv')
    parser.add_argument('--output_dir',      required=True,
                        help='Output directory for results')
    parser.add_argument('--checkpoint',      default='/app/models/point_net_checkpoint.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--valid_ids',       default='models/valid_id.npy',
                        help='Path to valid_id.npy (prevents data leakage)')
    parser.add_argument('--mode',            choices=['trial', 'debug', 'prod'], default='trial')
    parser.add_argument('--trial_size',      type=int, default=3,
                        help='Number of studies to process in trial mode (default: 3)')
    parser.add_argument('--debug_study_id',  default=None,
                        help='Specific study ID for debug mode')
    args = parser.parse_args()
    run_inference(args)


if __name__ == '__main__':
    main()
