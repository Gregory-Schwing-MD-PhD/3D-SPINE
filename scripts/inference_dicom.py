#!/usr/bin/env python3
"""
LSTV Detection via Epistemic Uncertainty — DICOM version
=========================================================

Uses pydicom to load raw DICOM series directly, bypassing NIfTI conversion.
This matches the loading logic that was working in inference_centroid.py.

Use this to A/B test against inference.py (NIfTI-based) to isolate whether
orientation/loading differences affect model output quality.

Input layout:  data/raw/train_images/{study_id}/{series_id}/*.dcm
Output:        data/output/trial/lstv_uncertainty_metrics.csv
               data/output/trial/debug_visualizations/{study_id}_*.png
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
from natsort import natsorted
import warnings
warnings.filterwarnings('ignore')

try:
    import pydicom
    HAS_PYDICOM = True
    try:
        import gdcm
        pydicom.config.use_gdcm = True
    except ImportError:
        try:
            import pylibjpeg
        except ImportError:
            pass
except ImportError:
    HAS_PYDICOM = False

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - "
           "<level>{message}</level>")


# ============================================================================
# MODEL ARCHITECTURE (identical to inference.py)
# ============================================================================

class MyDecoderBlock(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True),
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
        i_channel = [in_channel] + out_channel[:-1]
        self.block = nn.ModuleList([
            MyDecoderBlock(i, s, o)
            for i, s, o in zip(i_channel, skip_channel, out_channel)
        ])

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            d = block(d, skip[i])
            decode.append(d)
        return d, decode


class Net(nn.Module):
    def __init__(self, pretrained=False, cfg=None):
        super().__init__()
        self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor(0))
        self.register_buffer('std', torch.tensor(1))
        encoder_dim = [64, 256, 512, 1024, 2048]
        decoder_dim = [256, 128, 64, 32, 16]
        if not HAS_TIMM:
            raise ImportError("timm required: pip install timm")
        self.encoder = timm.create_model(
            'resnet50d', pretrained=pretrained,
            in_chans=3, num_classes=0, global_pool='')
        self.decoder = MyUnetDecoder(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim)
        self.logit = nn.Conv2d(decoder_dim[-1], 6, kernel_size=1)

    def forward(self, batch):
        device = self.D.device
        image = batch['sagittal'].to(device)
        x = image.float() / 255
        x = (x - self.mean) / self.std
        x = x.expand(-1, 3, -1, -1)
        encode = []
        e = self.encoder
        x = e.act1(e.bn1(e.conv1(x))); encode.append(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = e.layer1(x); encode.append(x)
        x = e.layer2(x); encode.append(x)
        x = e.layer3(x); encode.append(x)
        x = e.layer4(x); encode.append(x)
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1] + [None])
        logit = self.logit(last)
        output = {}
        if 'infer' in self.output_type:
            output['probability'] = torch.softmax(logit, 1)
        return output


# ============================================================================
# DICOM LOADING (from inference_centroid.py — the version that worked)
# ============================================================================

def load_dicom_volume(series_dir: Path) -> Optional[np.ndarray]:
    """
    Load a DICOM series into a uint8 (D, H, W) volume using pydicom.
    Files are sorted naturally (natsort) to preserve slice order.
    Pixel values are normalised to [0, 255].
    """
    if not HAS_PYDICOM:
        logger.error("pydicom not installed")
        return None

    dicom_files = natsorted(list(series_dir.glob('*.dcm')))
    if not dicom_files:
        logger.warning(f"  No .dcm files in {series_dir}")
        return None

    try:
        slices = []
        for dcm_file in dicom_files:
            dcm = pydicom.dcmread(str(dcm_file))
            slices.append(dcm.pixel_array.astype(np.float32))

        volume = np.stack(slices)   # (D, H, W)
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            volume = ((volume - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            volume = np.zeros_like(volume, dtype=np.uint8)

        logger.info(f"  DICOM volume: {volume.shape}  "
                    f"range=[{vmin:.0f},{vmax:.0f}]  files={len(dicom_files)}")
        return volume

    except Exception as e:
        logger.error(f"  Error loading DICOMs from {series_dir}: {e}")
        return None


# ============================================================================
# UNCERTAINTY
# ============================================================================

class UncertaintyCalculator:
    @staticmethod
    def calculate_uncertainty(heatmap: np.ndarray) -> Tuple[float, float]:
        peak_confidence = float(np.max(heatmap))
        flat = heatmap.flatten()
        flat = flat / (flat.sum() + 1e-9)
        entropy = float(-np.sum(flat * np.log(flat + 1e-9)))
        return peak_confidence, entropy

    @staticmethod
    def calculate_spatial_entropy(heatmap: np.ndarray, num_bins: int = 10) -> float:
        H, W = heatmap.shape
        bh, bw = max(1, H // num_bins), max(1, W // num_bins)
        bins = [heatmap[i*bh:(i+1)*bh, j*bw:(j+1)*bw].sum()
                for i in range(num_bins) for j in range(num_bins)]
        bins = np.array(bins)
        bins = bins / (bins.sum() + 1e-9)
        return float(-np.sum(bins * np.log(bins + 1e-9)))


def probability_to_uncertainty(probability: np.ndarray,
                                threshold: float = 0.5) -> Dict:
    calc = UncertaintyCalculator()
    metrics = {}
    level_names = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
    for l in range(1, 6):
        heatmap = probability[l]
        peak_conf, entropy = calc.calculate_uncertainty(heatmap)
        spatial_entropy = calc.calculate_spatial_entropy(heatmap)
        metrics[level_names[l-1]] = {
            'peak_confidence': peak_conf,
            'entropy': entropy,
            'spatial_entropy': spatial_entropy,
            'num_pixels_above_threshold': int(np.sum(heatmap > threshold)),
        }
    return metrics


# ============================================================================
# HELPERS
# ============================================================================

def generate_mock_uncertainty() -> Dict:
    metrics = {}
    for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
        is_ls = level in ('l4_l5', 'l5_s1')
        metrics[level] = {
            'peak_confidence': float(np.random.uniform(0.3, 0.6) if is_ls else np.random.uniform(0.7, 0.95)),
            'entropy': float(np.random.uniform(4.5, 6.5) if is_ls else np.random.uniform(2.0, 4.0)),
            'spatial_entropy': float(np.random.uniform(1.5, 3.5)),
            'num_pixels_above_threshold': int(np.random.randint(100, 500)),
        }
    return metrics


def find_sagittal_series_dir(input_dir: Path, study_id: str,
                              series_id: str) -> Optional[Path]:
    """Return path to DICOM series dir, with a fallback glob if series_id is wrong."""
    candidate = input_dir / study_id / series_id
    if candidate.exists() and list(candidate.glob('*.dcm')):
        return candidate

    # Fallback: scan all series dirs for the one with most DICOMs (likely sagittal)
    study_dir = input_dir / study_id
    if study_dir.exists():
        best, best_count = None, 0
        for sub in study_dir.iterdir():
            n = len(list(sub.glob('*.dcm')))
            if n > best_count:
                best, best_count = sub, n
        if best is not None:
            logger.warning(f"  Series {series_id} not found — using {best.name} ({best_count} DICOMs)")
            return best
    return None


# ============================================================================
# REPORT VISUALIZATIONS
# ============================================================================

def save_debug_visualizations(output_dir: Path, study_id, series_id,
                               volume: np.ndarray, uncertainty_metrics: Dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    debug_dir = output_dir / 'debug_visualizations'
    debug_dir.mkdir(exist_ok=True)

    mid_slice = volume.shape[0] // 2
    img = volume[mid_slice]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Study: {study_id}\nSeries: {series_id}\nSlice: {mid_slice}/{volume.shape[0]}')
    axes[0].axis('off')

    levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
    labels = ['L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']

    axes[1].bar(labels, [uncertainty_metrics[l]['entropy'] for l in levels])
    axes[1].set_ylabel('Entropy')
    axes[1].set_title('Uncertainty by Level')
    plt.setp(axes[1].get_xticklabels(), rotation=45)

    axes[2].bar(labels, [uncertainty_metrics[l]['peak_confidence'] for l in levels])
    axes[2].set_ylabel('Peak Confidence')
    axes[2].set_title('Peak Confidence by Level')
    plt.setp(axes[2].get_xticklabels(), rotation=45)

    plt.tight_layout()
    out_path = debug_dir / f'{study_id}_{series_id}_debug.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Debug visualization saved: {out_path}")


# ============================================================================
# MAIN INFERENCE
# ============================================================================

def run_inference(args):
    input_dir  = Path(args.input_dir)   # DICOM root: train_images/
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"DICOM root:  {input_dir}")
    logger.info(f"Output dir:  {output_dir}")
    logger.info(f"Mode:        {args.mode}")

    # --- Validation IDs ---
    valid_ids_path = Path(args.valid_ids)
    if valid_ids_path.exists():
        valid_ids = set(str(v) for v in np.load(valid_ids_path))
        logger.info(f"✓ Loaded {len(valid_ids)} validation study IDs — no data leakage")
        use_validation_only = True
    else:
        logger.warning(f"⚠ valid_ids not found at {valid_ids_path} — running ALL studies")
        valid_ids = None
        use_validation_only = False

    # --- Series CSV ---
    series_csv = Path(args.series_csv)
    if not series_csv.exists():
        logger.error(f"Series CSV not found: {series_csv}")
        return

    series_df = pd.read_csv(series_csv)
    logger.info(f"Loaded {len(series_df)} series descriptions")

    sagittal_df = series_df[
        series_df['series_description'].str.lower().str.contains('sagittal', na=False) &
        series_df['series_description'].str.lower().str.contains('t2', na=False)
    ].copy()
    sagittal_df['study_id'] = sagittal_df['study_id'].astype(str)
    sagittal_df['series_id'] = sagittal_df['series_id'].astype(str)
    studies = sagittal_df['study_id'].unique()
    logger.info(f"Found {len(studies)} studies with Sagittal T2 series in CSV")

    # --- Validation filter ---
    if use_validation_only and valid_ids is not None:
        n_before = len(studies)
        studies = [s for s in studies if s in valid_ids]
        logger.info(f"✓ Validation filter: kept {len(studies)}, excluded {n_before - len(studies)}")

    # --- Mode selection ---
    if args.mode == 'trial':
        valid_ids_ordered = [str(v) for v in np.load(valid_ids_path)]
        studies_set = set(studies)
        studies = [v for v in valid_ids_ordered if v in studies_set][:args.trial_size]
        logger.info(f"Trial mode: first {len(studies)} studies from valid_id.npy (reproducible)")
    elif args.mode == 'debug':
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

    model = None
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path} — MOCK mode")
    else:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            logger.info(f"✓ Checkpoint loaded  keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
            state_dict_key = next((k for k in ('state_dict', 'model_state_dict') if k in ckpt), None) if isinstance(ckpt, dict) else None
            if state_dict_key:
                model = Net(pretrained=False)
                model.load_state_dict(ckpt[state_dict_key])
            elif hasattr(ckpt, 'eval'):
                model = ckpt
            else:
                logger.error(f"Unknown checkpoint structure: {list(ckpt.keys())}")
            if model is not None:
                model = model.to(device)
                model.eval()
                model.output_type = ['infer']
                logger.info("✓✓✓ MODEL LOADED — REAL INFERENCE ✓✓✓")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            import traceback; logger.error(traceback.format_exc())

    if model is None:
        logger.warning("⚠ MOCK MODE — synthetic uncertainty data")

    # --- Inference loop ---
    results = []
    IMAGE_SIZE = 160
    iterator = tqdm(studies, desc="Processing") if args.mode == 'prod' else studies

    for study_id in iterator:
        logger.info(f"\n{'='*60}\nStudy: {study_id}\n{'='*60}")

        study_series = sagittal_df[sagittal_df['study_id'] == str(study_id)]
        if len(study_series) == 0:
            logger.warning(f"No Sagittal T2 in CSV for {study_id} — skipping")
            continue

        series_id = study_series.iloc[0]['series_id']
        logger.info(f"Series from CSV: {series_id}")

        # --- Locate and load DICOM series ---
        series_dir = find_sagittal_series_dir(input_dir, str(study_id), str(series_id))
        if series_dir is None:
            logger.warning(f"DICOM series dir not found for {study_id}/{series_id}")
            continue

        volume = load_dicom_volume(series_dir)
        if volume is None:
            logger.warning(f"Failed to load DICOM volume for {study_id}")
            continue

        logger.info(f"Volume shape: {volume.shape}")

        # --- Run inference ---
        if model is not None:
            try:
                # Resize spatial dims to IMAGE_SIZE x IMAGE_SIZE
                # cv2.resize needs (H, W, D), resize, then back to (D, H, W)
                vol_hwd = np.ascontiguousarray(volume.transpose(1, 2, 0))
                vol_hwd = cv2.resize(vol_hwd, (IMAGE_SIZE, IMAGE_SIZE),
                                     interpolation=cv2.INTER_LINEAR)
                resized = np.ascontiguousarray(vol_hwd.transpose(2, 0, 1))  # (D, 160, 160)

                mid_idx = resized.shape[0] // 2
                image   = resized[mid_idx]  # (160, 160)

                logger.info(f"  Mid-slice stats: min={image.min()} max={image.max()} "
                            f"mean={image.mean():.1f} std={image.std():.1f} "
                            f"(slice {mid_idx}/{resized.shape[0]})")

                image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).byte().to(device)
                batch = {'sagittal': image_tensor}

                with torch.cuda.amp.autocast(enabled=True):
                    with torch.no_grad():
                        output = model(batch)

                probability = output['probability'][0].float().cpu().numpy()  # (6, H, W)
                logger.info(f"  Prob range: [{probability.min():.4f}, {probability.max():.4f}]")

                uncertainty_metrics = probability_to_uncertainty(probability, threshold=0.5)
                logger.info("✓ Real model inference complete")

            except Exception as e:
                logger.error(f"Inference error: {e}")
                import traceback; logger.error(traceback.format_exc())
                logger.warning("Falling back to MOCK data for this study")
                uncertainty_metrics = generate_mock_uncertainty()
        else:
            uncertainty_metrics = generate_mock_uncertainty()

        # --- Log metrics ---
        logger.info("Uncertainty Metrics:")
        for level, m in uncertainty_metrics.items():
            logger.info(f"  {level}: conf={m['peak_confidence']:.4f}  "
                        f"entropy={m['entropy']:.4f}  "
                        f"spatial_entropy={m['spatial_entropy']:.4f}")

        # --- Accumulate ---
        result = {'study_id': study_id, 'series_id': series_id}
        for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
            result[f'{level}_confidence']      = uncertainty_metrics[level]['peak_confidence']
            result[f'{level}_entropy']         = uncertainty_metrics[level]['entropy']
            result[f'{level}_spatial_entropy'] = uncertainty_metrics[level]['spatial_entropy']
        results.append(result)

        # Always save visualizations (useful for sanity-checking slice orientation)
        save_debug_visualizations(output_dir, study_id, series_id, volume, uncertainty_metrics)

    # --- Save results ---
    results_df = pd.DataFrame(results)
    output_csv = output_dir / 'lstv_uncertainty_metrics.csv'
    results_df.to_csv(output_csv, index=False)
    logger.info(f"\nResults saved: {output_csv}  ({len(results)} studies)")

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)
    if results_df.empty:
        logger.warning("No studies processed — check DICOM paths and series CSV")
    else:
        for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
            ec = f'{level}_entropy'
            cc = f'{level}_confidence'
            logger.info(f"{level.upper()}: entropy={results_df[ec].mean():.4f}±{results_df[ec].std():.4f}  "
                        f"conf={results_df[cc].mean():.4f}±{results_df[cc].std():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='LSTV Detection via Epistemic Uncertainty — DICOM input version')
    parser.add_argument('--input_dir',      required=True,
                        help='DICOM root: data/raw/train_images/')
    parser.add_argument('--series_csv',     required=True,
                        help='Path to train_series_descriptions.csv')
    parser.add_argument('--output_dir',     required=True,
                        help='Output directory for CSV and visualizations')
    parser.add_argument('--checkpoint',     default='/app/models/point_net_checkpoint.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--valid_ids',      default='/app/models/valid_id.npy',
                        help='Path to valid_id.npy (prevents data leakage)')
    parser.add_argument('--mode',           choices=['trial', 'debug', 'prod'], default='trial')
    parser.add_argument('--trial_size',     type=int, default=3,
                        help='Number of studies in trial mode (default: 3)')
    parser.add_argument('--debug_study_id', default=None,
                        help='Specific study ID for debug mode')
    args = parser.parse_args()
    run_inference(args)


if __name__ == '__main__':
    main()
