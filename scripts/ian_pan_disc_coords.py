#!/usr/bin/env python3
"""
ian_pan_disc_coords.py
======================
Ian Pan disc-level heatmap → 3D world coordinate extractor.

PURPOSE
-------
Ian Pan's Kaggle model outputs per-disc probability heatmaps at 160×160
resolution from the sagittal T2 mid-slice.  This script:

  1.  Runs inference identically to inference_dicom.py (same model weights,
      same preprocessing — pixel-accurate parity).
  2.  For each of 5 disc levels (L1/L2 → L5/S1):
        a. Extracts argmax and entropy from the 160×160 heatmap.
        b. Maps the argmax back to original DICOM pixel (row, col).
        c. Uses DICOM metadata (ImagePositionPatient, ImageOrientationPatient,
           PixelSpacing) to lift that pixel into LPS world-mm.
        d. Converts LPS → RAS for direct comparison with NIfTI segmentations.
  3.  Optionally loads TSS and SPINEPS disc label NIfTI centroids and computes
      Euclidean distances — a "model agreement score" per disc level.
  4.  Writes per-study JSON + two CSV files (wide and long form).

COORDINATE SYSTEMS
------------------
DICOM patient frame  → LPS  (Left, Posterior, Superior),  mm
NIfTI canonical      → RAS  (Right, Anterior, Superior),  mm
Conversion:           ras = lps * [-1, -1, +1]

The DICOM image-to-world transform is:
  P(i,j) = IPP + j · Δcol · F_row + i · Δrow · F_col
where
  IPP      = ImagePositionPatient       (origin of pixel [0,0])
  IOP      = ImageOrientationPatient    (6 direction cosines)
  F_row    = IOP[0:3]   direction cosines of the *row* axis
             (direction you move when column index increases)
  F_col    = IOP[3:6]   direction cosines of the *column* axis
             (direction you move when row index increases)
  Δcol     = PixelSpacing[1]   mm per column step
  Δrow     = PixelSpacing[0]   mm per row step

Note on sagittal T2 slice sorting:
  Slices are sorted by the projection of ImagePositionPatient onto the
  slice-normal vector (cross product of F_row × F_col).  This is more
  reliable than filename ordering for multi-echo / vendor-variable naming.

OUTPUT
------
results/ian_pan_disc_coords/
  ian_pan_disc_coords.json         — full per-study records
  ian_pan_disc_coords.csv          — wide CSV (one row per study)
  ian_pan_disc_per_level.csv       — long CSV (one row per study × level)
  progress_coords.json             — resume checkpoint

INTEGRATION
-----------
In 04_detect_lstv.py:
  from ian_pan_loader import load_ian_pan_disc_coords
  ip = load_ian_pan_disc_coords('results/ian_pan_disc_coords/ian_pan_disc_coords.json')
  coords = ip.get(study_id)   # dict keyed by disc level

The model_agreement distances can be passed to vertebral_alignment.py to
break H0/H1 ties: if Ian Pan's L5-S1 peak is significantly closer to
TSS label 100 than to the SPINEPS-implied L5-S1 position, prefer TSS.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from natsort import natsorted
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import pydicom
    HAS_PYDICOM = True
    try:
        import gdcm
        pydicom.config.use_gdcm = True
    except ImportError:
        try:
            import pylibjpeg  # noqa: F401
        except ImportError:
            pass
except ImportError:
    HAS_PYDICOM = False
    logger.error("pydicom not installed — cannot read DICOM files")

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    logger.error("timm not installed")

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    logger.warning("nibabel not installed — NIfTI comparison disabled")

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
           "<cyan>{function}</cyan> - <level>{message}</level>",
)

# ── constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 160
DISC_NAMES   = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
DISC_INDICES = [1, 2, 3, 4, 5]   # indices in the 6-channel output (0 = background)

# TSS disc labels (from lstv_engine constants)
# H0 hypothesis: TSS labels are correct — Ian Pan disc k should match TSS disc k
TSS_DISC_MAP_H0 = {
    "l1_l2": 91,
    "l2_l3": 92,
    "l3_l4": 93,
    "l4_l5": 95,
    "l5_s1": 100,
}

# H1 hypothesis: lumbarization — SPINEPS found an extra caudal mobile segment,
# so Ian Pan's disc k corresponds to TSS disc k-1 (one level cranial in TSS).
# Concretely: if S1 is lumbarized (becomes L6), TSS's "L5" is really L6,
# so Ian Pan's L5-S1 heatmap peak should land near TSS label 95 (L4-L5),
# Ian Pan's L4-L5 should land near TSS label 93 (L3-L4), etc.
# Levels where the shift goes above L1 have no TSS disc match → None.
TSS_DISC_MAP_H1 = {
    "l1_l2": None,   # shifted cranial past available TSS labels
    "l2_l3": 91,     # IP L2-L3  ↔  TSS L1-L2
    "l3_l4": 92,     # IP L3-L4  ↔  TSS L2-L3
    "l4_l5": 93,     # IP L4-L5  ↔  TSS L3-L4
    "l5_s1": 95,     # IP L5-S1  ↔  TSS L4-L5  (the key discriminating level)
}

# For legacy compatibility — default to H0
TSS_DISC_MAP = TSS_DISC_MAP_H0

# Ian Pan confidence thresholds (tuned on Kaggle val set)
HIGH_CONF_THRESH   = 0.70
MEDIUM_CONF_THRESH = 0.40

# Minimum peak_prob to include a disc level in the sequence vote
SEQ_VOTE_MIN_PROB  = 0.30

# Distance margin (mm) required for sequence vote to declare a winner
SEQ_VOTE_MIN_MARGIN_MM = 8.0


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  (identical to inference_dicom.py — do not modify)
# ══════════════════════════════════════════════════════════════════════════════

class MyDecoderBlock(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.attention2(x)


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
    def __init__(self, pretrained=False):
        super().__init__()
        self.output_type = ["infer", "loss"]
        self.register_buffer("D",    torch.tensor(0))
        self.register_buffer("mean", torch.tensor(0))
        self.register_buffer("std",  torch.tensor(1))
        encoder_dim = [64, 256, 512, 1024, 2048]
        decoder_dim = [256, 128, 64, 32, 16]
        if not HAS_TIMM:
            raise ImportError("timm required")
        self.encoder = timm.create_model(
            "resnet50d", pretrained=pretrained, in_chans=3,
            num_classes=0, global_pool="")
        self.decoder = MyUnetDecoder(
            in_channel  = encoder_dim[-1],
            skip_channel= encoder_dim[:-1][::-1] + [0],
            out_channel = decoder_dim)
        self.logit = nn.Conv2d(decoder_dim[-1], 6, kernel_size=1)

    def forward(self, batch):
        device = self.D.device
        image  = batch["sagittal"].to(device)
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
        last, _ = self.decoder(feature=encode[-1], skip=encode[:-1][::-1] + [None])
        logit   = self.logit(last)
        return {"probability": torch.softmax(logit, 1)}


# ══════════════════════════════════════════════════════════════════════════════
# DICOM LOADING + SORTING
# ══════════════════════════════════════════════════════════════════════════════

def _slice_position(dcm) -> float:
    """
    Project ImagePositionPatient onto the slice-normal (F_row × F_col).
    More reliable than filename ordering across vendors.
    Falls back to z-component of IPP if IOP is unavailable.
    """
    try:
        IPP = np.array([float(v) for v in dcm.ImagePositionPatient])
        IOP = np.array([float(v) for v in dcm.ImageOrientationPatient])
        normal = np.cross(IOP[0:3], IOP[3:6])
        return float(np.dot(IPP, normal))
    except Exception:
        try:
            return float(dcm.ImagePositionPatient[2])
        except Exception:
            return 0.0


def load_dicom_series(series_dir: Path) -> Optional[Tuple[np.ndarray, List]]:
    """
    Load a DICOM series sorted by physical slice position.

    Returns
    -------
    (volume, dcm_list)
        volume   : uint8 ndarray (D, H, W), normalised to [0, 255]
        dcm_list : list of pydicom.Dataset objects in slice order
    """
    if not HAS_PYDICOM:
        return None

    dcm_files = list(series_dir.glob("*.dcm"))
    if not dcm_files:
        logger.warning(f"  No .dcm files in {series_dir}")
        return None

    datasets = []
    for f in dcm_files:
        try:
            datasets.append(pydicom.dcmread(str(f)))
        except Exception as e:
            logger.warning(f"  Skipping {f.name}: {e}")

    if not datasets:
        return None

    # Sort by physical slice position (robust across vendors)
    datasets.sort(key=_slice_position)

    try:
        slices = [ds.pixel_array.astype(np.float32) for ds in datasets]
        volume = np.stack(slices)          # (D, H, W)
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            volume = ((volume - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            volume = np.zeros_like(volume, dtype=np.uint8)
        logger.info(f"  Loaded {len(datasets)} slices  shape={volume.shape}"
                    f"  range=[{vmin:.0f},{vmax:.0f}]")
        return volume, datasets
    except Exception as e:
        logger.error(f"  DICOM load error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# COORDINATE MAPPING
# ══════════════════════════════════════════════════════════════════════════════

def heatmap160_to_original_pixel(
    row_160: float, col_160: float,
    orig_h: int, orig_w: int,
) -> Tuple[float, float]:
    """
    Map a (row, col) position in the 160×160 resized heatmap back to the
    original DICOM slice pixel coordinates.

    The preprocessing chain is:
      volume (D,H,W) → transpose (H,W,D) → cv2.resize → (160,160,D)
                                                       ↑
                                     cv2.resize((160,160)) resizes the first
                                     two spatial dims (H, W) of the (H,W,D)
                                     array using INTER_LINEAR.
    So heatmap row ↔ original H axis, heatmap col ↔ original W axis.
    """
    row_orig = row_160 * orig_h / IMAGE_SIZE
    col_orig = col_160 * orig_w / IMAGE_SIZE
    return row_orig, col_orig


def dicom_pixel_to_lps(
    row: float, col: float,
    dcm,
) -> Optional[np.ndarray]:
    """
    Convert DICOM pixel (row i, col j) to LPS patient world coordinates (mm).

    DICOM image-to-patient transform (PS 3.3 C.7.6.2.1.1):
      P(i,j) = IPP + j · ΔC · F_row + i · ΔR · F_col

    where
      IPP   = ImagePositionPatient   [x, y, z] mm of pixel (0,0)
      IOP   = ImageOrientationPatient  6 direction cosines
      F_row = IOP[0:3]   unit vector along increasing column index
      F_col = IOP[3:6]   unit vector along increasing row index
      ΔC    = PixelSpacing[1]   mm per column step
      ΔR    = PixelSpacing[0]   mm per row step
    """
    try:
        IPP   = np.array([float(v) for v in dcm.ImagePositionPatient])
        IOP   = np.array([float(v) for v in dcm.ImageOrientationPatient])
        PS    = [float(v) for v in dcm.PixelSpacing]
        F_row = IOP[0:3]   # direction: increasing column index
        F_col = IOP[3:6]   # direction: increasing row index
        delta_col = PS[1]  # mm per column step
        delta_row = PS[0]  # mm per row step
        world_lps = IPP + col * delta_col * F_row + row * delta_row * F_col
        return world_lps
    except Exception as e:
        logger.warning(f"  pixel_to_lps failed: {e}")
        return None


def lps_to_ras(lps: np.ndarray) -> np.ndarray:
    """LPS → RAS coordinate system (flip x and y axes)."""
    return np.array([-lps[0], -lps[1], lps[2]])


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_disc_heatmap(
    probability: np.ndarray,   # (6, 160, 160)
    mid_dcm,                   # pydicom.Dataset for the mid-slice
    orig_h: int,
    orig_w: int,
) -> Dict:
    """
    Extract per-disc-level metrics from Ian Pan probability heatmaps.

    Returns a dict keyed by disc level name with:
      peak_prob          : float   max probability in heatmap
      entropy            : float   normalised entropy of heatmap distribution
      spatial_entropy    : float   spatial entropy (10×10 grid)
      heatmap_peak_rc160 : [r, c]  argmax in 160×160 space
      pixel_rc_orig      : [r, c]  argmax mapped to original DICOM pixel space
      world_lps_mm       : [x,y,z] LPS patient coordinates (mm)
      world_ras_mm       : [x,y,z] RAS coordinates (mm)  — NIfTI-compatible
      confidence_class   : str     'high' / 'medium' / 'low'
    """
    results = {}

    for disc_name, ch_idx in zip(DISC_NAMES, DISC_INDICES):
        heatmap = probability[ch_idx]   # (160, 160)

        # ── peak probability ──────────────────────────────────────────────────
        peak_prob = float(np.max(heatmap))

        # ── normalised entropy ────────────────────────────────────────────────
        flat  = heatmap.flatten()
        flat  = flat / (flat.sum() + 1e-9)
        entropy = float(-np.sum(flat * np.log(flat + 1e-9)))

        # ── spatial entropy (10×10 grid) ──────────────────────────────────────
        H, W  = heatmap.shape
        bh, bw = max(1, H // 10), max(1, W // 10)
        bins  = np.array([
            heatmap[i*bh:(i+1)*bh, j*bw:(j+1)*bw].sum()
            for i in range(10) for j in range(10)
        ])
        bins  = bins / (bins.sum() + 1e-9)
        spatial_entropy = float(-np.sum(bins * np.log(bins + 1e-9)))

        # ── argmax in 160×160 space ───────────────────────────────────────────
        flat_idx     = int(np.argmax(heatmap))
        r160, c160   = np.unravel_index(flat_idx, heatmap.shape)
        r160, c160   = int(r160), int(c160)

        # ── map back to original DICOM pixel ──────────────────────────────────
        r_orig, c_orig = heatmap160_to_original_pixel(r160, c160, orig_h, orig_w)

        # ── lift to 3D world coordinates ──────────────────────────────────────
        lps = dicom_pixel_to_lps(r_orig, c_orig, mid_dcm)
        ras = lps_to_ras(lps) if lps is not None else None

        # ── confidence class ──────────────────────────────────────────────────
        if peak_prob >= HIGH_CONF_THRESH:
            conf_cls = "high"
        elif peak_prob >= MEDIUM_CONF_THRESH:
            conf_cls = "medium"
        else:
            conf_cls = "low"

        results[disc_name] = {
            "peak_prob":          round(peak_prob, 5),
            "entropy":            round(entropy,   5),
            "spatial_entropy":    round(spatial_entropy, 5),
            "heatmap_peak_rc160": [r160, c160],
            "pixel_rc_orig":      [round(r_orig, 2), round(c_orig, 2)],
            "world_lps_mm":       lps.tolist() if lps is not None else None,
            "world_ras_mm":       ras.tolist() if ras is not None else None,
            "confidence_class":   conf_cls,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# NIFTI COMPARISON  (optional — requires nibabel + segmentations)
# ══════════════════════════════════════════════════════════════════════════════

def _nifti_label_centroid_ras(nii_path: Path, label: int) -> Optional[np.ndarray]:
    """
    Return centroid of a label in a NIfTI segmentation in RAS mm.
    Returns None if the label is absent or the file doesn't exist.
    """
    if not HAS_NIBABEL or not nii_path.exists():
        return None
    try:
        nii  = nib.load(str(nii_path))
        nii  = nib.as_closest_canonical(nii)
        data = nii.get_fdata()
        while data.ndim > 3:
            data = data[..., 0]
        mask = (data.astype(int) == label)
        if not mask.any():
            return None
        coords = np.array(np.where(mask), dtype=float)        # (3, N)
        vox_centroid = coords.mean(axis=1)                     # (3,) voxel
        ras_centroid = nib.affines.apply_affine(nii.affine, vox_centroid)
        return ras_centroid
    except Exception as e:
        logger.warning(f"  centroid_ras({nii_path.name}, label={label}): {e}")
        return None


def compute_model_agreement(
    disc_coords:      Dict,
    spineps_vert_path: Path,   # seg-vert_msk.nii.gz  (reserved for future use)
    tss_sag_path:     Path,    # sagittal_labeled.nii.gz
) -> Dict:
    """
    For every disc level, compute Ian Pan ↔ TSS centroid distances under
    BOTH hypotheses, then compute a sequence-level vote.

    Per-level fields
    ----------------
    ip_coords_valid    : bool    False if Ian Pan returned no RAS coord
    peak_prob          : float   Ian Pan peak probability (copied for convenience)
    dist_to_tss_h0_mm  : float | None  dist(IP_k, TSS_disc_k)       — H0
    dist_to_tss_h1_mm  : float | None  dist(IP_k, TSS_disc_{k-1})   — H1 shifted
    tss_centroid_h0_ras: list | None   TSS centroid under H0
    tss_centroid_h1_ras: list | None   TSS centroid under H1

    Top-level summary fields (under key '_sequence_vote')
    -----------------------------------------------------
    n_levels_h0        : int    levels with valid H0 distance
    n_levels_h1        : int    levels with valid H1 distance
    mean_dist_h0_mm    : float  mean H0 distance (only confident IP levels)
    mean_dist_h1_mm    : float  mean H1 distance (only confident IP levels)
    margin_mm          : float  mean_dist_h0 - mean_dist_h1
                                 > 0  → H0 better (TSS labels correct)
                                 < 0  → H1 better (TSS shifted, lumbarization)
    sequence_vote      : str    'H0' | 'H1' | 'neutral'
    vote_confidence    : str    'high' | 'moderate' | 'low' | 'insufficient'
    n_levels_voted     : int    number of disc levels that contributed

    H0: Ian Pan disc k aligns with TSS disc k  (normal anatomy / sacralization)
    H1: Ian Pan disc k aligns with TSS disc k-1 (lumbarization — extra mobile
        segment means TSS's labels are one level too cranial for SPINEPS).
    """
    # ── Pre-load all needed TSS centroids in one pass ──────────────────────────
    all_tss_labels = set()
    for lbl in TSS_DISC_MAP_H0.values():
        if lbl: all_tss_labels.add(lbl)
    for lbl in TSS_DISC_MAP_H1.values():
        if lbl: all_tss_labels.add(lbl)

    tss_centroids: Dict[int, Optional[np.ndarray]] = {}
    for lbl in all_tss_labels:
        tss_centroids[lbl] = _nifti_label_centroid_ras(tss_sag_path, lbl)

    # ── Per-level distances ────────────────────────────────────────────────────
    agreement: Dict = {}
    h0_dists: List[float] = []
    h1_dists: List[float] = []

    for disc_name in DISC_NAMES:
        entry  = disc_coords.get(disc_name, {})
        ip_ras = entry.get("world_ras_mm")
        prob   = entry.get("peak_prob", 0.0)

        if ip_ras is None:
            agreement[disc_name] = {"ip_coords_valid": False, "peak_prob": prob}
            continue

        ip_arr = np.array(ip_ras)
        row: Dict = {"ip_coords_valid": True, "peak_prob": round(prob, 5)}

        # H0 distance: Ian Pan disc k  ↔  TSS disc k
        lbl_h0 = TSS_DISC_MAP_H0.get(disc_name)
        ctr_h0 = tss_centroids.get(lbl_h0) if lbl_h0 else None
        if ctr_h0 is not None:
            d_h0 = float(np.linalg.norm(ip_arr - ctr_h0))
            row["dist_to_tss_h0_mm"]   = round(d_h0, 2)
            row["tss_centroid_h0_ras"] = ctr_h0.tolist()
        else:
            d_h0 = None
            row["dist_to_tss_h0_mm"]   = None
            row["tss_centroid_h0_ras"] = None

        # H1 distance: Ian Pan disc k  ↔  TSS disc k-1  (shifted hypothesis)
        lbl_h1 = TSS_DISC_MAP_H1.get(disc_name)
        ctr_h1 = tss_centroids.get(lbl_h1) if lbl_h1 else None
        if ctr_h1 is not None:
            d_h1 = float(np.linalg.norm(ip_arr - ctr_h1))
            row["dist_to_tss_h1_mm"]   = round(d_h1, 2)
            row["tss_centroid_h1_ras"] = ctr_h1.tolist()
        else:
            d_h1 = None
            row["dist_to_tss_h1_mm"]   = None
            row["tss_centroid_h1_ras"] = None

        # Only include in sequence vote if Ian Pan is sufficiently confident
        if prob >= SEQ_VOTE_MIN_PROB:
            if d_h0 is not None: h0_dists.append(d_h0)
            if d_h1 is not None: h1_dists.append(d_h1)

        agreement[disc_name] = row

    # ── Sequence-level vote ────────────────────────────────────────────────────
    n_voted = min(len(h0_dists), len(h1_dists))

    if n_voted < 2:
        seq_vote = "neutral"
        vote_conf = "insufficient"
        margin = 0.0
        mean_h0 = float(np.mean(h0_dists)) if h0_dists else None
        mean_h1 = float(np.mean(h1_dists)) if h1_dists else None
    else:
        # Use only levels that contributed to BOTH hypotheses for fair comparison
        # (pair them by position in DISC_NAMES order)
        paired_h0, paired_h1 = [], []
        for disc_name in DISC_NAMES:
            row = agreement.get(disc_name, {})
            d0  = row.get("dist_to_tss_h0_mm")
            d1  = row.get("dist_to_tss_h1_mm")
            p   = row.get("peak_prob", 0.0)
            if d0 is not None and d1 is not None and p >= SEQ_VOTE_MIN_PROB:
                paired_h0.append(d0)
                paired_h1.append(d1)

        n_voted = len(paired_h0)
        if n_voted < 2:
            seq_vote = "neutral"
            vote_conf = "insufficient"
            margin = 0.0
            mean_h0 = float(np.mean(paired_h0)) if paired_h0 else None
            mean_h1 = float(np.mean(paired_h1)) if paired_h1 else None
        else:
            mean_h0 = float(np.mean(paired_h0))
            mean_h1 = float(np.mean(paired_h1))
            margin  = mean_h0 - mean_h1   # positive = H0 better, negative = H1 better

            if abs(margin) < SEQ_VOTE_MIN_MARGIN_MM:
                seq_vote  = "neutral"
                vote_conf = "low"
            elif margin > 0:
                # H0 has smaller mean distance → TSS labels match Ian Pan
                seq_vote  = "H0"
                vote_conf = ("high"     if abs(margin) > 20.0 else
                             "moderate" if abs(margin) > SEQ_VOTE_MIN_MARGIN_MM else "low")
            else:
                # H1 has smaller mean distance → TSS shifted, SPINEPS has extra level
                seq_vote  = "H1"
                vote_conf = ("high"     if abs(margin) > 20.0 else
                             "moderate" if abs(margin) > SEQ_VOTE_MIN_MARGIN_MM else "low")

    agreement["_sequence_vote"] = {
        "n_levels_voted":   n_voted,
        "n_levels_h0":      len(h0_dists),
        "n_levels_h1":      len(h1_dists),
        "mean_dist_h0_mm":  round(mean_h0, 2) if mean_h0 is not None else None,
        "mean_dist_h1_mm":  round(mean_h1, 2) if mean_h1 is not None else None,
        "margin_mm":        round(margin, 2),
        "sequence_vote":    seq_vote,
        "vote_confidence":  vote_conf,
    }

    logger.info(
        f"  Seq vote: H0={mean_h0:.1f}mm  H1={mean_h1:.1f}mm  "
        f"margin={margin:+.1f}mm  → {seq_vote} [{vote_conf}]  "
        f"(n={n_voted} disc levels)"
        if (mean_h0 is not None and mean_h1 is not None)
        else f"  Seq vote: insufficient data (n_paired={n_voted})"
    )

    return agreement


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS / RESUME
# ══════════════════════════════════════════════════════════════════════════════

def load_progress(output_dir: Path) -> dict:
    pf = output_dir / "progress_coords.json"
    if pf.exists():
        try:
            with open(pf) as f:
                p = json.load(f)
            logger.info(f"Resume: {len(p.get('success',[]))} done, "
                        f"{len(p.get('failed',[]))} failed")
            return p
        except Exception:
            pass
    return {"success": [], "failed": []}


def save_progress(output_dir: Path, progress: dict):
    pf  = output_dir / "progress_coords.json"
    tmp = pf.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    tmp.replace(pf)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT WRITERS
# ══════════════════════════════════════════════════════════════════════════════

def append_result_json(output_dir: Path, record: dict):
    """Append record to the growing JSON list (load → append → write)."""
    json_path = output_dir / "ian_pan_disc_coords.json"
    records: List[dict] = []
    if json_path.exists():
        try:
            with open(json_path) as f:
                records = json.load(f)
        except Exception:
            records = []
    records.append(record)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)


def _flat_row(record: dict) -> dict:
    """Flatten a per-study record to a single CSV row (wide format)."""
    row: dict = {
        "study_id":  record["study_id"],
        "series_id": record.get("series_id", ""),
        "n_slices":  record.get("n_slices", ""),
        "mid_slice_idx":     record.get("mid_slice_idx", ""),
        "original_h":        record.get("original_shape", [None, None])[0],
        "original_w":        record.get("original_shape", [None, None])[1],
        "mid_slice_lps_x":   record.get("mid_slice_lps_origin", [None]*3)[0],
        "mid_slice_lps_y":   record.get("mid_slice_lps_origin", [None]*3)[1],
        "mid_slice_lps_z":   record.get("mid_slice_lps_origin", [None]*3)[2],
        "pixel_spacing_row": record.get("pixel_spacing_mm", [None, None])[0],
        "pixel_spacing_col": record.get("pixel_spacing_mm", [None, None])[1],
    }
    # Per disc-level fields
    for disc_name in DISC_NAMES:
        disc   = record.get("disc_levels", {}).get(disc_name, {})
        agr    = record.get("model_agreement", {}).get(disc_name, {})
        prefix = disc_name
        row[f"{prefix}_peak_prob"]          = disc.get("peak_prob")
        row[f"{prefix}_entropy"]            = disc.get("entropy")
        row[f"{prefix}_spatial_entropy"]    = disc.get("spatial_entropy")
        row[f"{prefix}_confidence_class"]   = disc.get("confidence_class")
        peak_rc = disc.get("heatmap_peak_rc160", [None, None])
        row[f"{prefix}_peak_row160"]        = peak_rc[0]
        row[f"{prefix}_peak_col160"]        = peak_rc[1]
        ras = disc.get("world_ras_mm", [None]*3)
        row[f"{prefix}_ras_x"]              = ras[0] if ras else None
        row[f"{prefix}_ras_y"]              = ras[1] if ras else None
        row[f"{prefix}_ras_z"]              = ras[2] if ras else None
        # H0 and H1 distances
        row[f"{prefix}_dist_h0_mm"]         = agr.get("dist_to_tss_h0_mm")
        row[f"{prefix}_dist_h1_mm"]         = agr.get("dist_to_tss_h1_mm")
        # Convenience: which hypothesis is closer for this level?
        d0 = agr.get("dist_to_tss_h0_mm"); d1 = agr.get("dist_to_tss_h1_mm")
        if d0 is not None and d1 is not None:
            row[f"{prefix}_closer_hyp"] = "H0" if d0 <= d1 else "H1"
            row[f"{prefix}_hyp_margin_mm"] = round(abs(d0 - d1), 2)
        else:
            row[f"{prefix}_closer_hyp"]    = None
            row[f"{prefix}_hyp_margin_mm"] = None

    # Sequence vote summary
    sv = record.get("model_agreement", {}).get("_sequence_vote", {})
    row["seq_vote"]          = sv.get("sequence_vote")
    row["seq_vote_conf"]     = sv.get("vote_confidence")
    row["seq_margin_mm"]     = sv.get("margin_mm")
    row["seq_mean_h0_mm"]    = sv.get("mean_dist_h0_mm")
    row["seq_mean_h1_mm"]    = sv.get("mean_dist_h1_mm")
    row["seq_n_levels_voted"]= sv.get("n_levels_voted")
    return row


def write_csvs(output_dir: Path):
    """(Re-)write both CSVs from the JSON master file."""
    json_path = output_dir / "ian_pan_disc_coords.json"
    if not json_path.exists():
        return
    with open(json_path) as f:
        records = json.load(f)
    if not records:
        return

    # Wide CSV (one row per study)
    wide_rows = [_flat_row(r) for r in records]
    pd.DataFrame(wide_rows).to_csv(
        output_dir / "ian_pan_disc_coords.csv", index=False)

    # Long CSV (one row per study × disc level, includes H0/H1 distances)
    long_rows = []
    for r in records:
        sv  = r.get("model_agreement", {}).get("_sequence_vote", {})
        for disc_name in DISC_NAMES:
            disc = r.get("disc_levels", {}).get(disc_name, {})
            agr  = r.get("model_agreement", {}).get(disc_name, {})
            ras  = disc.get("world_ras_mm", [None]*3)
            d0   = agr.get("dist_to_tss_h0_mm")
            d1   = agr.get("dist_to_tss_h1_mm")
            long_rows.append({
                "study_id":          r["study_id"],
                "disc_level":        disc_name,
                "peak_prob":         disc.get("peak_prob"),
                "entropy":           disc.get("entropy"),
                "spatial_entropy":   disc.get("spatial_entropy"),
                "confidence_class":  disc.get("confidence_class"),
                "ras_x":             ras[0] if ras else None,
                "ras_y":             ras[1] if ras else None,
                "ras_z":             ras[2] if ras else None,
                # H0: Ian Pan disc k  ↔  TSS disc k
                "dist_h0_mm":        d0,
                # H1: Ian Pan disc k  ↔  TSS disc k-1  (lumbarization shift)
                "dist_h1_mm":        d1,
                "closer_hyp":        ("H0" if (d0 is not None and d1 is not None and d0 <= d1)
                                      else "H1" if (d0 is not None and d1 is not None)
                                      else None),
                "hyp_margin_mm":     (round(abs(d0 - d1), 2)
                                      if d0 is not None and d1 is not None else None),
                # Study-level sequence vote (repeated per level for easy filtering)
                "study_seq_vote":    sv.get("sequence_vote"),
                "study_seq_conf":    sv.get("vote_confidence"),
                "study_seq_margin":  sv.get("margin_mm"),
            })

    pd.DataFrame(long_rows).to_csv(
        output_dir / "ian_pan_disc_per_level.csv", index=False)

    logger.info(f"  CSV: {len(wide_rows)} studies  "
                f"long: {len(long_rows)} rows ({len(DISC_NAMES)} levels each)")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def find_sagittal_series_dir(
    input_dir: Path, study_id: str, series_id: str
) -> Optional[Path]:
    candidate = input_dir / study_id / series_id
    if candidate.exists() and list(candidate.glob("*.dcm")):
        return candidate
    study_dir = input_dir / study_id
    if study_dir.exists():
        best, best_n = None, 0
        for sub in study_dir.iterdir():
            n = len(list(sub.glob("*.dcm")))
            if n > best_n:
                best, best_n = sub, n
        if best:
            logger.warning(f"  Series {series_id} not found — using {best.name}")
            return best
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(args):
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Segmentation dirs for NIfTI comparison ────────────────────────────────
    spineps_dir    = Path(args.spineps_dir)    if args.spineps_dir    else None
    totalspine_dir = Path(args.totalspine_dir) if args.totalspine_dir else None

    # ── Validation IDs ────────────────────────────────────────────────────────
    valid_ids: Optional[set] = None
    valid_ids_ordered: Optional[list] = None
    vp = Path(args.valid_ids)
    if vp.exists():
        arr = np.load(vp)
        valid_ids         = set(str(v) for v in arr)
        valid_ids_ordered = [str(v) for v in arr]
        logger.info(f"Validation IDs: {len(valid_ids)}")
    else:
        logger.warning(f"valid_ids not found at {vp} — running ALL studies")

    # ── Series CSV ────────────────────────────────────────────────────────────
    series_csv = Path(args.series_csv)
    if not series_csv.exists():
        logger.error(f"Series CSV not found: {series_csv}"); return
    series_df   = pd.read_csv(series_csv)
    sagittal_df = series_df[
        series_df["series_description"].str.lower().str.contains("sagittal", na=False) &
        series_df["series_description"].str.lower().str.contains("t2",       na=False)
    ].copy()
    sagittal_df["study_id"]  = sagittal_df["study_id"].astype(str)
    sagittal_df["series_id"] = sagittal_df["series_id"].astype(str)
    studies = list(sagittal_df["study_id"].unique())
    logger.info(f"Sagittal T2 studies: {len(studies)}")

    if valid_ids:
        studies = [s for s in studies if s in valid_ids]
        logger.info(f"After validation filter: {len(studies)}")

    if args.mode == "trial":
        if valid_ids_ordered:
            s_set   = set(studies)
            studies = [v for v in valid_ids_ordered if v in s_set][: args.trial_size]
        else:
            studies = studies[: args.trial_size]
        logger.info(f"Trial mode: {len(studies)} studies")
    elif args.mode == "debug":
        studies = [args.debug_study_id or studies[0]]
    else:
        logger.info(f"Production mode: {len(studies)} studies")

    # ── Resume ────────────────────────────────────────────────────────────────
    progress = load_progress(output_dir)
    if not args.retry_failed:
        done = set(progress.get("success", []))
        n_before = len(studies)
        studies  = [s for s in studies if s not in done]
        if skipped := n_before - len(studies):
            logger.info(f"Resume: skipping {skipped} already-done studies")

    if not studies:
        logger.info("All studies already processed."); return

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}"); return

    model = None
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        sd_key = next(
            (k for k in ("state_dict", "model_state_dict")
             if isinstance(ckpt, dict) and k in ckpt),
            None,
        )
        if sd_key:
            model = Net(); model.load_state_dict(ckpt[sd_key])
        elif hasattr(ckpt, "eval"):
            model = ckpt
        else:
            logger.error(f"Unknown checkpoint structure: {list(ckpt.keys())}"); return
        model = model.to(device).eval()
        model.output_type = ["infer"]
        logger.info("Model loaded ✓")
    except Exception as exc:
        logger.error(f"Checkpoint load failed: {exc}"); return

    # ── Inference loop ────────────────────────────────────────────────────────
    iterator  = tqdm(studies, desc="Processing") if args.mode == "prod" else studies
    success_n = failed_n = 0

    for study_id in iterator:
        logger.info(f"\n{'='*60}\nStudy: {study_id}")
        try:
            study_series = sagittal_df[sagittal_df["study_id"] == study_id]
            if study_series.empty:
                logger.warning(f"  No Sagittal T2 in CSV for {study_id}")
                progress.setdefault("failed", []).append(study_id)
                save_progress(output_dir, progress)
                failed_n += 1
                continue

            series_id  = study_series.iloc[0]["series_id"]
            series_dir = find_sagittal_series_dir(input_dir, study_id, series_id)
            if series_dir is None:
                logger.warning(f"  DICOM dir not found for {study_id}/{series_id}")
                progress.setdefault("failed", []).append(study_id)
                save_progress(output_dir, progress)
                failed_n += 1
                continue

            loaded = load_dicom_series(series_dir)
            if loaded is None:
                progress.setdefault("failed", []).append(study_id)
                save_progress(output_dir, progress)
                failed_n += 1
                continue

            volume, dcm_list = loaded
            D, orig_h, orig_w = volume.shape
            mid_idx  = D // 2
            mid_dcm  = dcm_list[mid_idx]

            # Record mid-slice origin for reference (LPS mm)
            try:
                mid_lps_origin = [float(v) for v in mid_dcm.ImagePositionPatient]
            except Exception:
                mid_lps_origin = None

            try:
                pixel_spacing = [float(v) for v in mid_dcm.PixelSpacing]
            except Exception:
                pixel_spacing = None

            # ── Preprocess (exact match to inference_dicom.py) ─────────────
            vol_hwd = np.ascontiguousarray(volume.transpose(1, 2, 0))       # (H,W,D)
            vol_hwd = cv2.resize(vol_hwd, (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_LINEAR)             # (160,160,D)
            resized  = np.ascontiguousarray(vol_hwd.transpose(2, 0, 1))     # (D,160,160)
            image    = resized[mid_idx]                                       # (160,160)

            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).byte().to(device)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                with torch.no_grad():
                    output = model({"sagittal": image_tensor})

            probability = output["probability"][0].float().cpu().numpy()     # (6,160,160)

            # ── Extract disc coordinates ───────────────────────────────────
            disc_coords = analyse_disc_heatmap(
                probability, mid_dcm, orig_h, orig_w)

            # Log summary
            for disc_name in DISC_NAMES:
                d = disc_coords[disc_name]
                logger.info(
                    f"  {disc_name}: prob={d['peak_prob']:.3f} "
                    f"[{d['confidence_class']:6s}]  "
                    f"H={d['entropy']:.3f}  "
                    f"RAS={[round(v,1) for v in (d['world_ras_mm'] or [])]}"
                )

            # ── NIfTI model agreement (optional) ──────────────────────────
            model_agreement: dict = {}
            if spineps_dir and totalspine_dir:
                tss_sag_path = (totalspine_dir / study_id / "sagittal"
                                / f"{study_id}_sagittal_labeled.nii.gz")
                spineps_vert = (spineps_dir / "segmentations" / study_id
                                / f"{study_id}_seg-vert_msk.nii.gz")
                if tss_sag_path.exists():
                    model_agreement = compute_model_agreement(
                        disc_coords, spineps_vert, tss_sag_path)
                    # Log disagreement summary
                    for disc_name, agr in model_agreement.items():
                        d_tss = agr.get("dist_to_tss_mm")
                        if d_tss is not None:
                            flag = "⚠" if d_tss > 15.0 else "✓"
                            logger.info(f"    {disc_name} → TSS: {d_tss:.1f}mm {flag}")
                else:
                    logger.warning(f"  TSS sagittal not found for {study_id}")

            # ── Build record ───────────────────────────────────────────────
            record = {
                "study_id":             study_id,
                "series_id":            series_id,
                "n_slices":             D,
                "mid_slice_idx":        mid_idx,
                "original_shape":       [orig_h, orig_w],
                "mid_slice_lps_origin": mid_lps_origin,
                "pixel_spacing_mm":     pixel_spacing,
                "disc_levels":          disc_coords,
                "model_agreement":      model_agreement,
            }

            append_result_json(output_dir, record)
            write_csvs(output_dir)         # safe incremental writes

            progress.setdefault("success", []).append(study_id)
            save_progress(output_dir, progress)
            success_n += 1

        except KeyboardInterrupt:
            logger.warning("Interrupted — progress saved")
            save_progress(output_dir, progress)
            break
        except Exception as exc:
            logger.error(f"  [{study_id}] Error: {exc}")
            logger.error(traceback.format_exc())
            progress.setdefault("failed", []).append(study_id)
            save_progress(output_dir, progress)
            failed_n += 1

    logger.info(f"\nDone — success={success_n}  failed={failed_n}")
    logger.info(f"JSON: {output_dir}/ian_pan_disc_coords.json")
    logger.info(f"CSV:  {output_dir}/ian_pan_disc_coords.csv")


# ══════════════════════════════════════════════════════════════════════════════
# LOADER UTILITY  (import into 04_detect_lstv.py etc.)
# ══════════════════════════════════════════════════════════════════════════════

def load_ian_pan_disc_coords(json_path: str) -> Dict[str, dict]:
    """
    Load Ian Pan disc coordinate results and return dict keyed by study_id.

    Usage in 04_detect_lstv.py:
        from ian_pan_disc_coords import load_ian_pan_disc_coords
        ip = load_ian_pan_disc_coords('results/ian_pan_disc_coords/ian_pan_disc_coords.json')
        study_ip = ip.get(study_id, {})
        l5s1_conf = study_ip.get('disc_levels', {}).get('l5_s1', {}).get('peak_prob', None)
        l5s1_ras  = study_ip.get('disc_levels', {}).get('l5_s1', {}).get('world_ras_mm', None)
        dist_tss  = study_ip.get('model_agreement', {}).get('l5_s1', {}).get('dist_to_tss_mm', None)
    """
    p = Path(json_path)
    if not p.exists():
        logger.warning(f"Ian Pan disc coords not found: {p}")
        return {}
    with open(p) as f:
        records = json.load(f)
    return {str(r["study_id"]): r for r in records}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ian Pan disc heatmap → 3D world coordinate extractor")
    parser.add_argument("--input_dir",       required=True,
                        help="DICOM root: data/raw/train_images/")
    parser.add_argument("--series_csv",      required=True,
                        help="train_series_descriptions.csv")
    parser.add_argument("--output_dir",      required=True,
                        help="Output: results/ian_pan_disc_coords/")
    parser.add_argument("--checkpoint",      required=True,
                        help="Ian Pan checkpoint: models/point_net_checkpoint.pth")
    parser.add_argument("--valid_ids",       default="models/valid_id.npy")
    parser.add_argument("--spineps_dir",     default=None,
                        help="results/spineps — for NIfTI comparison (optional)")
    parser.add_argument("--totalspine_dir",  default=None,
                        help="results/totalspineseg — for NIfTI comparison (optional)")
    parser.add_argument("--mode",            choices=["trial", "debug", "prod"],
                        default="trial")
    parser.add_argument("--trial_size",      type=int, default=3)
    parser.add_argument("--debug_study_id",  default=None)
    parser.add_argument("--retry_failed",    action="store_true")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
