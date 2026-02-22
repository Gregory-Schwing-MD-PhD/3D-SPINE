#!/usr/bin/env python3
"""
05_visualize_overlay.py — Hybrid Two-Phase LSTV Overlay Visualizer
===================================================================

Label reference (from READMEs — critical: do NOT confuse label namespaces):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPINEPS seg-spine_msk.nii.gz  (subregion/semantic mask):
  43 = Costal_Process_Left  (transverse process ← TP source)
  44 = Costal_Process_Right (transverse process ← TP source)
  26 = Sacrum

SPINEPS seg-vert_msk.nii.gz  (VERIDAH instance mask):
  20=L1  21=L2  22=L3  23=L4  24=L5  25=L6  26=Sacrum
  Used for: transitional vertebra identification only.

TotalSpineSeg step2_output (sagittal + axial labeled NIfTIs):
  41=L1  42=L2  43=L3  44=L4  45=L5   ← vertebral bodies
  50 = sacrum                          ← preferred sacrum source
  ⚠ TSS labels 43 and 44 are L3 and L4 vertebrae.
    They are NOT transverse processes — do NOT use for TP detection.

registered_dir/*_spineps_reg.nii.gz:
  Registered SPINEPS seg-spine labels in axial T2w space.
  Same scheme as seg-spine_msk: 43=TP-left, 44=TP-right (not TSS!).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5-row layout:
  Row 0  Ian Pan bar  | SPINEPS sagittal overview | TSS sagittal overview
  Row 1  P1 Height L  | P1 Height R               | TSS overview (TP midpoint)
  Row 2  P1 Prox L    | P1 Prox R                 | Cross-validation panel
  Row 3  P2 Axial overview (cols 0-1)              | P2 Bbox L + R (inset)
  Row 4  Classification summary (full width)
"""

import argparse
import json
import logging
import traceback
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt, label as cc_label

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# SPINEPS seg-spine_msk subregion labels — TP source
TP_LEFT_LABEL     = 43   # Costal_Process_Left
TP_RIGHT_LABEL    = 44   # Costal_Process_Right
SPINEPS_SACRUM    = 26   # Sacrum in seg-spine_msk

# VERIDAH seg-vert_msk instance labels — TV identification only
VERIDAH_L5        = 24
VERIDAH_L6        = 25
LUMBAR_LABELS     = [25, 24, 23, 22, 21, 20]
VERIDAH_NAMES     = {20:'L1', 21:'L2', 22:'L3', 23:'L4', 24:'L5', 25:'L6'}

# TotalSpineSeg labels — sacrum + vertebral body labels
# CRITICAL: TSS 43=L3, 44=L4 (vertebrae) — completely different from SPINEPS 43/44=TP!
TSS_SACRUM_LABEL  = 50   # sacrum (preferred sacrum source)
TSS_L5_LABEL      = 45   # vertebrae_L5
TSS_SACRUM_LABELS = {TSS_SACRUM_LABEL}

# Cross-validation thresholds
XVAL_MIN_DICE     = 0.30
XVAL_MAX_CENTROID = 20.0

TP_HEIGHT_MM      = 19.0
CONTACT_DIST_MM   = 2.0
BBOX_HALF         = 20
DISPLAY_DILATION  = 2

IAN_PAN_LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
IAN_PAN_LABELS = ['L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']

VERIDAH_LABEL_COLORS = {
    20: ([0.15, 0.35, 0.75], 'L1'), 21: ([0.20, 0.45, 0.80], 'L2'),
    22: ([0.25, 0.55, 0.85], 'L3'), 23: ([0.30, 0.65, 0.90], 'L4'),
    24: ([0.40, 0.80, 1.00], 'L5'), 25: ([0.00, 0.25, 0.65], 'L6'),
    26: ([1.00, 0.55, 0.00], 'Sacrum'),
}
# TSS lumbar vertebral bodies: 41=L1 42=L2 43=L3 44=L4 45=L5
# (43/44 here are vertebral bodies, NOT transverse processes)
TSS_LUMBAR_COLORS = {
    41: ([0.15, 0.35, 0.75], 'L1'), 42: ([0.20, 0.45, 0.80], 'L2'),
    43: ([0.25, 0.55, 0.85], 'L3'), 44: ([0.30, 0.65, 0.90], 'L4'),
    45: ([0.40, 0.80, 1.00], 'L5'),
}
COLOR_TP_LEFT    = [1.00, 0.10, 0.10]
COLOR_TP_RIGHT   = [0.00, 0.80, 1.00]
SACRUM_COLOR     = [1.00, 0.55, 0.00]
TSS_SACRUM_COLOR = [1.00, 0.40, 0.00]
LINE_TV          = 'cyan'
LINE_MINDIST     = '#FF8C00'
BG_DARK          = '#0d0d1a'
BG_MID           = '#1a1a2e'


# ============================================================================
# STUDY SELECTION
# ============================================================================

def select_studies(csv_path: Path, top_n: int, rank_by: str, valid_ids) -> list:
    if not csv_path.exists():
        raise FileNotFoundError(f"Uncertainty CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)
    if valid_ids is not None:
        before = len(df)
        df = df[df['study_id'].isin(valid_ids)]
        logger.info(f"Filtered to {len(df)} studies via valid_ids ({before-len(df)} excluded)")
    if rank_by not in df.columns:
        raise ValueError(f"Column '{rank_by}' not in CSV. Available: {', '.join(df.columns)}")
    df_sorted  = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    top_ids    = df_sorted.head(top_n)['study_id'].tolist()
    bottom_ids = df_sorted.tail(top_n)['study_id'].tolist()
    seen, selected = set(), []
    for sid in top_ids + bottom_ids:
        if sid not in seen:
            selected.append(sid); seen.add(sid)
    logger.info(f"Rank by: {rank_by}  Top {top_n}: {top_ids}  Bottom {top_n}: {bottom_ids}")
    return selected


# ============================================================================
# NIBABEL HELPERS
# ============================================================================

def load_canonical(path: Path):
    """Load NIfTI, reorient to RAS canonical, robustly reduce to 3D.

    4D volumes: strip trailing size-1 axes, then select volume 0 on axis 3.
    Raises ValueError if result is not 3D.
    """
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()

    # Strip trailing size-1 dims
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]

    # Still 4D → multiple volumes; take first on axis 3 (time/echo)
    if data.ndim == 4:
        logger.debug(f"  4D {path.name} shape={data.shape}; selecting volume 0 on axis 3")
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(
            f"Cannot reduce {path.name} to 3D: shape after squeeze = {data.shape}"
        )
    return data, nii


def voxel_size_mm(nii):
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


def norm(arr):
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo + 1e-8)


def _sag_sl(vol, x):
    if vol is None: return np.zeros((2, 2))
    return vol[min(x, vol.shape[0]-1), :, :]


def _ax_sl(vol, z):
    if vol is None: return np.zeros((2, 2))
    return vol[:, :, min(max(z, 0), vol.shape[2]-1)]


def overlay_mask(ax, mask2d, color_rgb, alpha=0.65):
    if mask2d is None or not np.asarray(mask2d).any(): return
    rgba = np.zeros((*mask2d.shape, 4), dtype=float)
    rgba[mask2d] = [*color_rgb, alpha]
    # rgba is (M, N, 4); .T gives (4, N, M) which imshow rejects.
    # Swap only the spatial axes to get (N, M, 4) for origin='lower'.
    ax.imshow(rgba.transpose(1, 0, 2), origin='lower')


def dilate2d(mask, r=2):
    if not np.asarray(mask).any() or r < 1: return mask
    struct = np.ones((r*2+1, r*2+1), dtype=bool)
    return binary_dilation(mask, structure=struct)


def _unavailable(ax, label):
    ax.set_facecolor(BG_DARK)
    ax.text(0.5, 0.5, label, ha='center', va='center',
            color='#666666', fontsize=10, transform=ax.transAxes,
            multialignment='center')
    ax.axis('off')


def _hline(ax, z, color, label=''):
    if z is None: return
    ax.axhline(y=float(z), color=color, lw=1.4, ls='--', alpha=0.9)
    if label:
        ax.text(3, float(z)+1, label, color=color, fontsize=7, va='bottom', fontweight='bold')


# ============================================================================
# MASK OPERATIONS
# ============================================================================

def get_tv_z_range(vert_data, tv_label):
    mask = vert_data == tv_label
    if not mask.any(): return None
    z = np.where(mask)[2]
    return int(z.min()), int(z.max())


def isolate_tp_at_tv(subreg, tp_label, z_min, z_max):
    """Extract TP label from seg-spine_msk within TV z-range."""
    tp_full = subreg == tp_label
    iso     = np.zeros_like(tp_full)
    iso[:, :, z_min:z_max+1] = tp_full[:, :, z_min:z_max+1]
    return iso


def inferiormost_tp_cc(tp3d, sacrum3d=None):
    if not tp3d.any(): return np.zeros_like(tp3d, dtype=bool)
    labeled, n = cc_label(tp3d)
    if n <= 1: return tp3d.astype(bool)
    sac_z_min = None
    if sacrum3d is not None and sacrum3d.any():
        sac_z_min = int(np.where(sacrum3d)[2].min())
    cc_info = []
    for i in range(1, n+1):
        comp = (labeled == i); zc = np.where(comp)[2]
        cc_info.append((float(zc.mean()), int(zc.max()), comp))
    cc_info.sort(key=lambda t: t[0])
    if sac_z_min is not None:
        cands = [c for _, zm, c in cc_info if zm < sac_z_min]
        if cands: return cands[0].astype(bool)
    return cc_info[0][2].astype(bool)


def min_dist_3d(mask_a, mask_b, vox_mm):
    if not mask_a.any() or not mask_b.any(): return float('inf'), None, None
    dt = distance_transform_edt(~mask_b, sampling=vox_mm)
    dist_at_a = np.where(mask_a, dt, np.inf)
    flat  = int(np.argmin(dist_at_a))
    vox_a = np.array(np.unravel_index(flat, mask_a.shape))
    dist_mm = float(dt[tuple(vox_a)])
    z_lo = max(0, int(vox_a[2])-20); z_hi = min(mask_b.shape[2], int(vox_a[2])+20)
    sub  = mask_b[:, :, z_lo:z_hi]
    if sub.any(): coords = np.array(np.where(sub)); coords[2] += z_lo
    else:         coords = np.array(np.where(mask_b))
    d2    = ((coords.T * vox_mm - vox_a * vox_mm)**2).sum(axis=1)
    vox_b = coords[:, int(np.argmin(d2))]
    return dist_mm, vox_a, vox_b


def best_x_for_tp_height(tp3d, vox_z_mm):
    if not tp3d.any(): return tp3d.shape[0]//2, 0.0
    best_x, best_span = tp3d.shape[0]//2, 0.0
    for x in range(tp3d.shape[0]):
        col = tp3d[x]
        if not col.any(): continue
        zc = np.where(col.any(axis=0))[0]
        if zc.size < 2: continue
        span = (zc.max() - zc.min()) * vox_z_mm
        if span > best_span: best_span = span; best_x = x
    return best_x, best_span


def tp_blob_z_midpoint(tp3d):
    if not tp3d.any(): return tp3d.shape[2]//2
    zc = np.where(tp3d)[2]
    return int((zc.min() + zc.max()) // 2)


def largest_cc_2d(mask2d):
    if not mask2d.any(): return np.zeros_like(mask2d, dtype=bool)
    labeled, n = cc_label(mask2d)
    if n == 0: return np.zeros_like(mask2d, dtype=bool)
    sizes = [(labeled == i).sum() for i in range(1, n+1)]
    return (labeled == (int(np.argmax(sizes)) + 1))


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a, b = mask_a.astype(bool), mask_b.astype(bool)
    intersection = (a & b).sum()
    denom = a.sum() + b.sum()
    return float('nan') if denom == 0 else float(2 * intersection / denom)


def centroid_mm(mask: np.ndarray, vox_mm: np.ndarray):
    coords = np.array(np.where(mask))
    if coords.size == 0: return None
    return coords.mean(axis=1) * vox_mm


def compute_cross_validation(sag_sp, sag_vert, sag_tss, vox_mm):
    """
    Cross-validate SPINEPS ↔ TSS label consistency.

    Returns dict with:
      sacrum_dice         : Dice(SPINEPS sacrum label 26, TSS sacrum label 50)
      l5_centroid_dist_mm : ||centroid(VERIDAH L5=24) - centroid(TSS L5=45)||
      tp_in_spine_mask    : bool — TP labels 43/44 found in seg-spine_msk
      tss_has_43_44       : bool — TSS has 43/44 (expected: L3/L4 bodies, not TP)
      warnings            : list[str]
    """
    xv = {'sacrum_dice': None, 'l5_centroid_dist_mm': None,
          'tp_in_spine_mask': False, 'tss_has_43_44': False, 'warnings': []}

    # 1. Sacrum Dice: SPINEPS label 26 vs TSS label 50
    sp_sac  = (sag_sp  == SPINEPS_SACRUM)
    tss_sac = (sag_tss == TSS_SACRUM_LABEL) if sag_tss is not None else None
    if not sp_sac.any():
        xv['warnings'].append('SPINEPS sacrum(26) absent from seg-spine_msk')
    if tss_sac is None or not tss_sac.any():
        xv['warnings'].append('TSS sacrum(50) absent from sagittal TSS mask')
    if sp_sac.any() and tss_sac is not None and tss_sac.any():
        d = dice_coefficient(sp_sac, tss_sac)
        xv['sacrum_dice'] = round(d, 4)
        if d < XVAL_MIN_DICE:
            xv['warnings'].append(
                f'Sacrum Dice={d:.3f} < {XVAL_MIN_DICE} — masks disagree significantly')

    # 2. L5 centroid: VERIDAH label 24 vs TSS label 45
    ver_l5 = (sag_vert == VERIDAH_L5) if sag_vert is not None else None
    tss_l5 = (sag_tss  == TSS_L5_LABEL) if sag_tss is not None else None
    if ver_l5 is None or not ver_l5.any():
        xv['warnings'].append('VERIDAH L5(24) absent from seg-vert_msk')
    if tss_l5 is None or not tss_l5.any():
        xv['warnings'].append('TSS L5(45) absent from sagittal TSS mask')
    if (ver_l5 is not None and ver_l5.any() and
            tss_l5 is not None and tss_l5.any()):
        c_v = centroid_mm(ver_l5, vox_mm)
        c_t = centroid_mm(tss_l5, vox_mm)
        if c_v is not None and c_t is not None:
            dist = float(np.linalg.norm(c_v - c_t))
            xv['l5_centroid_dist_mm'] = round(dist, 2)
            if dist > XVAL_MAX_CENTROID:
                xv['warnings'].append(
                    f'L5 centroid dist={dist:.1f}mm > {XVAL_MAX_CENTROID}mm — '
                    f'VERIDAH and TSS L5 labels disagree')

    # 3. TP presence in seg-spine_msk
    sp_labels = set(np.unique(sag_sp).tolist())
    xv['tp_in_spine_mask'] = (TP_LEFT_LABEL  in sp_labels or
                               TP_RIGHT_LABEL in sp_labels)
    if not xv['tp_in_spine_mask']:
        xv['warnings'].append(
            f'TP labels {TP_LEFT_LABEL}/{TP_RIGHT_LABEL} absent from seg-spine_msk')

    # 4. TSS namespace check — TSS 43/44 present means L3/L4 bodies (expected)
    if sag_tss is not None:
        tss_labels = set(np.unique(sag_tss).tolist())
        xv['tss_has_43_44'] = bool(43 in tss_labels or 44 in tss_labels)

    return xv


# ============================================================================
# RULERS
# ============================================================================

def _draw_height_ruler(ax, mask2d, vox_z_mm, color='yellow'):
    lcc = largest_cc_2d(mask2d)
    if not lcc.any(): return 0.0
    zc = np.where(lcc.any(axis=0))[0]
    if zc.size < 2: return 0.0
    z_lo, z_hi = int(zc.min()), int(zc.max())
    span_mm = (z_hi - z_lo) * vox_z_mm
    mid_z   = zc[len(zc)//2]
    col_at  = lcc[:, mid_z]
    x_mid   = int(np.where(col_at)[0].mean()) if col_at.any() else lcc.shape[0]//2
    tick    = max(3, int(lcc.shape[0]*0.025))
    ax.plot([x_mid, x_mid], [z_lo, z_hi], color=color, lw=1.8, alpha=0.95)
    for z_end in (z_lo, z_hi):
        ax.plot([x_mid-tick, x_mid+tick], [z_end, z_end], color=color, lw=1.8)
    ax.text(x_mid+tick+2, (z_lo+z_hi)/2, f'{span_mm:.1f} mm',
            color=color, fontsize=8, va='center', fontweight='bold')
    return span_mm


def _draw_gap_ruler(ax, tp2d, sac2d, vox_z_mm, color='#FF8C00'):
    tp_lcc  = largest_cc_2d(tp2d)
    sac_lcc = largest_cc_2d(sac2d)
    if not tp_lcc.any() or not sac_lcc.any(): return float('inf')
    tp_zc  = np.where(tp_lcc.any(axis=0))[0]
    sac_zc = np.where(sac_lcc.any(axis=0))[0]
    if tp_zc.size == 0 or sac_zc.size == 0: return float('inf')
    z_tp_inf  = int(tp_zc.min())
    z_sac_sup = int(sac_zc.max())
    gap_mm    = (z_tp_inf - z_sac_sup) * vox_z_mm
    x_ruler   = (int(np.where(tp_lcc.any(axis=1))[0].mean()) +
                 int(np.where(sac_lcc.any(axis=1))[0].mean())) // 2
    tick = max(3, int(tp2d.shape[0]*0.025))
    if z_sac_sup < z_tp_inf:
        ax.plot([x_ruler, x_ruler], [z_sac_sup, z_tp_inf], color=color, lw=1.8)
        for z_end in (z_sac_sup, z_tp_inf):
            ax.plot([x_ruler-tick, x_ruler+tick], [z_end, z_end], color=color, lw=1.8)
        lbl = f'{gap_mm:.1f} mm gap'
    else:
        z_mid = (z_tp_inf+z_sac_sup)//2
        ax.plot([x_ruler-tick, x_ruler+tick], [z_mid, z_mid], color=color, lw=2.0)
        lbl = 'overlap'
    ax.text(x_ruler+tick+2, (z_sac_sup+z_tp_inf)/2, lbl,
            color=color, fontsize=8, va='center', fontweight='bold')
    return gap_mm


# ============================================================================
# ROW 0 PANELS
# ============================================================================

def _panel_ian_pan(ax, study_id, uncertainty_row):
    ax.set_facecolor(BG_DARK)
    if uncertainty_row is None:
        _unavailable(ax, 'Ian Pan uncertainty\nnot available'); return
    confs = [uncertainty_row.get(f'{l}_confidence', float('nan')) for l in IAN_PAN_LEVELS]
    entrs = [uncertainty_row.get(f'{l}_entropy',    float('nan')) for l in IAN_PAN_LEVELS]
    x     = np.arange(len(IAN_PAN_LABELS))
    w     = 0.38
    max_c  = max((v for v in confs if not np.isnan(v)), default=0)
    colors = ['#e63946' if (not np.isnan(c) and c == max_c) else '#457b9d' for c in confs]
    bars   = ax.bar(x - w/2, confs, w, label='Confidence', color=colors, alpha=0.9)
    max_e  = max((v for v in entrs if not np.isnan(v)), default=1)
    enorm  = [e/max_e if not np.isnan(e) else 0 for e in entrs]
    ax.bar(x + w/2, enorm, w, label='Entropy (norm)', color='#f4a261', alpha=0.7)
    ax.axhline(0.5, color='white', lw=0.7, ls=':', alpha=0.35)
    ax.set_xticks(x); ax.set_xticklabels(IAN_PAN_LABELS, color='white', fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#444')
    ax.set_facecolor(BG_DARK)
    ax.set_ylabel('Score', color='white', fontsize=8)
    ax.legend(fontsize=7, labelcolor='white', facecolor=BG_MID, edgecolor='#444')
    l5s1_conf = uncertainty_row.get('l5_s1_confidence', float('nan'))
    ax.set_title(f'Ian Pan Epistemic Uncertainty\nL5-S1 confidence = {l5s1_conf:.3f}',
                 fontsize=9, color='white')
    for bar, val in zip(bars, confs):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.025, f'{val:.2f}',
                    ha='center', va='bottom', color='white', fontsize=7)


def _panel_spineps_overview(ax, sag_bg, vert_data, sag_sp, tp_L, tp_R,
                             x_mid, z_tv_mid, tv_name):
    """Show VERIDAH vertebrae (seg-vert) + TP blobs (seg-spine 43/44) + sacrum (26)."""
    ax.imshow(norm(_sag_sl(sag_bg, x_mid)).T, cmap='gray', origin='lower', alpha=0.78)
    patches = []
    if vert_data is not None:
        for label, (color, name) in VERIDAH_LABEL_COLORS.items():
            m = _sag_sl(vert_data == label, x_mid)
            if m.any():
                overlay_mask(ax, m, color, 0.40)
                patches.append(mpatches.Patch(color=color, label=f'VER {name}'))
    if sag_sp is not None:
        sac = _sag_sl(sag_sp == SPINEPS_SACRUM, x_mid)
        if sac.any():
            overlay_mask(ax, sac, SACRUM_COLOR, 0.55)
            patches.append(mpatches.Patch(color=SACRUM_COLOR, label='SPINEPS Sac(26)'))
    for mask, color, name in ((tp_L, COLOR_TP_LEFT,  'L TP(seg-spine 43)'),
                               (tp_R, COLOR_TP_RIGHT, 'R TP(seg-spine 44)')):
        if mask is not None and mask.any():
            overlay_mask(ax, dilate2d(_sag_sl(mask, x_mid), DISPLAY_DILATION), color, 0.85)
            patches.append(mpatches.Patch(color=color, label=name))
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=5, framealpha=0.55)
    _hline(ax, z_tv_mid, LINE_TV, tv_name)
    ax.set_title(f'SPINEPS Sagittal  x={x_mid}\n'
                 f'VERIDAH(seg-vert) + TP(seg-spine 43/44) + Sac(26)',
                 fontsize=9, color='white')
    ax.axis('off')


def _panel_tss_overview(ax, sag_bg, sag_tss, x_mid, z_tv_mid, tv_name):
    """TSS sagittal: vertebrae 41-45 (L1-L5) + sacrum 50.
    Note: TSS 43=L3, 44=L4 vertebral bodies — shown with those anatomical names."""
    ax.imshow(norm(_sag_sl(sag_bg, x_mid)).T, cmap='gray', origin='lower', alpha=0.78)
    if sag_tss is None:
        ax.text(0.5, 0.5, 'TSS sagittal\nnot available', ha='center', va='center',
                color='#666', fontsize=10, transform=ax.transAxes)
        ax.axis('off'); return
    patches = []
    for label, (color, name) in TSS_LUMBAR_COLORS.items():
        sl = _sag_sl(sag_tss == label, x_mid)
        if sl.any():
            overlay_mask(ax, sl, color, 0.45)
            patches.append(mpatches.Patch(color=color, label=f'TSS {name}({label})'))
    sac_sl = _sag_sl(sag_tss == TSS_SACRUM_LABEL, x_mid)
    if sac_sl.any():
        overlay_mask(ax, sac_sl, TSS_SACRUM_COLOR, 0.70)
        patches.append(mpatches.Patch(color=TSS_SACRUM_COLOR, label='TSS Sacrum(50)'))
    else:
        ax.text(0.5, 0.04, '⚠ TSS sacrum(50) not on this x-slice',
                transform=ax.transAxes, ha='center', color='#FF8C00', fontsize=7)
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=5, framealpha=0.55)
    _hline(ax, z_tv_mid, LINE_TV, tv_name)
    ax.set_title(f'TotalSpineSeg Sagittal  x={x_mid}\n'
                 f'L1-L5(41-45) + Sacrum(50)  [TSS 43=L3, 44=L4 ≠ SPINEPS TP!]',
                 fontsize=9, color='white')
    ax.axis('off')


# ============================================================================
# ROWS 1-2 PANELS
# ============================================================================

def _panel_p1_height(ax, sag_img, tp_this, tp_other, sag_sp,
                     side_name, x_idx, span_mm, vox_z_mm, z_ref, tv_name):
    c_this  = COLOR_TP_LEFT  if side_name == 'Left' else COLOR_TP_RIGHT
    c_other = COLOR_TP_RIGHT if side_name == 'Left' else COLOR_TP_LEFT
    tp_lbl  = TP_LEFT_LABEL  if side_name == 'Left' else TP_RIGHT_LABEL
    ax.imshow(norm(_sag_sl(sag_img, x_idx)).T, cmap='gray', origin='lower', alpha=0.80)
    if tp_other is not None and tp_other.any():
        overlay_mask(ax, _sag_sl(tp_other, x_idx), c_other, 0.22)
    if tp_this is not None and tp_this.any():
        sl = _sag_sl(tp_this, x_idx)
        overlay_mask(ax, sl, c_this, 0.85)
        _draw_height_ruler(ax, sl, vox_z_mm)
    if sag_sp is not None:
        overlay_mask(ax, _sag_sl(sag_sp == SPINEPS_SACRUM, x_idx), SACRUM_COLOR, 0.40)
    _hline(ax, z_ref, LINE_TV, 'TV ref')
    flag = '✓' if span_mm < TP_HEIGHT_MM else f'✗ ≥{TP_HEIGHT_MM:.0f}mm → Type I'
    ax.set_title(f'P1 Height — {side_name} (seg-spine lbl {tp_lbl})  x={x_idx}\n'
                 f'{span_mm:.1f} mm  {flag}', fontsize=9, color='white')
    ax.legend(handles=[mpatches.Patch(color=c_this, label=f'{side_name} TP(seg-spine {tp_lbl})'),
                       mpatches.Patch(color=SACRUM_COLOR, label='SPINEPS Sac(26)')],
              loc='lower right', fontsize=6, framealpha=0.55)
    ax.axis('off')


def _panel_p1_proximity(ax, sag_img, tp_this, tp_other, sag_sp, sag_tss,
                         side_name, x_idx, dist_mm, vox_z_mm, z_ref, tv_name):
    c_this  = COLOR_TP_LEFT  if side_name == 'Left' else COLOR_TP_RIGHT
    c_other = COLOR_TP_RIGHT if side_name == 'Left' else COLOR_TP_LEFT
    ax.imshow(norm(_sag_sl(sag_img, x_idx)).T, cmap='gray', origin='lower', alpha=0.80)
    this_sl = (_sag_sl(tp_this, x_idx) if (tp_this is not None and tp_this.any())
               else np.zeros((2, 2), bool))
    # Sacrum: prefer TSS label 50, fall back to SPINEPS label 26
    if sag_tss is not None and (sag_tss == TSS_SACRUM_LABEL).any():
        sac_sl    = _sag_sl(sag_tss == TSS_SACRUM_LABEL, x_idx)
        sac_label = 'TSS Sacrum(50)'
        sac_color = TSS_SACRUM_COLOR
    else:
        sac_sl    = (_sag_sl(sag_sp == SPINEPS_SACRUM, x_idx)
                     if sag_sp is not None else np.zeros((2, 2), bool))
        sac_label = 'SPINEPS Sacrum(26)'
        sac_color = SACRUM_COLOR
    if tp_other is not None and tp_other.any():
        overlay_mask(ax, _sag_sl(tp_other, x_idx), c_other, 0.22)
    overlay_mask(ax, this_sl, c_this, 0.85)
    overlay_mask(ax, sac_sl, sac_color, 0.65)
    _draw_gap_ruler(ax, this_sl, sac_sl, vox_z_mm, color=LINE_MINDIST)
    _hline(ax, z_ref, LINE_MINDIST, 'gap')
    dist_str = f'{dist_mm:.1f} mm' if np.isfinite(dist_mm) else 'N/A'
    contact  = np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM
    flag     = '✗ contact → P2' if contact else '✓ no contact'
    ax.set_title(f'P1 Proximity — {side_name}  x={x_idx}\n{dist_str}  {flag}',
                 fontsize=9, color='white')
    ax.legend(handles=[mpatches.Patch(color=c_this,    label=f'{side_name} TP'),
                       mpatches.Patch(color=sac_color, label=sac_label)],
              loc='lower right', fontsize=6, framealpha=0.55)
    ax.axis('off')


def _panel_cross_validation(ax, xv, sag_sp, sag_vert, sag_tss, x_mid, vox_z_mm):
    """
    Cross-validation panel (Row 2, Col 2).
    Overlays SPINEPS sacrum(26) vs TSS sacrum(50) and VERIDAH L5(24) vs TSS L5(45).
    Displays Dice / centroid metrics and label-namespace warning.
    """
    ax.set_facecolor(BG_DARK)
    if sag_sp is None or sag_tss is None:
        _unavailable(ax, 'Cross-validation\nnot available\n(missing masks)')
        return

    bg = norm(_sag_sl(sag_sp.astype(float), x_mid))
    ax.imshow(bg.T, cmap='gray', origin='lower', alpha=0.45)
    patches = []

    sp_sac_sl = _sag_sl(sag_sp == SPINEPS_SACRUM, x_mid)
    if sp_sac_sl.any():
        overlay_mask(ax, sp_sac_sl, SACRUM_COLOR, 0.55)
        patches.append(mpatches.Patch(color=SACRUM_COLOR, label='SPINEPS Sac(26)'))

    tss_sac_sl = _sag_sl(sag_tss == TSS_SACRUM_LABEL, x_mid)
    if tss_sac_sl.any():
        overlay_mask(ax, tss_sac_sl, TSS_SACRUM_COLOR, 0.55)
        patches.append(mpatches.Patch(color=TSS_SACRUM_COLOR, label='TSS Sac(50)'))

    if sag_vert is not None:
        ver_l5_sl = _sag_sl(sag_vert == VERIDAH_L5, x_mid)
        if ver_l5_sl.any():
            overlay_mask(ax, ver_l5_sl, [0.40, 0.80, 1.00], 0.55)
            patches.append(mpatches.Patch(color=[0.40, 0.80, 1.00], label='VERIDAH L5(24)'))

    tss_l5_sl = _sag_sl(sag_tss == TSS_L5_LABEL, x_mid)
    if tss_l5_sl.any():
        overlay_mask(ax, tss_l5_sl, [0.80, 1.00, 0.40], 0.55)
        patches.append(mpatches.Patch(color=[0.80, 1.00, 0.40], label='TSS L5(45)'))

    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=5, framealpha=0.55)

    dice  = xv.get('sacrum_dice')
    cdist = xv.get('l5_centroid_dist_mm')
    tp_ok = xv.get('tp_in_spine_mask', False)
    warns = xv.get('warnings', [])

    dice_c  = '#2dc653' if (dice  is not None and dice  >= XVAL_MIN_DICE)       else '#e63946'
    cdist_c = '#2dc653' if (cdist is not None and cdist <= XVAL_MAX_CENTROID)   else '#e63946'
    tp_c    = '#2dc653' if tp_ok else '#e63946'

    # Coloured metric lines drawn on top of each other (last wins per position)
    base_lines = [
        '',
        '',
        '',
        '',
        'TSS 43=L3, 44=L4 (vertebrae)',
        'SPINEPS 43=TP-L, 44=TP-R',
        '(different label namespaces)',
    ]
    if warns:
        base_lines += [''] + [f'⚠ {w[:48]}' for w in warns[:4]]
    ax.text(0.03, 0.97, '\n'.join(base_lines), transform=ax.transAxes,
            va='top', ha='left', fontsize=7.5, family='monospace',
            color='white', linespacing=1.45)

    metric_lines = [
        (f"Sacrum Dice: {f'{dice:.3f}' if dice is not None else 'N/A'}", dice_c),
        (f"L5 centroid: {f'{cdist:.1f} mm' if cdist is not None else 'N/A'}", cdist_c),
        (f"TP in seg-spine: {'✓' if tp_ok else '✗ MISSING'}", tp_c),
    ]
    for i, (line, color) in enumerate(metric_lines):
        ax.text(0.03, 0.97 - i * 0.088, line, transform=ax.transAxes,
                va='top', ha='left', fontsize=7.5, family='monospace',
                color=color, linespacing=1.45)

    ax.set_title('Cross-Validation: SPINEPS ↔ TSS\n'
                 'Sacrum Dice | L5 centroid | TP presence',
                 fontsize=9, color='white')
    ax.axis('off')


# ============================================================================
# ROW 3 PANELS
# ============================================================================

def _best_z_for_sacrum(ax_tss):
    if ax_tss is None: return None
    sac = (ax_tss == TSS_SACRUM_LABEL)
    if not sac.any(): return None
    z_vals = np.where(sac.any(axis=(0, 1)))[0]
    return int(z_vals.max()) if z_vals.size else None


def _best_z_for_tp(ax_spineps):
    """Median z-slice with registered SPINEPS TP labels 43 or 44."""
    if ax_spineps is None: return None
    tp = ((ax_spineps == TP_LEFT_LABEL) | (ax_spineps == TP_RIGHT_LABEL))
    if not tp.any(): return None
    z_vals = np.where(tp.any(axis=(0, 1)))[0]
    return int(z_vals[len(z_vals)//2])


def _panel_p2_axial_overview(ax, ax_t2w, ax_spineps, ax_tss, z_slice):
    """
    ax_spineps: registered SPINEPS seg-spine labels (43=TP-L, 44=TP-R).
    ax_tss:     native TSS axial labels (50=sacrum; 43/44 here = L3/L4 bodies — not shown).
    """
    if ax_t2w is None:
        _unavailable(ax, 'Axial T2w\nnot available'); return
    bg = _ax_sl(ax_t2w, z_slice)
    ax.imshow(norm(bg).T, cmap='gray', origin='lower', alpha=0.85)
    patches = []

    if ax_tss is not None:
        sac = (ax_tss == TSS_SACRUM_LABEL)
        sl  = _ax_sl(sac, z_slice)
        if sl.any():
            overlay_mask(ax, sl, TSS_SACRUM_COLOR, 0.65)
            patches.append(mpatches.Patch(color=TSS_SACRUM_COLOR, label='TSS Sacrum(50)'))
        else:
            ax.text(0.5, 0.03, f'⚠ TSS sacrum(50) not at z={z_slice}',
                    transform=ax.transAxes, ha='center', color='#FF8C00', fontsize=7)

    n_tp_total = 0
    if ax_spineps is not None:
        for label, color, name in (
            (TP_LEFT_LABEL,  COLOR_TP_LEFT,  'L TP(reg seg-spine 43)'),
            (TP_RIGHT_LABEL, COLOR_TP_RIGHT, 'R TP(reg seg-spine 44)'),
        ):
            m = dilate2d(_ax_sl(ax_spineps == label, z_slice), DISPLAY_DILATION)
            if m.any():
                overlay_mask(ax, m, color, 0.80)
                patches.append(mpatches.Patch(color=color, label=name))
        n_tp_total = int(((ax_spineps == TP_LEFT_LABEL) |
                          (ax_spineps == TP_RIGHT_LABEL)).sum())
        if n_tp_total == 0:
            ax.text(0.5, 0.97, '⚠ No TP labels(43/44) in registered SPINEPS volume',
                    transform=ax.transAxes, ha='center', va='top', color='#FF8C00', fontsize=7)
    else:
        ax.text(0.5, 0.97, '⚠ Registered SPINEPS mask missing',
                transform=ax.transAxes, ha='center', va='top', color='#FF8C00', fontsize=7)

    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.55)
    ax.set_title(f'P2 Axial Overview  z={z_slice}  (TP voxels: {n_tp_total})\n'
                 f'TSS Sacrum(50) + Reg. SPINEPS TP(43/44=costal process)',
                 fontsize=9, color='white')
    ax.axis('off')


def _panel_p2_bboxes_inset(parent_ax, fig, ax_t2w, midpt_L, midpt_R,
                            cls_L, cls_R, feat_L, feat_R):
    parent_ax.axis('off')
    if ax_t2w is None:
        _unavailable(parent_ax, 'Axial T2w\nnot available\n(bbox impossible)')
        return

    nx, ny, nz = ax_t2w.shape
    pos   = parent_ax.get_position()
    sub_w = pos.width / 2
    ax_L  = fig.add_axes([pos.x0,         pos.y0, sub_w, pos.height], facecolor=BG_DARK)
    ax_R  = fig.add_axes([pos.x0 + sub_w, pos.y0, sub_w, pos.height], facecolor=BG_DARK)

    def _render(sub_ax, midpt, side, cls, feat):
        c = COLOR_TP_LEFT if side == 'Left' else COLOR_TP_RIGHT
        if midpt is None:
            _unavailable(sub_ax, f'P2 Bbox {side}\nnot available\n(no contact)'); return
        x0, y0, z0 = int(midpt[0]), int(midpt[1]), int(midpt[2])
        if not (0 <= z0 < nz):
            _unavailable(sub_ax, f'z={z0} out of range\n{side}'); return
        x_lo = max(0, x0-BBOX_HALF); x_hi = min(nx, x0+BBOX_HALF)
        y_lo = max(0, y0-BBOX_HALF); y_hi = min(ny, y0+BBOX_HALF)
        patch = ax_t2w[x_lo:x_hi, y_lo:y_hi, z0]
        if patch.size == 0:
            _unavailable(sub_ax, f'Empty patch\n{side}'); return
        sub_ax.imshow(norm(patch).T, cmap='hot', origin='lower', alpha=1.0)
        cx, cy = x0 - x_lo, y0 - y_lo
        sub_ax.axhline(y=cx, color=c, lw=0.9, ls=':', alpha=0.7)
        sub_ax.axvline(x=cy, color=c, lw=0.9, ls=':', alpha=0.7)
        sub_ax.scatter([cy], [cx], c=[c], s=55, marker='+', zorder=5, linewidths=1.8)
        cls_color = '#e63946' if cls == 'Type III' else '#ff9f1c'
        sub_ax.text(0.05, 0.97, f'→ {cls}', transform=sub_ax.transAxes,
                    va='top', ha='left', color=cls_color, fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc=BG_MID, alpha=0.75))
        feat_text = ''
        if feat and feat.get('valid'):
            feat_text = f"mean={feat['patch_mean']:.0f}  CV={feat['coeff_var']:.2f}"
        sub_ax.set_title(f'P2 Bbox {side}  z={z0}\n{feat_text}', fontsize=8, color='white')
        sub_ax.axis('off')

    _render(ax_L, midpt_L, 'Left',  cls_L, feat_L)
    _render(ax_R, midpt_R, 'Right', cls_R, feat_R)


# ============================================================================
# ROW 4: Summary
# ============================================================================

def _panel_summary_wide(ax, study_id, result, tv_name, span_L, span_R,
                         dist_L, dist_R, uncertainty_row, xv):
    ax.axis('off'); ax.set_facecolor(BG_MID)

    def _d(v): return f'{v:.1f}' if (v is not None and np.isfinite(v)) else 'N/A'
    cls_L = (result or {}).get('left',  {}).get('classification', '—')
    cls_R = (result or {}).get('right', {}).get('classification', '—')
    p2L   = ((result or {}).get('left',  {}).get('phase2') or {})
    p2R   = ((result or {}).get('right', {}).get('phase2') or {})

    ian_str = ('not available' if uncertainty_row is None else
               f"{uncertainty_row.get('l5_s1_confidence', float('nan')):.3f} conf  /  "
               f"{uncertainty_row.get('l5_s1_entropy', float('nan')):.3f} entropy")

    dice  = xv.get('sacrum_dice')
    cdist = xv.get('l5_centroid_dist_mm')
    tp_ok = xv.get('tp_in_spine_mask', False)
    xv_warns = xv.get('warnings', [])
    xv_str   = (f"Sacrum Dice={f'{dice:.3f}' if dice is not None else 'N/A'}  "
                f"L5 ctr={f'{cdist:.1f}mm' if cdist is not None else 'N/A'}  "
                f"TP={'✓' if tp_ok else '✗ MISSING'}")

    left_col = [
        f'Study    : {study_id}',
        f'TV       : {tv_name}',
        f'Pipeline : Hybrid Phase1-Sagittal / Phase2-Axial',
        '',
        '── Ian Pan (L5-S1) ───────────────────────────',
        f'  {ian_str}',
        '',
        '── Cross-Validation (SPINEPS ↔ TSS) ──────────',
        f'  {xv_str}',
        *(f'  ⚠ {w[:58]}' for w in xv_warns[:3]),
        '',
        '── Phase 1 (Sagittal) ────────────────────────',
        '  TP source: seg-spine_msk labels 43(L)/44(R)',
        f'  Left  TP height  : {_d(span_L):>7} mm  (thresh {TP_HEIGHT_MM:.0f} mm)',
        f'  Right TP height  : {_d(span_R):>7} mm  (thresh {TP_HEIGHT_MM:.0f} mm)',
        f'  Left  TP–Sacrum  : {_d(dist_L):>7} mm  (thresh {CONTACT_DIST_MM:.0f} mm)',
        f'  Right TP–Sacrum  : {_d(dist_R):>7} mm  (thresh {CONTACT_DIST_MM:.0f} mm)',
    ]
    right_col = [
        '── Phase 2 (Axial) ───────────────────────────',
        f'  Left  attempted  : {p2L.get("phase2_attempted", False)}',
        f'  Left  valid      : {p2L.get("p2_valid", "—")}',
        f'  Right attempted  : {p2R.get("phase2_attempted", False)}',
        f'  Right valid      : {p2R.get("p2_valid", "—")}',
        '',
        '── Label Namespace Reminder ──────────────────',
        '  seg-spine 43 = TP-Left  (costal process)',
        '  seg-spine 44 = TP-Right (costal process)',
        '  seg-spine 26 = Sacrum (fallback)',
        '  TSS 43 = L3 vertebra  ← different meaning!',
        '  TSS 44 = L4 vertebra  ← different meaning!',
        '  TSS 50 = Sacrum (preferred)',
        '',
    ]
    if result is None:
        right_col += ['── Classifier ────────────────────────────────',
                      '  (run 04_detect_lstv.py first)']
    else:
        ct = result.get('castellvi_type') or 'None'
        right_col += [
            '── Classifier Output ─────────────────────────',
            f'  Castellvi  : {ct}',
            f'  LSTV       : {"YES  !" if result.get("lstv_detected") else "No"}',
            '',
            f'  Left  → {cls_L}',
            f'  Right → {cls_R}',
        ]
        if result.get('errors'):
            right_col += ['', '  Errors:'] + [f'    {e}' for e in result['errors'][:4]]

    ax.text(0.01, 0.97, '\n'.join(left_col),  transform=ax.transAxes, va='top', ha='left',
            fontsize=8.5, family='monospace', color='white', linespacing=1.32)
    ax.text(0.50, 0.97, '\n'.join(right_col), transform=ax.transAxes, va='top', ha='left',
            fontsize=8.5, family='monospace', color='white', linespacing=1.32)
    ax.set_title('Classification Summary', fontsize=11, color='white')


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def find_t2w(nifti_dir, study_id, acq):
    d = nifti_dir / study_id
    if not d.exists(): return None
    for sd in sorted(d.iterdir()):
        p = sd / f"sub-{study_id}_acq-{acq}_T2w.nii.gz"
        if p.exists(): return p
    return None


# ============================================================================
# CORE VISUALIZER
# ============================================================================

def visualize_study(study_id, spineps_dir, totalspine_dir, registered_dir,
                    nifti_dir, output_dir, result=None, uncertainty_row=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{study_id}_lstv_overlay.png"

    def _load(path, label):
        if not path.exists():
            logger.warning(f"  Missing: {path.name}"); return None, None
        try:
            return load_canonical(path)
        except Exception as e:
            logger.warning(f"  Cannot load {label}: {e}"); return None, None

    # seg-spine_msk: subregion labels — TP source (43/44) + sacrum fallback (26)
    sag_sp,     sag_sp_nii = _load(
        spineps_dir/'segmentations'/study_id/f"{study_id}_seg-spine_msk.nii.gz",
        'SPINEPS seg-spine_msk (subregion/TP source)')
    # seg-vert_msk: VERIDAH instance labels — TV identification only
    sag_vert,   _          = _load(
        spineps_dir/'segmentations'/study_id/f"{study_id}_seg-vert_msk.nii.gz",
        'SPINEPS seg-vert_msk (VERIDAH instance)')
    # TSS sagittal: 41-45=L1-L5 vertebral bodies, 50=sacrum (TSS 43=L3, 44=L4 ≠ SPINEPS TP)
    sag_tss,    _          = _load(
        totalspine_dir/study_id/'sagittal'/f"{study_id}_sagittal_labeled.nii.gz",
        'TotalSpineSeg sagittal labeled')
    ax_tss,     _          = _load(
        totalspine_dir/study_id/'axial'/f"{study_id}_axial_labeled.nii.gz",
        'TotalSpineSeg axial labeled')
    # Registered SPINEPS: seg-spine labels in axial T2w space (43=TP-L, 44=TP-R)
    ax_spineps, _          = _load(
        registered_dir/study_id/f"{study_id}_spineps_reg.nii.gz",
        'Registered SPINEPS (seg-spine in axial space)')

    sag_bg = None
    sag_path = find_t2w(nifti_dir, study_id, 'sag')
    if sag_path: sag_bg, _ = _load(sag_path, 'sag T2w')
    ax_t2w = None
    ax_path = find_t2w(nifti_dir, study_id, 'ax')
    if ax_path: ax_t2w, _ = _load(ax_path, 'axial T2w')

    if sag_sp is None:   logger.error("  No seg-spine_msk — skipping"); return None
    if sag_vert is None: logger.error("  No seg-vert_msk — skipping");  return None

    sag_sp   = sag_sp.astype(int)
    sag_vert = sag_vert.astype(int)
    if sag_bg is None:         sag_bg     = sag_sp.astype(float)
    if sag_tss is not None:    sag_tss    = sag_tss.astype(int)
    if ax_tss is not None:     ax_tss     = ax_tss.astype(int)
    if ax_spineps is not None: ax_spineps = ax_spineps.astype(int)

    vox_mm = voxel_size_mm(sag_sp_nii)

    # ── Cross-validation ─────────────────────────────────────────────────────
    _tss_for_xv = sag_tss if sag_tss is not None else np.zeros((2, 2, 2), int)
    xv = compute_cross_validation(sag_sp, sag_vert, _tss_for_xv, vox_mm)
    for w in xv.get('warnings', []):
        logger.warning(f"  XVAL: {w}")

    # ── TV from VERIDAH seg-vert_msk ─────────────────────────────────────────
    unique_vert = set(np.unique(sag_vert).tolist())
    tv_label, tv_name = None, None
    for cand in LUMBAR_LABELS:
        if cand in unique_vert:
            tv_label = cand; tv_name = VERIDAH_NAMES.get(cand, str(cand)); break
    if tv_label is None:
        logger.error("  No lumbar VERIDAH labels — skipping"); return None
    z_range = get_tv_z_range(sag_vert, tv_label)
    if z_range is None: return None
    z_min_tv, z_max_tv = z_range
    z_tv_mid = (z_min_tv + z_max_tv) // 2

    # ── Sacrum mask: prefer TSS 50, fall back to SPINEPS 26 ──────────────────
    if sag_tss is not None and (sag_tss == TSS_SACRUM_LABEL).any():
        sac_mask_sag = (sag_tss == TSS_SACRUM_LABEL)
        logger.info("  Sacrum: TSS label 50")
    else:
        sac_mask_sag = (sag_sp == SPINEPS_SACRUM)
        logger.warning("  Sacrum: falling back to SPINEPS label 26")

    # ── TP isolation from seg-spine_msk — NOT from TSS (TSS 43/44 = L3/L4 bodies!) ──
    def _isolate(tp_label):
        at_tv = isolate_tp_at_tv(sag_sp, tp_label, z_min_tv, z_max_tv)
        return inferiormost_tp_cc(at_tv, sac_mask_sag if sac_mask_sag.any() else None)

    tp_L = _isolate(TP_LEFT_LABEL)
    tp_R = _isolate(TP_RIGHT_LABEL)
    dist_L, tp_vox_L, _ = min_dist_3d(tp_L, sac_mask_sag, vox_mm)
    dist_R, tp_vox_R, _ = min_dist_3d(tp_R, sac_mask_sag, vox_mm)
    x_left,  span_L = best_x_for_tp_height(tp_L, vox_mm[2])
    x_right, span_R = best_x_for_tp_height(tp_R, vox_mm[2])
    x_mid           = sag_bg.shape[0] // 2
    z_md_L = int(tp_vox_L[2]) if tp_vox_L is not None else z_tv_mid
    z_md_R = int(tp_vox_R[2]) if tp_vox_R is not None else z_tv_mid
    z_md   = z_md_L if dist_L <= dist_R else z_md_R
    z_row0 = (tp_blob_z_midpoint(tp_L) + tp_blob_z_midpoint(tp_R)) // 2

    # ── Phase 2 info from detection JSON ─────────────────────────────────────
    midpt_L = midpt_R = cls_L = cls_R = feat_L = feat_R = None
    cls_L = cls_R = 'N/A'
    if result:
        ld, rd = result.get('left', {}), result.get('right', {})
        cls_L  = ld.get('classification', 'N/A')
        cls_R  = rd.get('classification', 'N/A')
        p2l    = ld.get('phase2') or {}
        p2r    = rd.get('phase2') or {}
        if p2l.get('midpoint_vox'): midpt_L = p2l['midpoint_vox']; feat_L = p2l.get('p2_features')
        if p2r.get('midpoint_vox'): midpt_R = p2r['midpoint_vox']; feat_R = p2r.get('p2_features')

    # ── Best axial z ─────────────────────────────────────────────────────────
    z_sac_ax = _best_z_for_sacrum(ax_tss)
    z_tp_ax  = _best_z_for_tp(ax_spineps)
    if z_sac_ax is not None:  best_z_ax = z_sac_ax
    elif midpt_L is not None: best_z_ax = int(midpt_L[2])
    elif midpt_R is not None: best_z_ax = int(midpt_R[2])
    elif z_tp_ax is not None: best_z_ax = z_tp_ax
    elif ax_t2w is not None:  best_z_ax = ax_t2w.shape[2] // 2
    else:                      best_z_ax = 0

    logger.info(f"  TV={tv_name}  dL={dist_L:.1f}  dR={dist_R:.1f}  "
                f"sL={span_L:.1f}  sR={span_R:.1f}  ax_z={best_z_ax}"
                f"(sac={z_sac_ax}, tp={z_tp_ax})")
    logger.info(f"  XVAL: sacrum_dice={xv.get('sacrum_dice')}  "
                f"l5_dist={xv.get('l5_centroid_dist_mm')}mm  "
                f"tp_ok={xv.get('tp_in_spine_mask')}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(21, 36))
    fig.patch.set_facecolor(BG_DARK)
    gs = gridspec.GridSpec(
        5, 3, figure=fig,
        height_ratios=[0.65, 1, 1, 1, 0.62],
        hspace=0.36, wspace=0.12,
        top=0.975, bottom=0.005, left=0.02, right=0.98,
    )
    def _ax(r, c, **kw):
        a = fig.add_subplot(gs[r, c], **kw); a.set_facecolor(BG_DARK); return a

    ax00 = _ax(0, 0); ax01 = _ax(0, 1); ax02 = _ax(0, 2)
    ax10 = _ax(1, 0); ax11 = _ax(1, 1); ax12 = _ax(1, 2)
    ax20 = _ax(2, 0); ax21 = _ax(2, 1); ax22 = _ax(2, 2)
    ax30 = fig.add_subplot(gs[3, 0:2]); ax30.set_facecolor(BG_DARK)
    ax31 = fig.add_subplot(gs[3, 2]);   ax31.set_facecolor(BG_DARK)
    ax40 = fig.add_subplot(gs[4, 0:3]); ax40.set_facecolor(BG_MID)

    dice_str = f'{xv["sacrum_dice"]:.3f}' if xv.get('sacrum_dice') is not None else 'N/A'
    fig.suptitle(
        f"Study {study_id}   |   Castellvi: {(result or {}).get('castellvi_type') or 'N/A'}"
        f"   |   TV: {tv_name}   |   Phase1-Sag/Phase2-Ax"
        f"   |   Sacrum Dice SPINEPS↔TSS: {dice_str}",
        fontsize=12, color='white', y=0.983,
    )

    # Row 0
    _panel_ian_pan(ax00, study_id, uncertainty_row)
    _panel_spineps_overview(ax01, sag_bg, sag_vert, sag_sp,
                            tp_L, tp_R, x_mid, z_tv_mid, tv_name)
    _panel_tss_overview(ax02, sag_bg, sag_tss, x_mid, z_tv_mid, tv_name)

    # Row 1 — P1 height panels + TSS overview at TP midpoint z
    _panel_p1_height(ax10, sag_bg, tp_L, tp_R, sag_sp,
                     'Left',  x_left,  span_L, vox_mm[2], z_row0, tv_name)
    _panel_p1_height(ax11, sag_bg, tp_R, tp_L, sag_sp,
                     'Right', x_right, span_R, vox_mm[2], z_row0, tv_name)
    _panel_tss_overview(ax12, sag_bg, sag_tss, x_mid, z_row0, tv_name)

    # Row 2 — P1 proximity + cross-validation panel
    _panel_p1_proximity(ax20, sag_bg, tp_L, tp_R, sag_sp, sag_tss,
                        'Left',  x_left,  dist_L, vox_mm[2], z_md, tv_name)
    _panel_p1_proximity(ax21, sag_bg, tp_R, tp_L, sag_sp, sag_tss,
                        'Right', x_right, dist_R, vox_mm[2], z_md, tv_name)
    _panel_cross_validation(ax22, xv, sag_sp, sag_vert, sag_tss, x_mid, vox_mm[2])

    # Row 3 — P2 axial
    _panel_p2_axial_overview(ax30, ax_t2w, ax_spineps, ax_tss, best_z_ax)
    fig.canvas.draw()
    _panel_p2_bboxes_inset(ax31, fig, ax_t2w, midpt_L, midpt_R,
                            cls_L, cls_R, feat_L, feat_R)

    # Row 4 — summary
    _panel_summary_wide(ax40, study_id, result, tv_name,
                        span_L, span_R, dist_L, dist_R, uncertainty_row, xv)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  OK → {out_path}")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Two-Phase LSTV Overlay Visualizer — 5 rows + cross-validation')
    parser.add_argument('--spineps_dir',     required=True)
    parser.add_argument('--totalspine_dir',  required=True)
    parser.add_argument('--registered_dir',  required=True)
    parser.add_argument('--nifti_dir',       required=True)
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--uncertainty_csv', default=None)
    parser.add_argument('--valid_ids',       default=None)
    parser.add_argument('--top_n',           type=int, default=None)
    parser.add_argument('--rank_by',         default='l5_s1_confidence')
    parser.add_argument('--all',             action='store_true')
    parser.add_argument('--study_id',        default=None)
    parser.add_argument('--lstv_json',       default=None)
    args = parser.parse_args()

    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    registered_dir = Path(args.registered_dir)
    nifti_dir      = Path(args.nifti_dir)
    output_dir     = Path(args.output_dir)
    seg_root       = spineps_dir / 'segmentations'

    results_by_id = {}
    if args.lstv_json:
        p = Path(args.lstv_json)
        if p.exists():
            with open(p) as f:
                results_by_id = {str(r['study_id']): r for r in json.load(f)}
            logger.info(f"Loaded {len(results_by_id)} detection results")

    uncertainty_by_id = {}
    csv_path = Path(args.uncertainty_csv) if args.uncertainty_csv else None
    if csv_path and csv_path.exists():
        df_unc = pd.read_csv(csv_path)
        df_unc['study_id'] = df_unc['study_id'].astype(str)
        uncertainty_by_id  = {r['study_id']: r for r in df_unc.to_dict('records')}
        logger.info(f"Loaded Ian Pan uncertainty for {len(uncertainty_by_id)} studies")

    if args.study_id:
        study_ids = [args.study_id]
    elif args.all:
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        logger.info(f"ALL mode: {len(study_ids)} studies")
    else:
        if not args.uncertainty_csv or args.top_n is None:
            parser.error("--uncertainty_csv and --top_n required unless --all or --study_id")
        valid_ids = None
        if args.valid_ids:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
        study_ids = select_studies(csv_path, args.top_n, args.rank_by, valid_ids)
        study_ids = [s for s in study_ids if (seg_root / s).is_dir()]
        logger.info(f"Selective mode: {len(study_ids)} studies")

    errors = 0
    for sid in study_ids:
        logger.info(f"\n[{sid}]")
        try:
            visualize_study(
                sid, spineps_dir, totalspine_dir, registered_dir, nifti_dir, output_dir,
                result          = results_by_id.get(sid),
                uncertainty_row = uncertainty_by_id.get(sid),
            )
        except Exception as e:
            logger.error(f"  [{sid}] Failed: {e}")
            logger.debug(traceback.format_exc())
            errors += 1

    logger.info(f"\nDone. {len(study_ids)-errors}/{len(study_ids)} PNGs → {output_dir}")


if __name__ == '__main__':
    main()
