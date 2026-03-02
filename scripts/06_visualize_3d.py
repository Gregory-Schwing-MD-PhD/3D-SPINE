#!/usr/bin/env python3
"""
06_visualize_3d.py — LSTV 3D Visualization (v5.2)
==================================================
v5.2 CHANGES vs v5.1:
  1. PAPER-ACCURATE DORSAL ANGLE OVERLAYS (build_angle_overlays_3d):
       Lines project POSTERIORLY outside the spine at a fixed dorsal Y plane,
       matching Figures 1–4 of Seilanian Toosi 2025.  Each angle group sits at
       a distinct dorsal depth so they never overlap:
         δ  (white/red)  — furthest dorsal, width-5 lines, 15pt label — most prominent
         D  (orange)     — intermediate dorsal depth
         D1 (cyan)       — slightly further than D
         A  (yellow)     — sacral level
         B  (red)        — full spine height
         C  (magenta)    — posterior body lines with yellow vertical ref
       Each has: tilted endplate line, dashed vertical connector, arc, bold label.
       δ ≤ 8.5°  → lines/arc/label turn red, label appends "⚠ Type2 LSTV".
       C ≤ 35.5° → label turns red, appends "⚠".

  2. BUG FIXES vs v5.1:
       - Arc: was using wrong Rodrigues formula; fixed to proper spherical linear
         interpolation (slerp) so arc endpoints exactly match the two input lines.
       - Endplate line: Y was incorrectly perturbed by tilt; line now stays at
         fixed y_dorsal and tilts only in Z (matching paper figures).
       - Docstring / version tag updated.

  3. COLOR CONFLICT FIX (from v5.1, retained):
       TP-Left  = #00ccff (cyan)    — was #ff3333 (red, clashed with TSS sacrum)
       TP-Right = #ff6600 (orange)
       TSS Sacrum = #ff8c00 — now unambiguous

  4. ANGLE PANEL (right sidebar, from v5.1, retained):
       δ, C, A, B, D, D1 with color-coded threshold badges.

  CLI: batch mode with --rank_by lstv / --all / --study_id (underscore args,
       matching the v5 batch runner). JSON format: list of dicts each with
       a 'study_id' key, or dict keyed by study_id.

OUTPUT
------
  {output_dir}/{study_id}_lstv_3d.html  — self-contained Plotly HTML

USAGE — single study
-----
  python 06_visualize_3d.py \\
      --study_id SUB-001 \\
      --spineps_dir /data/spineps \\
      --totalspine_dir /data/totalspine \\
      --lstv_json /data/results/lstv_results.json \\
      --output_dir /data/visualizations

USAGE — batch (top 10 pathologic + 1 normal)
-----
  python 06_visualize_3d.py \\
      --spineps_dir /data/spineps \\
      --totalspine_dir /data/totalspine \\
      --lstv_json /data/results/lstv_results.json \\
      --output_dir /data/visualizations \\
      --rank_by lstv --top_n 10 --top_normal 1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as ndizoom

logger = logging.getLogger(__name__)

# ── Label constants ───────────────────────────────────────────────────────────
SP_TP_L    = 43
SP_TP_R    = 44
SP_SACRUM  = 26
SP_ARCUS   = 41
SP_SPINOUS = 42
SP_SAL     = 45
SP_SAR     = 46
SP_CORPUS  = 49
SP_CORD    = 60
SP_CANAL   = 61

TSS_SACRUM    = 50
TSS_L4L5_DISC = 95
TSS_L5S1_DISC = 100
TSS_LUMBAR    = {41:'L1', 42:'L2', 43:'L3', 44:'L4', 45:'L5'}

VD_L1=20; VD_L2=21; VD_L3=22; VD_L4=23; VD_L5=24; VD_L6=25; VD_SAC=26

# ── Morphology thresholds ─────────────────────────────────────────────────────
TP_HEIGHT_MM    = 19.0
CONTACT_DIST_MM = 2.0
EXPECTED_LUMBAR = 5

# ── RENDER CONFIG ─────────────────────────────────────────────────────────────
SP_RENDER: List[Tuple[int, str, str, float]] = [
    (SP_TP_L,    'TP Left',      '#00ccff', 0.95),   # cyan  — v5.1 fix
    (SP_TP_R,    'TP Right',     '#ff6600', 0.95),   # orange — v5.1 fix
    (SP_SACRUM,  'SP Sacrum',    '#b06000', 0.50),
    (SP_CORPUS,  'SP Corpus',    '#c0c0ff', 0.30),
    (SP_ARCUS,   'SP Arcus',     '#a0a0dd', 0.25),
    (SP_SPINOUS, 'SP Spinous',   '#8888cc', 0.25),
    (SP_SAL,     'SP SAL',       '#88aacc', 0.20),
    (SP_SAR,     'SP SAR',       '#88ccaa', 0.20),
    (SP_CORD,    'Cord',         '#ffff88', 0.45),
    (SP_CANAL,   'Canal',        '#ffffcc', 0.18),
]

TSS_RENDER: List[Tuple[int, str, str, float]] = [
    (TSS_SACRUM,    'TSS Sacrum',    '#ff8c00', 0.72),
    (TSS_L4L5_DISC, 'L4-L5 Disc',   '#44dd44', 0.60),
    (TSS_L5S1_DISC, 'L5-S1 Disc',   '#22bb22', 0.60),
    (41,            'TSS L1',        '#dde0ff', 0.20),
    (42,            'TSS L2',        '#ccd0ff', 0.20),
    (43,            'TSS L3',        '#bbc0ff', 0.20),
    (44,            'TSS L4',        '#aab0ff', 0.20),
    (45,            'TSS L5',        '#9090ee', 0.22),
]

VD_RENDER: List[Tuple[int, str, str, float]] = [
    (VD_L1, 'VD L1', '#e8e8ff', 0.18),
    (VD_L2, 'VD L2', '#d8d8ff', 0.18),
    (VD_L3, 'VD L3', '#c8c8ff', 0.18),
    (VD_L4, 'VD L4', '#b8b8ee', 0.18),
    (VD_L5, 'VD L5', '#9898dd', 0.20),
    (VD_L6, 'VD L6', '#7878cc', 0.22),
    (VD_SAC,'VD Sac','#cc8800', 0.40),
]

PHENOTYPE_COLORS = {
    'sacralization':               '#ff4444',
    'lumbarization':               '#4488ff',
    'transitional_indeterminate':  '#ff9900',
    'normal':                      '#44cc44',
}

# ── Angle thresholds (Seilanian Toosi 2025) ───────────────────────────────────
DELTA_THRESHOLD = 8.5
C_THRESHOLD     = 35.5
A_UPPER_NORMAL  = 41.0
D_LOWER_NORMAL  = 13.5

# ── Dorsal projection geometry constants ──────────────────────────────────────
DORSAL_OFFSET   = 20.0   # vox posterior to spine posterior surface
LINE_HALF_LEN   = 38.0   # half-length of each endplate line (vox)
ARC_R_SMALL     = 11.0   # arc radius for D/D1/delta
ARC_R_LARGE     = 17.0   # arc radius for A/B/C
LINE_WIDTH_PRI  = 4
LINE_WIDTH_SEC  = 3
DOT_SIZE        = 8


# ══════════════════════════════════════════════════════════════════════════════
# NIfTI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_canonical_iso(path: Path, target_mm: float = 1.0) -> np.ndarray:
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 4:
        data = data[..., 0]
    vox  = np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))
    factors = (vox / target_mm).tolist()
    return ndizoom(data.astype(np.int32), factors,
                   order=0, mode='nearest', prefilter=False).astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# MESH EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_mesh(vol: np.ndarray, label: int,
                   level: float = 0.5, step_size: int = 2,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        logger.error("scikit-image not installed")
        return None
    mask = (vol == label).astype(np.uint8)
    if mask.sum() < 27:
        return None
    try:
        verts, faces, _, _ = marching_cubes(mask, level=level, step_size=step_size)
        return verts, faces, None
    except Exception:
        return None


def _mesh_to_plotly(verts: np.ndarray, faces: np.ndarray,
                     color: str, opacity: float, name: str) -> dict:
    return dict(
        type        = 'mesh3d',
        x           = verts[:, 0].tolist(),
        y           = verts[:, 1].tolist(),
        z           = verts[:, 2].tolist(),
        i           = faces[:, 0].tolist(),
        j           = faces[:, 1].tolist(),
        k           = faces[:, 2].tolist(),
        color       = color,
        opacity     = opacity,
        name        = name,
        flatshading = False,
        lighting    = dict(ambient=0.6, diffuse=0.8, specular=0.3,
                           roughness=0.5, fresnel=0.1),
        lightposition = dict(x=100, y=200, z=300),
        showscale   = False,
        hovertemplate = f'<b>{name}</b><extra></extra>',
    )


# ══════════════════════════════════════════════════════════════════════════════
# BASIC GEOMETRY / TP RULER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _centroid(vol: np.ndarray, label: int) -> Optional[np.ndarray]:
    mask = (vol == label)
    if not mask.any():
        return None
    coords = np.array(np.where(mask), dtype=float)
    return coords.mean(axis=1)


def _tp_sacrum_gap_mm(sp_vol: np.ndarray,
                       tp_label: int,
                       sac_label: int = SP_SACRUM) -> Optional[float]:
    tp_mask  = (sp_vol == tp_label)
    sac_mask = (sp_vol == sac_label)
    if not tp_mask.any() or not sac_mask.any():
        return None
    tp_pts  = np.array(np.where(tp_mask),  dtype=float).T
    sac_pts = np.array(np.where(sac_mask), dtype=float).T
    if len(tp_pts)  > 5000: tp_pts  = tp_pts [np.random.choice(len(tp_pts),  5000, False)]
    if len(sac_pts) > 5000: sac_pts = sac_pts[np.random.choice(len(sac_pts), 5000, False)]
    from scipy.spatial import cKDTree
    tree = cKDTree(sac_pts)
    dists, _ = tree.query(tp_pts, k=1)
    return float(dists.min())


def _ruler_trace(p0: np.ndarray, p1: np.ndarray,
                  color: str = '#ffffff', name: str = '',
                  width: int = 3) -> dict:
    return dict(
        type   = 'scatter3d',
        x      = [float(p0[0]), float(p1[0])],
        y      = [float(p0[1]), float(p1[1])],
        z      = [float(p0[2]), float(p1[2])],
        mode   = 'lines+markers',
        line   = dict(color=color, width=width),
        marker = dict(size=4, color=color),
        name   = name,
        hovertemplate = f'<b>{name}</b><extra></extra>',
        showlegend = True,
    )


def _annotation_trace(pos: np.ndarray, text: str, color: str = '#ffffff') -> dict:
    return dict(
        type     = 'scatter3d',
        x        = [float(pos[0])],
        y        = [float(pos[1])],
        z        = [float(pos[2])],
        mode     = 'markers+text',
        marker   = dict(size=6, color=color, symbol='circle'),
        text     = [text],
        textposition = 'top center',
        textfont = dict(color=color, size=11),
        name     = text,
        hovertemplate = f'<b>{text}</b><extra></extra>',
        showlegend = False,
    )


def _plane_trace(center: np.ndarray, normal: np.ndarray,
                  color: str, opacity: float, name: str,
                  extent: float = 25.0) -> dict:
    n   = normal / (np.linalg.norm(normal) + 1e-9)
    ref = np.array([0., 1., 0.]) if abs(n[1]) < 0.9 else np.array([1., 0., 0.])
    t1  = np.cross(n, ref); t1 /= (np.linalg.norm(t1) + 1e-9)
    t2  = np.cross(n, t1);  t2 /= (np.linalg.norm(t2) + 1e-9)
    corners = [center + extent * (s1 * t1 + s2 * t2)
               for s1, s2 in [(-1,-1),(1,-1),(1,1),(-1,1)]]
    return dict(
        type    = 'mesh3d',
        x       = [float(c[0]) for c in corners],
        y       = [float(c[1]) for c in corners],
        z       = [float(c[2]) for c in corners],
        i       = [0, 0], j = [1, 2], k = [2, 3],
        color   = color, opacity = opacity, name = name,
        flatshading = True,
        hovertemplate = f'<b>{name}</b><extra></extra>',
        showscale = False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ANGLE PANEL HTML  (sidebar)
# ══════════════════════════════════════════════════════════════════════════════

def _angle_color(angle: Optional[float], threshold: float,
                  direction: str = 'below') -> str:
    if angle is None:
        return '#555555'
    triggered = (angle <= threshold if direction == 'below' else angle >= threshold)
    return '#cc2222' if triggered else '#226622'


def _fmt_angle(v: Optional[float]) -> str:
    return f'{v:.1f}°' if v is not None else 'N/A'


def build_angle_panel_html(angles: Optional[dict]) -> str:
    if angles is None or not angles.get('angles_available', False):
        return '<div style="color:#888;padding:12px">Angles not available</div>'

    a  = angles.get('a_angle_deg')
    b  = angles.get('b_angle_deg')
    c  = angles.get('c_angle_deg')
    d  = angles.get('d_angle_deg')
    d1 = angles.get('d1_angle_deg')
    de = angles.get('delta_angle_deg')

    rows = [
        ('δ (delta)', de, DELTA_THRESHOLD, 'below',
         '≤8.5° → Type 2 LSTV (sens 92%, spec 88%)'),
        ('C', c, C_THRESHOLD, 'below',
         '≤35.5° → LSTV (sens 72%, spec 58%)'),
        ('A', a, A_UPPER_NORMAL, 'above',
         f'>{A_UPPER_NORMAL}° = elevated (normal ~37°)'),
        ('D', d, D_LOWER_NORMAL, 'below',
         f'<{D_LOWER_NORMAL}° = decreased (normal ~26°)'),
        ('D1', d1, None, None, 'TV-1 vs TV superior endplates'),
        ('B',  b,  None, None, 'L3 vs sacral surface'),
    ]

    html = '''<table style="width:100%;border-collapse:collapse;
        font-family:monospace;font-size:12px;color:#eee;">
      <thead><tr style="background:#333">
        <th style="padding:4px 8px;text-align:left">Angle</th>
        <th style="padding:4px 8px;text-align:right">Value</th>
        <th style="padding:4px 8px;text-align:left;font-weight:normal;font-size:11px">Criterion</th>
      </tr></thead><tbody>'''

    for label, val, thr, direction, note in rows:
        bg   = _angle_color(val, thr, direction) if thr is not None else (
               '#335533' if val is not None else '#555555')
        flag = ''
        if val is not None and thr is not None:
            triggered = (val <= thr if direction == 'below' else val >= thr)
            flag = ' ⚠' if triggered else ' ✓'
        html += (f'<tr style="background:{bg};border-bottom:1px solid #444">'
                 f'<td style="padding:5px 8px;font-weight:bold">{label}</td>'
                 f'<td style="padding:5px 8px;text-align:right">{_fmt_angle(val)}{flag}</td>'
                 f'<td style="padding:5px 8px;font-size:10px;color:#ccc">{note}</td></tr>')

    html += '</tbody></table>'

    tc = angles.get('tp_concordance')
    if tc:
        lf = tc.get('left_in_bounds');  rf = tc.get('right_in_bounds')
        lstr = '✓' if lf is True else ('✗' if lf is False else '—')
        rstr = '✓' if rf is True else ('✗' if rf is False else '—')
        tc_color = '#cc2222' if (lf is False or rf is False) else '#226622'
        html += (f'<div style="margin-top:6px;padding:4px 8px;background:{tc_color};'
                 f'font-size:11px;">TP Concordance  L:{lstr}  R:{rstr}</div>')
    return html


# ══════════════════════════════════════════════════════════════════════════════
# ENDPLATE GEOMETRY HELPERS  (v5.2: bugs fixed)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_endplate_normal_and_center(
        vol: np.ndarray, label: int,
        surface: str = 'superior', slab_vox: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    PCA plane fit to the superior or inferior slab of a segmented vertebra.
    Returns (center_xyz, normal_unit_vec) with normal pointing cranially (+Z).
    """
    mask = (vol == label)
    if not mask.any():
        return None, None
    coords = np.array(np.where(mask), dtype=float).T   # (N, 3)
    z_all  = coords[:, 2]
    if surface == 'superior':
        slab = coords[z_all >= z_all.max() - slab_vox]
    else:
        slab = coords[z_all <= z_all.min() + slab_vox]
    if len(slab) < 6:
        slab = coords
    center = slab.mean(axis=0)
    cov    = np.cov((slab - center).T)
    try:
        _, vecs = np.linalg.eigh(cov)
        normal  = vecs[:, 0]          # smallest eigenvector = plane normal
    except Exception:
        normal = np.array([0., 0., 1.])
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal) + 1e-9
    return center, normal


def _spine_mid_x(sp_vol: np.ndarray, vert_vol: np.ndarray) -> float:
    """Mid ML (X) coordinate — used to anchor all overlays in the sagittal plane."""
    for vol in (vert_vol, sp_vol):
        mask = vol > 0
        if mask.any():
            return float(np.where(mask)[0].mean())
    return float(sp_vol.shape[0] / 2)


def _post_y_for_label(tss_vol: Optional[np.ndarray], tss_lbl: Optional[int],
                       vert_vol: np.ndarray, vd_lbl: Optional[int]) -> Optional[float]:
    """Return posterior (min-Y) surface of a label — used to set y_dorsal."""
    for vol, lbl in [(tss_vol, tss_lbl), (vert_vol, vd_lbl)]:
        if vol is None or lbl is None:
            continue
        mask = (vol == lbl)
        if mask.any():
            return float(np.where(mask)[1].min())
    return None


def _endplate_line(
        center:   np.ndarray,
        normal:   np.ndarray,
        x_mid:    float,
        y_dorsal: float,
        half_len: float = LINE_HALF_LEN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an endplate line in the dorsal projection plane.

    The line sits at fixed y = y_dorsal (posterior to the spine) and
    extends ±half_len in the Z direction, with a tilt derived from the
    endplate normal.  Y is held constant so the line is always visible
    from a lateral/posterior view and doesn't dive into or behind the mesh.

    Returns (anchor, p0, p1):
      anchor — (x_mid, y_dorsal, z_center) : where the angle arc is drawn
      p0     — caudal endpoint
      p1     — cranial endpoint
    """
    z_c    = float(center[2])
    anchor = np.array([x_mid, y_dorsal, z_c])

    ny = float(normal[1]); nz = float(normal[2])
    dz_tilt = -ny / (abs(nz) + 1e-4)
    dz_tilt = float(np.clip(dz_tilt, -1.5, 1.5))

    p0 = np.array([x_mid, y_dorsal, z_c - half_len + dz_tilt * half_len * 0.3])
    p1 = np.array([x_mid, y_dorsal, z_c + half_len + dz_tilt * half_len * 0.3])
    return anchor, p0, p1


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE 3D TRACE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _line3d_trace(p0: np.ndarray, p1: np.ndarray,
                   color: str, width: int = 3, name: str = '',
                   dash: str = 'solid', show_legend: bool = True) -> dict:
    return dict(
        type  = 'scatter3d',
        x     = [float(p0[0]), float(p1[0])],
        y     = [float(p0[1]), float(p1[1])],
        z     = [float(p0[2]), float(p1[2])],
        mode  = 'lines',
        line  = dict(color=color, width=width, dash=dash),
        name  = name,
        hovertemplate = f'<b>{name}</b><extra></extra>',
        showlegend = show_legend,
    )


def _arc3d_trace(xs: List[float], ys: List[float], zs: List[float],
                  color: str, width: int = 3, name: str = '') -> dict:
    return dict(
        type  = 'scatter3d',
        x = xs, y = ys, z = zs,
        mode  = 'lines',
        line  = dict(color=color, width=width),
        name  = name,
        hovertemplate = f'<b>{name}</b><extra></extra>',
        showlegend = False,
    )


def _label3d_trace(pos: np.ndarray, text: str, color: str, size: int = 13) -> dict:
    return dict(
        type  = 'scatter3d',
        x     = [float(pos[0])], y = [float(pos[1])], z = [float(pos[2])],
        mode  = 'text',
        text  = [text],
        textfont = dict(color=color, size=size, family='Arial Black'),
        name  = text,
        hovertemplate = f'<b>{text}</b><extra></extra>',
        showlegend = False,
    )


def _dot3d_trace(pos: np.ndarray, color: str, size: int = 7) -> dict:
    return dict(
        type   = 'scatter3d',
        x      = [float(pos[0])], y = [float(pos[1])], z = [float(pos[2])],
        mode   = 'markers',
        marker = dict(size=size, color=color),
        showlegend = False,
        hoverinfo  = 'skip',
    )


def _arc_slerp(
        apex:   np.ndarray,
        v1:     np.ndarray,
        v2:     np.ndarray,
        radius: float = 14.0,
        n_pts:  int   = 28,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Correct spherical linear interpolation arc between unit directions v1 and v2.

    Projects both vectors into the sagittal (YZ) plane so the arc stays
    in the plane of the paper figure regardless of ML tilt.

    FIX vs v5.1: old code used  sin(t*θ)*v2 + cos(t*θ)*v1  which is only
    correct at t=0 and gives the wrong endpoint at t=1.  Correct slerp is:
        p(t) = sin((1-t)*θ)/sin(θ) * v1  +  sin(t*θ)/sin(θ) * v2
    """
    def _sag_unit(v: np.ndarray) -> np.ndarray:
        s = np.array([0.0, float(v[1]), float(v[2])])
        n = np.linalg.norm(s)
        return s / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

    s1 = _sag_unit(v1)
    s2 = _sag_unit(v2)

    cos_t = float(np.clip(np.dot(s1, s2), -1.0, 1.0))
    theta = math.acos(cos_t)

    xs, ys, zs = [], [], []
    if theta < 1e-6:
        p = apex + radius * s1
        return ([float(p[0])], [float(p[1])], [float(p[2])])

    sin_t = math.sin(theta)
    for i in range(n_pts + 1):
        t  = i / n_pts
        w1 = math.sin((1.0 - t) * theta) / sin_t
        w2 = math.sin(t          * theta) / sin_t
        s  = w1 * s1 + w2 * s2
        sn = np.linalg.norm(s)
        if sn > 1e-9:
            s /= sn
        p = apex + radius * np.array([0.0, s[1], s[2]])
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
    return xs, ys, zs


# ══════════════════════════════════════════════════════════════════════════════
# PAPER-ACCURATE DORSAL ANGLE OVERLAYS  (v5.2)
# ══════════════════════════════════════════════════════════════════════════════

def build_angle_overlays_3d(
        sp_vol:      np.ndarray,
        tss_vol:     Optional[np.ndarray],
        vert_vol:    np.ndarray,
        tv_vd_label: int,
        angles:      dict,
) -> List[dict]:
    """
    Build paper-accurate 3D angle overlays (Seilanian Toosi 2025, Figs 1-4).

    All lines are drawn at a FIXED y_dorsal value (posterior to the spine),
    extending in Z with the endplate tilt.  Each angle group is placed at a
    distinct dorsal depth so overlays never obscure each other:

        δ  (deepest dorsal) — white or red
        D1 (next)           — cyan
        D  (next)           — orange
        C  (shared mid)     — magenta
        B  (near spine)     — red
        A  (near spine)     — yellow

    Arc is drawn via slerp so it exactly spans the two endplate lines.
    """
    traces: List[dict] = []
    if not angles.get('angles_available', False):
        return traces

    x_mid = _spine_mid_x(sp_vol, vert_vol)

    # Label mappings
    tv_tss  = {VD_L5: 45, VD_L4: 44, VD_L6: 45, VD_L3: 43}.get(tv_vd_label)
    tv1_vd  = {VD_L5: VD_L4, VD_L4: VD_L3, VD_L6: VD_L5}.get(tv_vd_label)
    tv1_tss = {VD_L4: 44, VD_L3: 43, VD_L5: 45}.get(tv1_vd) if tv1_vd else None

    def _cn(tss_lbl, vd_lbl, surface='superior'):
        if tss_lbl and tss_vol is not None:
            c, n = _fit_endplate_normal_and_center(tss_vol, tss_lbl, surface)
            if c is not None:
                return c, n
        if vd_lbl:
            c, n = _fit_endplate_normal_and_center(vert_vol, vd_lbl, surface)
            if c is not None:
                return c, n
        return None, None

    def _py(tss_lbl, vd_lbl):
        y = _post_y_for_label(tss_vol, tss_lbl, vert_vol, vd_lbl)
        return (y - DORSAL_OFFSET) if y is not None else (-DORSAL_OFFSET)

    # Gather endplate geometry
    c_sac,  n_sac  = _cn(TSS_SACRUM, VD_SAC, 'superior')
    c_tv,   n_tv   = _cn(tv_tss, tv_vd_label, 'superior')
    c_tv1,  n_tv1  = _cn(tv1_tss, tv1_vd, 'superior') if tv1_vd else (None, None)
    c_l3,   n_l3   = _cn(43, VD_L3, 'superior')

    # Determine shared Y anchor
    y_vals = [_py(TSS_SACRUM, VD_SAC)]
    if tv_tss:    y_vals.append(_py(tv_tss, tv_vd_label))
    if tv1_tss:   y_vals.append(_py(tv1_tss, tv1_vd))
    y_base = min(y_vals)

    Y = {
        'delta': y_base - 24.0,
        'D1':    y_base - 16.0,
        'D':     y_base - 8.0,
        'C':     y_base - 2.0,
        'B':     y_base,
        'A':     y_base + 2.0,
    }

    def _overlay(label: str, center_a, normal_a, center_b, normal_b,
                  color: str, val: Optional[float],
                  line_w: int, arc_r: float,
                  name_a: str, name_b: str,
                  flag_str: str = '',
                  show_vert_connector: bool = True) -> None:
        y_d = Y[label]
        anc_a, pa0, pa1 = _endplate_line(center_a, normal_a, x_mid, y_d)
        anc_b, pb0, pb1 = _endplate_line(center_b, normal_b, x_mid, y_d)

        if show_vert_connector:
            va = np.array([x_mid, y_d, float(center_a[2])])
            vb = np.array([x_mid, y_d, float(center_b[2])])
            traces.append(_line3d_trace(va, vb, '#888888', 2,
                                         f'{label}: connector', dash='dash',
                                         show_legend=False))

        traces.append(_line3d_trace(pa0, pa1, color, line_w, name_a))
        traces.append(_line3d_trace(pb0, pb1, color, LINE_WIDTH_SEC, name_b,
                                     show_legend=False))
        traces.append(_dot3d_trace(anc_a, color, DOT_SIZE))

        da = pa1 - pa0; da /= np.linalg.norm(da) + 1e-9
        db = pb1 - pb0; db /= np.linalg.norm(db) + 1e-9
        ax, ay, az = _arc_slerp(anc_a, da, db, arc_r)
        traces.append(_arc3d_trace(ax, ay, az, color, line_w, f'{label}-arc'))

        if val is not None:
            lbl_text = f'{label}={val:.1f}°{flag_str}'
            lbl_pos  = anc_a + np.array([0.0, -arc_r * 1.6, arc_r * 0.4])
            traces.append(_label3d_trace(lbl_pos, lbl_text, color,
                                          size=15 if label == 'δ' else 12))

    # A-ANGLE (yellow) — sacral superior surface vs vertical reference
    a_val = angles.get('a_angle_deg')
    if c_sac is not None and n_sac is not None and a_val is not None:
        col  = '#ffdd00'
        y_d  = Y['A']
        anc, p0, p1 = _endplate_line(c_sac, n_sac, x_mid, y_d, LINE_HALF_LEN * 1.1)
        vert_up   = anc + np.array([0., 0.,  LINE_HALF_LEN * 0.85])
        vert_down = anc + np.array([0., 0., -LINE_HALF_LEN * 0.45])
        traces.append(_line3d_trace(p0, p1, col, LINE_WIDTH_PRI, 'A: sacral surface'))
        traces.append(_line3d_trace(vert_down, vert_up, col, LINE_WIDTH_SEC,
                                     'A: vertical ref', dash='dash', show_legend=False))
        traces.append(_dot3d_trace(anc, col, DOT_SIZE))
        da = p1 - p0; da /= np.linalg.norm(da) + 1e-9
        dv = np.array([0., 0., 1.])
        ax, ay, az = _arc_slerp(anc, da, dv, ARC_R_LARGE)
        traces.append(_arc3d_trace(ax, ay, az, col, 3, 'A-arc'))
        lp = anc + np.array([0., -ARC_R_LARGE * 1.5, ARC_R_LARGE * 0.5])
        traces.append(_label3d_trace(lp, f'A={a_val:.1f}°', col, 12))

    # B-ANGLE (red) — L3 superior vs sacral superior
    b_val = angles.get('b_angle_deg')
    if (c_sac is not None and c_l3 is not None
            and n_sac is not None and n_l3 is not None and b_val is not None):
        _overlay('B', c_sac, n_sac, c_l3, n_l3,
                 '#ff4444', b_val, LINE_WIDTH_PRI, ARC_R_LARGE * 0.9,
                 'B: sacral ref', 'B: L3 endplate')

    # C-ANGLE (magenta) — posterior body lines, largest pair
    c_val = angles.get('c_angle_deg')
    if c_val is not None:
        col_c     = '#ff44ff'
        triggered = c_val <= C_THRESHOLD
        col_lbl   = '#ff0000' if triggered else col_c
        flag_c    = ' ⚠' if triggered else ''

        def _post_wall_dir(tss_lbl, vd_lbl):
            for vol, lbl in [(tss_vol, tss_lbl), (vert_vol, vd_lbl)]:
                if vol is None or lbl is None: continue
                mask = (vol == lbl)
                if not mask.any(): continue
                coords = np.array(np.where(mask), dtype=float).T
                y_min  = coords[:, 1].min()
                post   = coords[coords[:, 1] <= y_min + 3]
                if len(post) < 4: continue
                ctr = post.mean(axis=0)
                cov = np.cov((post - ctr).T)
                try:
                    _, vecs = np.linalg.eigh(cov)
                    d = vecs[:, 2]
                except Exception:
                    d = np.array([0., 0., 1.])
                if d[2] < 0: d = -d
                d /= np.linalg.norm(d) + 1e-9
                return ctr, d
            return None, None

        pairs = [
            (TSS_SACRUM, VD_SAC),
            (tv_tss, tv_vd_label),
            (tv1_tss, tv1_vd) if tv1_vd else (None, None),
        ]
        post_lines = [(c, d) for tl, vl in pairs
                      for c, d in [_post_wall_dir(tl, vl)] if c is not None]

        if len(post_lines) >= 2:
            best = (0, 1)
            best_cos = 1.0
            for i in range(len(post_lines)):
                for j in range(i+1, len(post_lines)):
                    cos_a = abs(float(np.dot(post_lines[i][1], post_lines[j][1])))
                    if cos_a < best_cos:
                        best_cos = cos_a; best = (i, j)

            c0, d0 = post_lines[best[0]]
            c1, d1 = post_lines[best[1]]
            y_C    = Y['C']
            z_mid  = float((c0[2] + c1[2]) / 2)
            anc_C  = np.array([x_mid, y_C, z_mid])

            vr_a = anc_C + np.array([0., 0.,  LINE_HALF_LEN * 0.8])
            vr_b = anc_C + np.array([0., 0., -LINE_HALF_LEN * 0.4])
            traces.append(_line3d_trace(vr_b, vr_a, '#ffdd00', LINE_WIDTH_SEC,
                                         'C: vert ref', dash='dash', show_legend=False))

            traces.append(_line3d_trace(
                anc_C - d0 * LINE_HALF_LEN * 0.8,
                anc_C + d0 * LINE_HALF_LEN * 0.8,
                col_c, LINE_WIDTH_PRI, 'C: post body 1'))
            traces.append(_line3d_trace(
                anc_C - d1 * LINE_HALF_LEN * 0.8,
                anc_C + d1 * LINE_HALF_LEN * 0.8,
                col_c, LINE_WIDTH_SEC, 'C: post body 2', show_legend=False))
            traces.append(_dot3d_trace(anc_C, col_c, DOT_SIZE))

            ax, ay, az = _arc_slerp(anc_C, d0, d1, ARC_R_LARGE)
            traces.append(_arc3d_trace(ax, ay, az, col_c, 3, 'C-arc'))
            lp = anc_C + np.array([0., -ARC_R_LARGE * 1.4, ARC_R_LARGE * 0.2])
            traces.append(_label3d_trace(lp, f'C={c_val:.1f}°{flag_c}', col_lbl, 13))

    # D-ANGLE (orange) — TV superior vs S1 superior
    d_val = angles.get('d_angle_deg')
    if d_val is not None and c_tv is not None and c_sac is not None:
        _overlay('D', c_tv, n_tv, c_sac, n_sac,
                 '#ff8800', d_val, LINE_WIDTH_PRI, ARC_R_SMALL,
                 'D: TV sup endplate', 'D: S1 sup endplate')

    # D1-ANGLE (cyan) — TV-1 superior vs TV superior
    d1_val = angles.get('d1_angle_deg')
    if d1_val is not None and c_tv1 is not None and c_tv is not None:
        _overlay('D1', c_tv1, n_tv1, c_tv, n_tv,
                 '#00ccff', d1_val, LINE_WIDTH_PRI, ARC_R_SMALL,
                 'D1: TV-1 endplate', 'D1: TV endplate ref')

    # DELTA (white/red) — most prominent, furthest dorsal
    de_val = angles.get('delta_angle_deg')
    if de_val is not None and c_tv is not None and c_tv1 is not None:
        triggered = de_val <= DELTA_THRESHOLD
        col_de    = '#ff2222' if triggered else '#ffffff'
        flag_de   = ' ⚠ Type2 LSTV' if triggered else ''

        _overlay('δ', c_tv, n_tv, c_tv1, n_tv1,
                 col_de, de_val, 5, ARC_R_SMALL * 1.4,
                 f'δ TV endplate{flag_de}', 'δ TV-1 endplate',
                 flag_str=flag_de)

    return traces


# ══════════════════════════════════════════════════════════════════════════════
# MAIN VISUALIZATION BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_lstv_visualization(
        study_id:    str,
        sp_vol:      np.ndarray,
        vert_vol:    np.ndarray,
        tss_vol:     Optional[np.ndarray],
        lstv_result: Dict[str, Any],
        show_sp:     bool = True,
        show_tss:    bool = True,
        show_vd:     bool = False,
        step_size:   int  = 2,
) -> str:
    traces: List[dict] = []
    sp_labels  = frozenset(np.unique(sp_vol).tolist())   - {0}
    vd_labels  = frozenset(np.unique(vert_vol).tolist()) - {0}
    tss_labels = (frozenset(np.unique(tss_vol).tolist()) - {0}
                  if tss_vol is not None else frozenset())

    # SPINEPS meshes
    if show_sp:
        for label, name, color, opacity in SP_RENDER:
            if label not in sp_labels: continue
            res = _extract_mesh(sp_vol, label, step_size=step_size)
            if res:
                traces.append(_mesh_to_plotly(res[0], res[1], color, opacity, name))

    # TSS meshes
    if show_tss and tss_vol is not None:
        for label, name, color, opacity in TSS_RENDER:
            if label not in tss_labels: continue
            res = _extract_mesh(tss_vol, label, step_size=step_size)
            if res:
                traces.append(_mesh_to_plotly(res[0], res[1], color, opacity, name))

    # VERIDAH meshes
    if show_vd:
        for label, name, color, opacity in VD_RENDER:
            if label not in vd_labels: continue
            res = _extract_mesh(vert_vol, label, step_size=step_size)
            if res:
                traces.append(_mesh_to_plotly(res[0], res[1], color, opacity, name))

    # TP height rulers
    for tp_label, side, ruler_color in [
            (SP_TP_L, 'L', '#00ccff'),
            (SP_TP_R, 'R', '#ff6600')]:
        if tp_label not in sp_labels: continue
        mask   = (sp_vol == tp_label)
        coords = np.array(np.where(mask), dtype=float)
        z_min  = float(coords[2].min()); z_max = float(coords[2].max())
        h_mm   = z_max - z_min + 1.0
        cx     = float(coords[0].mean()); cy = float(coords[1].mean())
        triggered = h_mm >= TP_HEIGHT_MM
        rc    = '#ff4444' if triggered else ruler_color
        flag  = f' ≥{TP_HEIGHT_MM}mm ✓' if triggered else f' <{TP_HEIGHT_MM}mm'
        p_lo  = np.array([cx, cy, z_min]); p_hi = np.array([cx, cy, z_max])
        traces.append(_ruler_trace(p_lo, p_hi, rc, f'TP-{side} height {h_mm:.1f}mm{flag}'))
        mid   = (p_lo + p_hi) / 2.0 + np.array([5, 0, 0])
        traces.append(_annotation_trace(mid, f'TP-{side} {h_mm:.1f}mm', rc))

    # TP–Sacrum gap rulers
    for tp_label, side, ruler_color in [
            (SP_TP_L, 'L', '#00ccff'),
            (SP_TP_R, 'R', '#ff6600')]:
        if tp_label not in sp_labels or SP_SACRUM not in sp_labels: continue
        gap   = _tp_sacrum_gap_mm(sp_vol, tp_label)
        if gap is None: continue
        tp_c  = _centroid(sp_vol, tp_label)
        sac_c = _centroid(sp_vol, SP_SACRUM)
        if tp_c is None or sac_c is None: continue
        contact = gap <= CONTACT_DIST_MM
        gc   = '#ff4444' if contact else '#aaaaff'
        flag = f' CONTACT ≤{CONTACT_DIST_MM}mm' if contact else f' gap {gap:.1f}mm'
        traces.append(_ruler_trace(tp_c, sac_c, gc, f'TP-{side}–Sacrum{flag}', width=2))

    # TV mid-plane
    tv_label  = lstv_result.get('tv_label_veridah')
    tv_name   = lstv_result.get('tv_name', 'TV')
    phenotype = lstv_result.get('lstv_phenotype', 'normal')
    plane_color = PHENOTYPE_COLORS.get(phenotype, '#888888')

    if tv_label and tv_label in vd_labels:
        c_tv = _centroid(vert_vol, tv_label)
        if c_tv is not None:
            traces.append(_plane_trace(c_tv, np.array([0., 0., 1.]),
                                        plane_color, 0.18, f'TV ({tv_name}) mid-plane'))
            traces.append(_annotation_trace(c_tv + np.array([0, 0, 8]),
                                             f'{tv_name} ({phenotype})', plane_color))

    # Dorsal angle overlays (v5.2)
    angles = lstv_result.get('vertebral_angles')
    if angles and isinstance(angles, dict) and angles.get('angles_available'):
        if tv_label is not None:
            traces.extend(build_angle_overlays_3d(
                sp_vol, tss_vol, vert_vol, tv_label, angles))

    # Layout
    probs  = lstv_result.get('probabilities') or {}
    p_sac  = probs.get('p_sacralization', 0.0)
    p_lumb = probs.get('p_lumbarization', 0.0)
    title_str = (f'{study_id} | {tv_name} | {phenotype} | '
                 f'P(sac)={p_sac:.0%} P(lumb)={p_lumb:.0%}')

    layout = dict(
        title         = dict(text=title_str, font=dict(color='#eeeeee', size=13)),
        paper_bgcolor = '#1a1a2e',
        plot_bgcolor  = '#1a1a2e',
        scene = dict(
            xaxis = dict(showgrid=False, zeroline=False, showticklabels=False,
                         backgroundcolor='#1a1a2e', title=''),
            yaxis = dict(showgrid=False, zeroline=False, showticklabels=False,
                         backgroundcolor='#1a1a2e', title=''),
            zaxis = dict(showgrid=True,  zeroline=False, showticklabels=False,
                         backgroundcolor='#1a1a2e', gridcolor='#333355', title=''),
            bgcolor     = '#1a1a2e',
            aspectmode  = 'data',
            camera      = dict(eye=dict(x=1.8, y=1.8, z=0.5),
                               up =dict(x=0,   y=0,   z=1)),
        ),
        legend = dict(bgcolor='rgba(20,20,40,0.85)', bordercolor='#444466',
                      borderwidth=1, font=dict(color='#cccccc', size=10),
                      x=0.01, y=0.99),
        margin = dict(l=0, r=0, t=40, b=0),
        height = 750,
    )

    angle_panel_html = build_angle_panel_html(
        angles if isinstance(angles, dict) else None)
    color_legend_html = _build_color_legend_html()

    return _render_html(study_id, traces, layout,
                         angle_panel_html, color_legend_html, lstv_result)


# ══════════════════════════════════════════════════════════════════════════════
# COLOR LEGEND
# ══════════════════════════════════════════════════════════════════════════════

def _build_color_legend_html() -> str:
    items = [
        ('#00ccff', 'TP Left (SP 43)'),
        ('#ff6600', 'TP Right (SP 44)'),
        ('#b06000', 'SP Sacrum (SP 26)'),
        ('#ff8c00', 'TSS Sacrum (TSS 50)'),
        ('#44dd44', 'L4-L5 Disc (TSS 95)'),
        ('#22bb22', 'L5-S1 Disc (TSS 100)'),
        ('#9090ee', 'TSS L5 (TSS 45)'),
        ('#ffdd00', 'A-angle'),
        ('#ff4444', 'B-angle / Sacralization'),
        ('#ff44ff', 'C-angle'),
        ('#ff8800', 'D-angle'),
        ('#00ccff', 'D1-angle'),
        ('#ffffff', 'δ normal'),
        ('#ff2222', 'δ ≤8.5° (Type2)'),
        ('#4488ff', 'Lumbarization'),
        ('#ff9900', 'Indeterminate'),
        ('#44cc44', 'Normal'),
    ]
    html = '<div style="font-family:monospace;font-size:11px;color:#ccc">'
    for color, label in items:
        html += (f'<div style="display:flex;align-items:center;margin:2px 0">'
                 f'<div style="width:14px;height:14px;background:{color};'
                 f'margin-right:6px;border-radius:2px;flex-shrink:0"></div>'
                 f'{label}</div>')
    html += '</div>'
    return html


# ══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

def _render_html(study_id: str,
                  traces: List[dict], layout: dict,
                  angle_panel_html: str, color_legend_html: str,
                  lstv_result: dict) -> str:
    import json as _json
    traces_json = _json.dumps(traces)
    layout_json = _json.dumps(layout)

    phenotype   = lstv_result.get('lstv_phenotype', 'N/A')
    confidence  = lstv_result.get('phenotype_confidence', '')
    tv_name     = lstv_result.get('tv_name', 'N/A')
    lc          = lstv_result.get('lumbar_count_consensus', 'N/A')
    count_note  = lstv_result.get('lumbar_count_note', '')
    castv       = (lstv_result.get('castellvi_result') or {}).get('castellvi_type', 'N/A')
    probs       = lstv_result.get('probabilities') or {}
    p_sac       = probs.get('p_sacralization', 0)
    p_lumb      = probs.get('p_lumbarization', 0)
    p_norm      = probs.get('p_normal', 0)
    phenotype_c = PHENOTYPE_COLORS.get(phenotype, '#888888')

    tc          = lstv_result.get('tp_concordance') or {}
    tc_l        = tc.get('left_in_bounds')
    tc_r        = tc.get('right_in_bounds')
    tc_l_str    = '✓' if tc_l is True else ('✗ FAIL' if tc_l is False else '—')
    tc_r_str    = '✓' if tc_r is True else ('✗ FAIL' if tc_r is False else '—')
    tc_color    = '#cc2222' if (tc_l is False or tc_r is False) else '#226622'

    sr          = lstv_result.get('surgical_relevance') or {}
    wlr         = sr.get('wrong_level_risk', 'N/A')
    wlr_pct     = sr.get('wrong_level_risk_pct', 0)
    wlr_colors  = {'low':'#226622','low-moderate':'#665500',
                   'moderate':'#996600','high':'#cc4400','critical':'#cc0000'}
    wlr_color   = wlr_colors.get(wlr, '#888888')
    bert_p      = sr.get('bertolotti_probability', 0)
    flags       = '\n'.join(f'• {f}' for f in (sr.get('surgical_flags') or []))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LSTV 3D — {study_id}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    background:#0e0e1a; color:#e0e0f0;
    font-family:'Courier New',monospace;
    display:flex; flex-direction:column; height:100vh; overflow:hidden;
  }}
  #header {{
    background:#15152a; border-bottom:1px solid #333366;
    padding:8px 16px; display:flex; align-items:center; gap:16px; flex-shrink:0;
  }}
  #header h1 {{ font-size:13px; color:#aaaadd; font-weight:normal; }}
  .badge {{
    padding:3px 8px; border-radius:4px; font-size:12px; font-weight:bold;
  }}
  #main {{ display:flex; flex:1; overflow:hidden; }}
  #plotDiv {{ flex:1; min-width:0; }}
  #sidebar {{
    width:285px; flex-shrink:0; background:#15152a;
    border-left:1px solid #333366; overflow-y:auto;
    display:flex; flex-direction:column;
  }}
  .sb {{ border-bottom:1px solid #2a2a4a; padding:10px 12px; }}
  .sb h3 {{
    font-size:10px; color:#8888bb; text-transform:uppercase;
    letter-spacing:1px; margin-bottom:6px;
  }}
  .sr {{ display:flex; justify-content:space-between; font-size:12px; padding:2px 0; color:#ccccdd; }}
  .sv {{ color:#ffffff; font-weight:bold; }}
  .pb {{ height:6px; border-radius:3px; margin:2px 0 4px; }}
</style>
</head>
<body>
<div id="header">
  <h1>LSTV v5.2 — {study_id}</h1>
  <span class="badge" style="background:{phenotype_c}">{phenotype} ({confidence})</span>
  <span class="badge" style="background:#333366">TV: {tv_name}</span>
  <span class="badge" style="background:#333366">L: {lc}</span>
  <span class="badge" style="background:{tc_color}">TP L:{tc_l_str} R:{tc_r_str}</span>
</div>
<div id="main">
  <div id="plotDiv"></div>
  <div id="sidebar">

    <div class="sb">
      <h3>Probability</h3>
      <div class="sr"><span>Sacralization</span><span class="sv">{p_sac:.1%}</span></div>
      <div class="pb" style="width:{p_sac*100:.0f}%;background:#ff4444"></div>
      <div class="sr"><span>Lumbarization</span><span class="sv">{p_lumb:.1%}</span></div>
      <div class="pb" style="width:{p_lumb*100:.0f}%;background:#4488ff"></div>
      <div class="sr"><span>Normal</span><span class="sv">{p_norm:.1%}</span></div>
      <div class="pb" style="width:{p_norm*100:.0f}%;background:#44cc44"></div>
    </div>

    <div class="sb">
      <h3>Vertebral Angles — Seilanian Toosi 2025</h3>
      {angle_panel_html}
    </div>

    <div class="sb">
      <h3>Surgical Risk</h3>
      <div class="sr">
        <span>Wrong-level risk</span>
        <span class="sv" style="color:{wlr_color}">{wlr} ({wlr_pct:.0%})</span>
      </div>
      <div class="sr"><span>Bertolotti P</span><span class="sv">{bert_p:.0%}</span></div>
      <div style="font-size:10px;color:#aaaacc;margin-top:6px;white-space:pre-wrap">{flags}</div>
    </div>

    <div class="sb">
      <h3>Color Legend</h3>
      {color_legend_html}
    </div>

    <div class="sb" style="font-size:11px;color:#888899">
      <div>Castellvi: {castv}</div>
      <div style="margin-top:4px">{count_note[:90] if count_note else ''}</div>
    </div>

  </div>
</div>
<script>
  var traces = {traces_json};
  var layout = {layout_json};
  Plotly.newPlot('plotDiv', traces, layout, {{
    responsive:true, displayModeBar:true,
    modeBarButtonsToRemove:['toImage'],
  }});
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# STUDY RANKING  (from v5 batch runner)
# ══════════════════════════════════════════════════════════════════════════════

def rank_studies(results: List[dict],
                 n_pathologic: int,
                 n_normal: int) -> Tuple[List[str], List[str]]:
    scored = sorted(
        ((r['study_id'], r.get('pathology_score') or 0) for r in results),
        key=lambda t: t[1], reverse=True,
    )
    pathologic_ids = {sid for sid, _ in scored[:n_pathologic]}
    pathologic     = [sid for sid, _ in scored[:n_pathologic]]
    result_by_id   = {r['study_id']: r for r in results}

    def _is_normal(sid):
        r = result_by_id.get(sid, {})
        if r.get('lstv_detected'): return False
        cnt = (r.get('lstv_morphometrics') or {}).get('lumbar_count_consensus')
        if cnt is not None and cnt != EXPECTED_LUMBAR: return False
        if (r.get('pathology_score') or 0) > 0: return False
        return True

    normal_pool = [(sid, sc) for sid, sc in reversed(scored)
                   if sid not in pathologic_ids and _is_normal(sid)]
    if len(normal_pool) < n_normal:
        extra = [(sid, sc) for sid, sc in reversed(scored)
                 if sid not in pathologic_ids and sid not in {s for s, _ in normal_pool}]
        normal_pool = (normal_pool + extra)[:n_normal]

    normal = [sid for sid, _ in normal_pool[:n_normal]]
    return pathologic, normal


# ══════════════════════════════════════════════════════════════════════════════
# CLI  (underscore args, batch-capable — matches v5 SLURM runner)
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description='LSTV 3D Visualization v5.2')
    parser.add_argument('--spineps_dir',    required=True, type=Path)
    parser.add_argument('--totalspine_dir', required=True, type=Path)
    parser.add_argument('--output_dir',     required=True, type=Path)
    parser.add_argument('--lstv_json',      default=None,  type=Path)
    parser.add_argument('--study_id',       default=None)
    parser.add_argument('--all',            action='store_true',
                        help='Render every study in spineps_dir/segmentations/')
    parser.add_argument('--rank_by',        default='lstv',
                        choices=['lstv', 'all'],
                        help='lstv = top_n by pathology score; all = same as --all')
    parser.add_argument('--top_n',          type=int, default=10,
                        help='Number of most-pathologic studies to render')
    parser.add_argument('--top_normal',     type=int, default=1,
                        help='Number of most-normal (score=0) studies to render')
    parser.add_argument('--smooth',         type=float, default=2.0)
    parser.add_argument('--no_tss',         action='store_true')
    parser.add_argument('--show_vd',        action='store_true')
    parser.add_argument('--step_size',      type=int, default=2)
    parser.add_argument('--log_level',      default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s  %(levelname)-7s  %(message)s')

    spineps_dir    = args.spineps_dir
    totalspine_dir = args.totalspine_dir
    output_dir     = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_root = spineps_dir / 'segmentations'

    # ── JSON loading (list-of-dicts or dict-keyed-by-study_id) ────────────────
    all_results:   List[dict]     = []
    result_by_id:  Dict[str, dict] = {}

    if args.lstv_json and args.lstv_json.exists():
        with open(args.lstv_json) as fh:
            raw = json.load(fh)
        if isinstance(raw, list):
            all_results = raw
        else:
            # dict keyed by study_id → convert to list format
            all_results = [{'study_id': k, **v} for k, v in raw.items()]
        result_by_id = {str(r['study_id']): r for r in all_results}
        logger.info(f'Loaded {len(all_results)} results from {args.lstv_json}')
    elif args.lstv_json:
        logger.warning(f'lstv_json not found: {args.lstv_json}')

    # ── Study selection ────────────────────────────────────────────────────────
    if args.study_id:
        study_ids = [args.study_id]
        logger.info(f'Single-study mode: {args.study_id}')

    elif args.all or args.rank_by == 'all':
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        logger.info(f'Rendering all {len(study_ids)} studies in {seg_root}')

    elif args.rank_by == 'lstv':
        if not all_results:
            parser.error('--rank_by lstv requires --lstv_json with valid data')
        pathologic, normal = rank_studies(all_results, args.top_n, args.top_normal)
        seen = set(pathologic)
        study_ids = pathologic + [s for s in normal if s not in seen]
        study_ids = [s for s in study_ids if (seg_root / s).is_dir()]
        logger.info(
            f'Ranked selection: {len(pathologic)} pathologic + {len(normal)} normal '
            f'= {len(study_ids)} studies')
        for sid in pathologic:
            sc = result_by_id.get(sid, {}).get('pathology_score', 0)
            logger.info(f'  pathologic: {sid}  score={sc:.1f}')
        for sid in normal:
            sc = result_by_id.get(sid, {}).get('pathology_score') or 0
            logger.info(f'  normal:     {sid}  score={sc:.1f}')

    else:
        parser.error("Use --study_id, --all, or --rank_by lstv")

    if not study_ids:
        logger.error('No studies to render — check paths and lstv_json')
        return 1

    # ── Per-study rendering ────────────────────────────────────────────────────
    ok = 0
    for sid in study_ids:
        logger.info(f"\n{'='*60}\n[{sid}] Rendering...")
        try:
            seg      = spineps_dir / 'segmentations' / sid
            sp_path  = seg / f'{sid}_seg-spine_msk.nii.gz'
            vt_path  = seg / f'{sid}_seg-vert_msk.nii.gz'
            tss_path = (totalspine_dir / sid / 'sagittal'
                        / f'{sid}_sagittal_labeled.nii.gz')

            if not sp_path.exists():
                logger.warning(f'  [{sid}] Missing seg-spine_msk, skipping')
                continue
            if not vt_path.exists():
                logger.warning(f'  [{sid}] Missing seg-vert_msk, skipping')
                continue

            sp_vol   = _load_canonical_iso(sp_path)
            vert_vol = _load_canonical_iso(vt_path)
            tss_vol  = None
            if not args.no_tss and tss_path.exists():
                try:
                    tss_vol = _load_canonical_iso(tss_path)
                except Exception as exc:
                    logger.warning(f'  [{sid}] TSS load failed: {exc}')

            lstv_result = result_by_id.get(sid, {
                'study_id':      sid,
                'lstv_detected': False,
                'lstv_reason':   [],
                'castellvi_type': None,
                'left': {}, 'right': {},
                'lstv_morphometrics': None,
                'pathology_score': 0,
            })

            html = build_lstv_visualization(
                study_id    = sid,
                sp_vol      = sp_vol,
                vert_vol    = vert_vol,
                tss_vol     = tss_vol,
                lstv_result = lstv_result,
                show_sp     = True,
                show_tss    = not args.no_tss,
                show_vd     = args.show_vd,
                step_size   = args.step_size,
            )

            out = output_dir / f'{sid}_lstv_3d.html'
            out.write_text(html, encoding='utf-8')
            logger.info(f'  [{sid}] Saved → {out}  ({out.stat().st_size / 1e6:.1f} MB)')
            ok += 1

        except Exception as exc:
            logger.error(f'  [{sid}] FAILED: {exc}')
            logger.debug(traceback.format_exc())

    logger.info(f'\nDone. {ok}/{len(study_ids)} HTMLs written → {output_dir}')
    return 0 if ok == len(study_ids) else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
