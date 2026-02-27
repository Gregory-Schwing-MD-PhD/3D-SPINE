#!/usr/bin/env python3
"""
06_visualize_3d.py — LSTV-Focused Interactive 3D Spine Viewer
==============================================================
Renders 3D interactive HTML for LSTV cases.

PHENOTYPE BANNER
----------------
  SACRALIZATION (red)            — L5/L4 incorporating into sacrum
  LUMBARIZATION (orange)         — S1/L6 acquiring lumbar morphology
  TRANSITIONAL INDETERMINATE (yellow) — Castellvi positive, ambiguous phenotype
  NORMAL VARIANT (green)         — no LSTV criteria met

NOTE: Castellvi type and phenotype are ORTHOGONAL — both are displayed
simultaneously if applicable. A lumbarized L6 may have Castellvi IIb
on its TP; this is shown as both "Castellvi: Type IIb" AND
"LUMBARIZATION" in the banner.

STUDY SELECTION MODES
---------------------
  --study_id ID               single study
  --all                       every study with SPINEPS segmentation
  --rank_by lstv              rank by LSTV pathology score (requires --lstv_json)
  --top_n N                   render N most-pathologic studies  (default 5)
  --top_normal N              render N most-normal studies       (default 2)
  --lstv_json PATH            required for --rank_by lstv

PATHOLOGY SCORE (lstv_engine.compute_lstv_pathology_score):
  Castellvi IV=5  III=4  II=3  I=1
  Phenotype high confidence    +3
  Phenotype moderate           +2
  Transitional indeterminate   +1
  Lumbar count anomaly         +2
  Disc below DHI < 50%         +2
  Disc below DHI < 70%         +1
  TV body sacral-like          +2
  Rib anomaly                  +1
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import binary_fill_holes, gaussian_filter, label as cc_label, zoom as ndizoom
from skimage.measure import marching_cubes

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lstv_engine import (
    ISO_MM, TP_HEIGHT_MM, CONTACT_DIST_MM, EXPECTED_LUMBAR,
    TSS_SACRUM, TSS_LUMBAR, TSS_THORACIC, TSS_DISCS,
    SP_TP_L, SP_TP_R, SP_SACRUM, SP_ARCUS, SP_SPINOUS, SP_CORPUS,
    SP_CORD, SP_CANAL, SP_SAL, SP_SAR,
    VD_L1, VD_L2, VD_L3, VD_L4, VD_L5, VD_L6, VD_SAC,
    VD_IVD_BASE, VD_EP_BASE,
    VERIDAH_NAMES, TV_SHAPE_LUMBAR, TV_SHAPE_SACRAL,
    DHI_REDUCED_PCT, DHI_MODERATE_PCT,
    compute_lstv_pathology_score,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-7s  %(message)s')
logger = logging.getLogger(__name__)

# ── Phenotype colours ─────────────────────────────────────────────────────────
PHENOTYPE_CONFIG = {
    'sacralization': {
        'color': '#ff2222', 'bg': '#3a0000', 'border': '#ff4444',
        'label': 'SACRALIZATION', 'emoji': '🔴',
    },
    'lumbarization': {
        'color': '#ff8c00', 'bg': '#2a1800', 'border': '#ffaa33',
        'label': 'LUMBARIZATION', 'emoji': '🟠',
    },
    'transitional_indeterminate': {
        'color': '#ffe033', 'bg': '#2a2200', 'border': '#ffe066',
        'label': 'TRANSITIONAL (INDETERMINATE)', 'emoji': '🟡',
    },
    # legacy key in case old json has 'transitional'
    'transitional': {
        'color': '#ffe033', 'bg': '#2a2200', 'border': '#ffe066',
        'label': 'TRANSITIONAL (INDETERMINATE)', 'emoji': '🟡',
    },
    'normal': {
        'color': '#2dc653', 'bg': '#001a06', 'border': '#44ff77',
        'label': 'NORMAL VARIANT', 'emoji': '🟢',
    },
}

# ── Mesh label tables ─────────────────────────────────────────────────────────
SPINE_LABELS: List[Tuple] = [
    (SP_SACRUM, 'Sacrum (spine)',    '#ff8c00', 0.75, True,  1.5),
    (SP_ARCUS,  'Arcus Vertebrae',   '#7744bb', 0.45, True,  1.5),
    (SP_SPINOUS,'Spinous Processes', '#d4b830', 0.55, True,  1.5),
    (SP_TP_L,   'TP Left',           '#ff3333', 0.92, False, 0.8),
    (SP_TP_R,   'TP Right',          '#00ccff', 0.92, False, 0.8),
    (SP_SAL,    'Sup Articular L',   '#55aa88', 0.55, True,  1.5),
    (SP_SAR,    'Sup Articular R',   '#338866', 0.55, True,  1.5),
    (SP_CORPUS, 'Corpus Border',     '#5588bb', 0.30, True,  1.5),
    (SP_CORD,   'Spinal Cord',       '#ffe066', 0.60, False, 1.0),
    (SP_CANAL,  'Spinal Canal',      '#00ffb3', 0.14, False, 0.8),
]

VERIDAH_COLOURS: Dict[int, Tuple[str, float]] = {
    20: ('#aabbcc', 0.18),  # L1
    21: ('#99aabb', 0.18),  # L2
    22: ('#2288cc', 0.25),  # L3
    23: ('#2266aa', 0.30),  # L4
    24: ('#1e6fa8', 0.45),  # L5 — often TV
    25: ('#33aaff', 0.55),  # L6 — extra lumbar; highlighted
    26: ('#ff8c00', 0.62),  # Sacrum
}
VERIDAH_IVD_COLOURS: Dict[int, str] = {
    20: '#ffe28a', 21: '#ffd060', 22: '#ffb830',
    23: '#ff9900', 24: '#ff7700', 25: '#ff5500',
}

TSS_RENDER: List[Tuple] = [
    (50,  'TSS Sacrum',     '#ff8c00', 0.65),
    (41,  'TSS L1',         '#aabbcc', 0.18),
    (42,  'TSS L2',         '#99aabb', 0.18),
    (43,  'TSS L3',         '#2288cc', 0.22),
    (44,  'TSS L4',         '#2266aa', 0.28),
    (45,  'TSS L5',         '#1e6fa8', 0.40),
    (95,  'TSS disc L4-L5', '#ff9900', 0.45),
    (100, 'TSS disc L5-S1', '#ff5500', 0.50),
]


# ── NIfTI helpers ──────────────────────────────────────────────────────────────

def _load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Cannot reduce {path.name} to 3D")
    return data, nii


def _voxmm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


def _resample(vol: np.ndarray, vox_mm: np.ndarray) -> np.ndarray:
    return ndizoom(vol.astype(np.int32), (vox_mm / ISO_MM).tolist(),
                   order=0, mode='nearest', prefilter=False).astype(np.int32)


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    c = np.array(np.where(mask))
    return c.mean(axis=1) * ISO_MM if c.size else None


def _z_range(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    if not mask.any(): return None
    zc = np.where(mask)[2]
    return int(zc.min()), int(zc.max())


def _inferiormost_cc(mask: np.ndarray,
                     sac: Optional[np.ndarray] = None) -> np.ndarray:
    if not mask.any(): return np.zeros_like(mask, bool)
    lab, n = cc_label(mask)
    if n == 1: return mask.astype(bool)
    sac_zmin = None
    if sac is not None and sac.any():
        sac_zmin = int(np.where(sac)[2].min())
    comps = []
    for i in range(1, n + 1):
        c = (lab == i); zc = np.where(c)[2]
        comps.append((float(zc.mean()), int(zc.max()), c))
    comps.sort(key=lambda t: t[0])
    if sac_zmin is not None:
        cands = [c for _, zm, c in comps if zm < sac_zmin]
        if cands: return cands[0].astype(bool)
    return comps[0][2].astype(bool)


def _tp_height_mm(tp: np.ndarray) -> float:
    if not tp.any(): return 0.0
    zc = np.where(tp)[2]
    return float((int(zc.max()) - int(zc.min()) + 1) * ISO_MM)


def _min_dist(a: np.ndarray, b: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    from scipy.ndimage import distance_transform_edt
    if not a.any() or not b.any(): return float('inf'), None, None
    dt   = distance_transform_edt(~b) * ISO_MM
    di   = np.where(a, dt, np.inf)
    flat = int(np.argmin(di))
    va   = np.array(np.unravel_index(flat, a.shape), dtype=float)
    dist = float(dt[tuple(va.astype(int))])
    cb   = np.array(np.where(b), dtype=float)
    d2   = ((cb.T - va) ** 2).sum(axis=1)
    vb   = cb[:, int(np.argmin(d2))]
    return dist, va * ISO_MM, vb * ISO_MM


def _isolate_z(mask: np.ndarray, z_lo: int, z_hi: int, margin: int = 15) -> np.ndarray:
    out = np.zeros_like(mask)
    lo  = max(0, z_lo - margin)
    hi  = min(mask.shape[2] - 1, z_hi + margin)
    out[:, :, lo:hi + 1] = mask[:, :, lo:hi + 1]
    return out


# ── Marching cubes → Mesh3d ───────────────────────────────────────────────────

def mask_to_mesh3d(iso_mask: np.ndarray,
                   origin_mm: np.ndarray,
                   name: str, colour: str, opacity: float,
                   smooth: float = 1.5, fill: bool = True) -> Optional[go.Mesh3d]:
    if not iso_mask.any(): return None
    m   = binary_fill_holes(iso_mask) if fill else iso_mask.copy()
    vol = gaussian_filter(m.astype(np.float32), sigma=smooth)
    vol = np.pad(vol, 1, mode='constant', constant_values=0)
    if vol.max() <= 0.5 or vol.min() >= 0.5: return None
    try:
        verts, faces, _, _ = marching_cubes(vol, level=0.5,
                                             spacing=(ISO_MM, ISO_MM, ISO_MM))
    except Exception as exc:
        logger.warning(f"  MC failed '{name}': {exc}"); return None
    verts -= ISO_MM
    verts -= origin_mm[np.newaxis, :]
    return go.Mesh3d(
        x=verts[:, 0].tolist(), y=verts[:, 1].tolist(), z=verts[:, 2].tolist(),
        i=faces[:, 0].tolist(), j=faces[:, 1].tolist(), k=faces[:, 2].tolist(),
        color=colour, opacity=opacity, name=name,
        showlegend=True, flatshading=False,
        lighting=dict(ambient=0.38, diffuse=0.72, specular=0.25,
                      roughness=0.55, fresnel=0.15),
        lightposition=dict(x=100, y=200, z=150),
        hoverinfo='name', showscale=False,
    )


# ── 3D annotation traces ───────────────────────────────────────────────────────

def _line(p0, p1, colour, name, width=6, dash='solid') -> go.Scatter3d:
    return go.Scatter3d(
        x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
        mode='lines', line=dict(color=colour, width=width, dash=dash),
        name=name, showlegend=True, hoverinfo='name')


def _marker(pos, text, colour, size=12, sym='circle') -> go.Scatter3d:
    return go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text',
        marker=dict(size=size, color=colour, symbol=sym,
                    line=dict(color='white', width=1)),
        text=[text], textposition='top center',
        textfont=dict(size=10, color=colour),
        name=text, showlegend=False, hoverinfo='text')


def midpt(a, b) -> np.ndarray:
    return (np.asarray(a) + np.asarray(b)) / 2.0


def tp_ruler_traces(tp_iso: np.ndarray, origin_mm: np.ndarray,
                    colour: str, side: str, span_mm: float) -> List:
    if not tp_iso.any(): return []
    zc = np.where(tp_iso)[2]
    z_lo, z_hi = int(zc.min()), int(zc.max())
    xc = np.where(tp_iso)[0]; yc = np.where(tp_iso)[1]
    x_c = int(np.median(xc)); y_c = int(np.median(yc))
    p_lo = np.array([x_c, y_c, z_lo], float) * ISO_MM - origin_mm
    p_hi = np.array([x_c, y_c, z_hi], float) * ISO_MM - origin_mm
    mid  = midpt(p_lo, p_hi)
    flag = f'✗ ≥{TP_HEIGHT_MM:.0f}mm → Type I' if span_mm >= TP_HEIGHT_MM else '✓ <19mm'
    clr  = '#ff4444' if span_mm >= TP_HEIGHT_MM else '#44ff88'
    traces = [_line(p_lo, p_hi, colour, f'TP ruler {side}', width=8)]
    traces.append(_marker(mid, f'{side} TP: {span_mm:.1f}mm  {flag}', clr, size=10, sym='diamond'))
    off = np.array([4., 0., 0.])
    for pt in (p_lo, p_hi):
        traces.append(_line(pt - off, pt + off, colour, f'tick {side}', width=3))
    return traces


def gap_ruler_traces(tp: np.ndarray, sac: np.ndarray,
                     origin_mm: np.ndarray, colour: str,
                     side: str, dist_mm: float) -> List:
    if not tp.any() or not sac.any(): return []
    _, pt_a, pt_b = _min_dist(tp, sac)
    if pt_a is None: return []
    p_a = pt_a - origin_mm; p_b = pt_b - origin_mm
    mid = midpt(p_a, p_b)
    contact = np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM
    lbl = f'CONTACT {dist_mm:.1f}mm' if contact else f'Gap {dist_mm:.1f}mm ✓'
    clr = '#ff2222' if contact else '#44ff88'
    return [
        _line(p_a, p_b, colour, f'gap ruler {side}', width=5,
              dash='dot' if contact else 'dash'),
        _marker(mid, f'{side}: {lbl}', clr, size=8, sym='square'),
    ]


def tv_plane_traces(vert_iso: np.ndarray, tv_label: int,
                    origin_mm: np.ndarray, tv_name: str, phenotype: str) -> List:
    mask = (vert_iso == tv_label)
    if not mask.any(): return []
    zc    = np.where(mask)[2]
    z_mid = int((zc.min() + zc.max()) // 2)
    xs    = np.linspace(0, vert_iso.shape[0] - 1, 14)
    ys    = np.linspace(0, vert_iso.shape[1] - 1, 14)
    xg, yg = np.meshgrid(xs, ys)
    zg    = np.full_like(xg, z_mid)
    cfg   = PHENOTYPE_CONFIG.get(phenotype, PHENOTYPE_CONFIG['normal'])
    col   = cfg['color']

    def _h2r(h):
        h = h.lstrip('#')
        return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

    plane = go.Surface(
        x=xg * ISO_MM - origin_mm[0],
        y=yg * ISO_MM - origin_mm[1],
        z=zg * ISO_MM - origin_mm[2],
        colorscale=[[0, f"rgba({_h2r(col)},0.12)"],
                    [1, f"rgba({_h2r(col)},0.12)"]],
        showscale=False, opacity=0.20,
        name=f'TV plane ({tv_name})', showlegend=True, hoverinfo='name',
    )
    ctr = _centroid(mask)
    if ctr is not None:
        return [plane, _marker(ctr - origin_mm, f'TV: {tv_name}', col, size=16, sym='cross')]
    return [plane]


def tv_body_annotation_traces(vert_iso: np.ndarray, tv_label: int,
                               origin_mm: np.ndarray, shape: dict,
                               phenotype: str) -> List:
    mask = (vert_iso == tv_label)
    if not mask.any() or not shape: return []
    cfg    = PHENOTYPE_CONFIG.get(phenotype, PHENOTYPE_CONFIG['transitional'])
    colour = cfg['color']
    coords = np.array(np.where(mask))
    y_min  = int(coords[1].min()); y_max = int(coords[1].max())
    z_min  = int(coords[2].min()); z_max = int(coords[2].max())
    x_mid  = int(coords[0].mean())
    p_ant  = np.array([x_mid, y_min, (z_min + z_max) // 2], float) * ISO_MM - origin_mm
    p_post = np.array([x_mid, y_max, (z_min + z_max) // 2], float) * ISO_MM - origin_mm
    p_sup  = np.array([x_mid, (y_min + y_max) // 2, z_max], float) * ISO_MM - origin_mm
    p_inf  = np.array([x_mid, (y_min + y_max) // 2, z_min], float) * ISO_MM - origin_mm
    h_ap   = shape.get('h_ap_ratio', 0)
    sc     = shape.get('shape_class', 'unknown')
    return [
        _line(p_ant, p_post, '#8888aa', 'TV AP depth ruler', width=3, dash='dot'),
        _line(p_sup, p_inf,  colour,    'TV SI height ruler', width=4),
        _marker(midpt(p_sup, p_inf), f'TV H/AP={h_ap:.2f} ({sc})', colour, size=9, sym='diamond-open'),
    ]


# ── Metrics HTML panel ─────────────────────────────────────────────────────────

def build_metrics_panel(result: dict) -> str:
    morpho       = result.get('lstv_morphometrics') or {}
    castellvi    = result.get('castellvi_type') or 'None'
    phenotype    = morpho.get('lstv_phenotype', 'normal')
    confidence   = morpho.get('phenotype_confidence', '')
    rationale    = morpho.get('phenotype_rationale', '')
    criteria     = morpho.get('phenotype_criteria', [])
    primary      = morpho.get('primary_criteria_met', [])
    lumbar_count = morpho.get('lumbar_count_consensus', '?')
    tv_name      = morpho.get('tv_name', '?')
    lc_anomaly   = morpho.get('lumbar_count_anomaly', False)
    tv_shape     = morpho.get('tv_shape') or {}
    disc_above   = morpho.get('disc_above') or {}
    disc_below   = morpho.get('disc_below') or {}
    rib          = morpho.get('rib_anomaly') or {}
    score        = result.get('pathology_score', 0)
    xval         = result.get('cross_validation') or {}
    lft          = result.get('left')   or {}
    rgt          = result.get('right')  or {}
    lstv_reasons = result.get('lstv_reason', [])

    cfg   = PHENOTYPE_CONFIG.get(phenotype, PHENOTYPE_CONFIG['normal'])
    p_col = cfg['color']; p_bg = cfg['bg']; p_bdr = cfg['border']
    p_lbl = cfg['label']; p_emj = cfg['emoji']

    def sect(t): return f'<div class="ps">{t}</div>'
    def row(k, v, cls='ok'):
        return (f'<div class="pr"><span class="pk">{k}</span>'
                f'<span class="pv {cls}">{v}</span></div>')
    def crit(txt): return f'<div class="pc">• {txt}</div>'

    lines = []

    # ── Status banner ──────────────────────────────────────────────────────────
    lines.append(
        f'<div class="pstatus" style="background:{p_bg};border:2px solid {p_bdr};color:{p_col}">'
        f'{p_emj} {p_lbl}</div>'
    )
    lines.append(
        f'<div class="pconf" style="color:{p_col}">'
        f'Confidence: {confidence.upper() if confidence else "—"}</div>'
    )

    # ── LSTV detection basis ───────────────────────────────────────────────────
    lines.append(sect('LSTV Detection Basis'))
    if lstv_reasons:
        for r in lstv_reasons:
            lines.append(f'<div class="pc">▶ {r}</div>')
    else:
        lines.append(
            '<div class="pc" style="color:#2dc653">'
            'No LSTV criteria met — normal study</div>'
        )

    # ── Pathology score ────────────────────────────────────────────────────────
    lines.append(sect('Pathology Burden'))
    sc_cls = 'cr' if score >= 8 else 'wn' if score >= 3 else 'ok'
    lines.append(row('Score', f'{score:.0f}', sc_cls))

    # ── Castellvi ─────────────────────────────────────────────────────────────
    lines.append(sect('Castellvi Classification (TP Morphology)'))
    ct_cls = ('cr' if castellvi and any(x in castellvi for x in ('III','IV'))
              else 'wn' if castellvi and any(x in castellvi for x in ('I','II'))
              else 'ok')
    lines.append(row('Type', castellvi, ct_cls))
    l_h = lft.get('tp_height_mm', 0); r_h = rgt.get('tp_height_mm', 0)
    l_d = lft.get('dist_mm', float('inf')); r_d = rgt.get('dist_mm', float('inf'))
    l_cls_r = lft.get('classification', '—'); r_cls_r = rgt.get('classification', '—')
    lines.append(row('Left',
                     f'{l_cls_r} | h={l_h:.1f}mm | d={l_d:.1f}mm',
                     'cr' if (l_h >= TP_HEIGHT_MM or l_d <= CONTACT_DIST_MM) else 'ok'))
    lines.append(row('Right',
                     f'{r_cls_r} | h={r_h:.1f}mm | d={r_d:.1f}mm',
                     'cr' if (r_h >= TP_HEIGHT_MM or r_d <= CONTACT_DIST_MM) else 'ok'))
    lines.append(row('TP≥19mm (Castellvi 1984)',
                     f'L:{l_h:.1f}mm {"✗" if l_h>=TP_HEIGHT_MM else "✓"}  '
                     f'R:{r_h:.1f}mm {"✗" if r_h>=TP_HEIGHT_MM else "✓"}',
                     'cr' if (l_h >= TP_HEIGHT_MM or r_h >= TP_HEIGHT_MM) else 'ok'))

    # ── Lumbar count ───────────────────────────────────────────────────────────
    lines.append(sect('Lumbar Vertebrae'))
    lc_cls  = 'cr' if lc_anomaly else 'ok'
    lc_note = '⚠ Expected 5' if lc_anomaly else 'Normal'
    lines.append(row('Count (consensus)', f'{lumbar_count}  ({lc_note})', lc_cls))
    lines.append(row('TSS count',     str(morpho.get('lumbar_count_tss', '—'))))
    lines.append(row('VERIDAH count', str(morpho.get('lumbar_count_veridah', '—'))))
    if morpho.get('has_l6'):
        lines.append(row('L6 present', 'YES — LUMBARIZATION signal', 'wn'))
    lc_note_full = morpho.get('lumbar_count_note', '')
    if lc_note_full:
        lines.append(f'<div class="pc">{lc_note_full}</div>')

    # ── TV identification ──────────────────────────────────────────────────────
    lines.append(sect('Transitional Vertebra (TV)'))
    lines.append(row('TV name', tv_name, 'wn' if tv_name not in ('N/A', '?', None) else 'ok'))
    h_ap = tv_shape.get('h_ap_ratio'); shpc = tv_shape.get('shape_class', '—')
    shp_cls = ('cr' if shpc == 'sacral-like' else 'wn' if shpc == 'transitional' else 'ok')
    if h_ap:
        lines.append(row('Body H/AP ratio', f'{h_ap:.2f}  ({shpc})', shp_cls))
        lines.append(row('H/AP reference (Nardo 2012)',
                         f'Lumbar >{TV_SHAPE_LUMBAR}  '
                         f'Trans {TV_SHAPE_SACRAL}–{TV_SHAPE_LUMBAR}  '
                         f'Sacral <{TV_SHAPE_SACRAL}', 'pm'))
    nr = tv_shape.get('norm_ratio')
    if nr:
        lines.append(row('TV/L4 H:AP ratio',
                         f'{nr:.2f}  {"↓ squarer than L4" if nr < 0.90 else "similar to L4"}',
                         'wn' if nr < 0.80 else 'ok'))

    # ── Adjacent disc heights ──────────────────────────────────────────────────
    lines.append(sect('Adjacent Disc Heights — DHI (Farfan 1972)'))
    for disc, label in ((disc_above, 'Above TV'), (disc_below, 'Below TV')):
        dhi = disc.get('dhi_pct'); lvl = disc.get('level', '—')
        if dhi is not None:
            cls = ('cr' if dhi < DHI_REDUCED_PCT else
                   'wn' if dhi < DHI_MODERATE_PCT else 'ok')
            lines.append(row(f'{label} ({lvl})', f'{dhi:.0f}%  [{disc.get("grade","?")}]', cls))
        elif disc.get('is_absent'):
            lines.append(row(f'{label} ({lvl})', 'ABSENT — possible fusion', 'cr'))
        else:
            lines.append(row(f'{label} ({lvl})', 'Not detected', 'pm'))

    # ── Rib anomaly ────────────────────────────────────────────────────────────
    lines.append(sect('Rib / Thoracic Count'))
    thr_count = rib.get('thoracic_count')
    if thr_count is not None:
        cls = 'cr' if rib.get('count_anomaly') else 'ok'
        lines.append(row('Thoracic vertebrae',
                         f'{thr_count}  (expected {rib.get("expected_thoracic", 12)})', cls))
    if rib.get('lumbar_rib_l1'):
        h_lr = rib.get('lumbar_rib_l1_h_mm', 0)
        lines.append(row('Lumbar rib (L1 TP)',
                         f'⚠ Suspected — {h_lr:.1f}mm ≥ {TP_HEIGHT_MM}mm', 'cr'))
    if rib.get('description'):
        lines.append(f'<div class="pc">{rib["description"]}</div>')

    # ── Primary criteria ───────────────────────────────────────────────────────
    if primary:
        lines.append(sect('Primary Criteria Met'))
        for p in primary:
            lines.append(f'<div class="pc">✓ {p}</div>')

    # ── Evidence ───────────────────────────────────────────────────────────────
    if criteria:
        lines.append(sect('Classification Evidence'))
        for c in criteria[:6]:
            lines.append(crit(c))
        if len(criteria) > 6:
            lines.append(f'<div class="pc" style="color:#4455aa">+{len(criteria)-6} more…</div>')

    # ── Rationale ──────────────────────────────────────────────────────────────
    if rationale:
        lines.append(sect('Rationale'))
        lines.append(f'<div class="preason">{rationale}</div>')

    # ── Cross-validation ───────────────────────────────────────────────────────
    xval_warns = xval.get('warnings', [])
    if xval_warns:
        lines.append(sect('⚠ QC Warnings'))
        for w in xval_warns: lines.append(f'<div class="pwarning">{w}</div>')
    else:
        sd = xval.get('sacrum_dice')
        if sd is not None:
            lines.append(sect('QC (Cross-Validation)'))
            lines.append(row('Sacrum Dice (SPINEPS/TSS)', f'{sd:.3f}',
                             'ok' if sd >= 0.30 else 'cr'))

    return '\n'.join(lines)


# ── Main figure builder ────────────────────────────────────────────────────────

def build_3d_figure(study_id: str,
                    spineps_dir: Path,
                    totalspine_dir: Path,
                    result: dict,
                    smooth: float = 2.0,
                    show_tss: bool = True):
    seg_dir   = spineps_dir / 'segmentations' / study_id
    sp_path   = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    vert_path = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
    tss_path  = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"

    def _load(p, tag):
        if not p.exists(): logger.warning(f"  Missing: {p.name}"); return None, None
        try: return _load_canonical(p)
        except Exception as exc: logger.warning(f"  {tag}: {exc}"); return None, None

    sp_arr,  nii_ref  = _load(sp_path,   'seg-spine_msk')
    vert_arr, _       = _load(vert_path, 'seg-vert_msk')
    tss_arr,  tss_nii = _load(tss_path,  'TSS sagittal')

    if sp_arr is None or vert_arr is None:
        logger.error(f"[{study_id}] Missing required masks"); return None

    vox_mm   = _voxmm(nii_ref)
    sp_iso   = _resample(sp_arr.astype(np.int32),   vox_mm)
    vert_iso = _resample(vert_arr.astype(np.int32), vox_mm)
    tss_iso  = (_resample(tss_arr.astype(np.int32), _voxmm(tss_nii))
                if tss_arr is not None and tss_nii is not None else None)

    sp_labels   = frozenset(np.unique(sp_iso).tolist())   - {0}
    vert_labels = frozenset(np.unique(vert_iso).tolist()) - {0}
    tss_labels  = (frozenset(np.unique(tss_iso).tolist()) - {0}
                   if tss_iso is not None else frozenset())

    morpho    = result.get('lstv_morphometrics') or {}
    phenotype = morpho.get('lstv_phenotype', 'normal')
    tv_name   = morpho.get('tv_name') or result.get('details', {}).get('tv_name', 'L5')
    cfg       = PHENOTYPE_CONFIG.get(phenotype, PHENOTYPE_CONFIG['normal'])
    tv_label  = morpho.get('tv_label_veridah') or result.get('details', {}).get('tv_label')

    sac_iso = np.zeros(sp_iso.shape, bool)
    if tss_iso is not None and TSS_SACRUM in tss_labels:
        sac_iso = (tss_iso == TSS_SACRUM)
    elif SP_SACRUM in sp_labels:
        sac_iso = (sp_iso == SP_SACRUM)

    col_mask  = (vert_iso > 0)
    origin_mm = (_centroid(col_mask)
                 if col_mask.any()
                 else np.array(sp_iso.shape, float) / 2.0 * ISO_MM)

    tv_z = _z_range(vert_iso == tv_label) if tv_label else None

    def _get_tp(tp_lbl):
        if tp_lbl not in sp_labels: return np.zeros(sp_iso.shape, bool)
        if tv_z:
            isolated = _isolate_z(sp_iso == tp_lbl, tv_z[0], tv_z[1])
            tp       = _inferiormost_cc(isolated, sac_iso if sac_iso.any() else None)
            if not tp.any(): tp = (sp_iso == tp_lbl)
        else:
            tp = (sp_iso == tp_lbl)
        return tp

    tp_L   = _get_tp(SP_TP_L);  tp_R   = _get_tp(SP_TP_R)
    span_L = _tp_height_mm(tp_L); span_R = _tp_height_mm(tp_R)
    dist_L = _min_dist(tp_L, sac_iso)[0]; dist_R = _min_dist(tp_R, sac_iso)[0]
    castellvi = result.get('castellvi_type') or 'None'
    cls_L  = result.get('left',  {}).get('classification', 'Normal')
    cls_R  = result.get('right', {}).get('classification', 'Normal')

    traces = []

    # SPINEPS subregion meshes
    for lbl, name, col, op, fh, mx_s in SPINE_LABELS:
        if lbl not in sp_labels: continue
        mask = (tp_L if lbl == SP_TP_L else tp_R if lbl == SP_TP_R else (sp_iso == lbl))
        if not mask.any(): continue
        t = mask_to_mesh3d(mask, origin_mm, name, col, op, min(smooth, mx_s), fh)
        if t: traces.append(t)

    # VERIDAH vertebrae
    for lbl, (col, op) in sorted(VERIDAH_COLOURS.items()):
        if lbl not in vert_labels: continue
        eff_col = cfg['color'] if lbl == tv_label else col
        eff_op  = 0.65 if lbl == tv_label else op
        t = mask_to_mesh3d(vert_iso == lbl, origin_mm,
                           VERIDAH_NAMES.get(lbl, str(lbl)), eff_col, eff_op, smooth, True)
        if t: traces.append(t)

    # VERIDAH IVDs
    for base, col in VERIDAH_IVD_COLOURS.items():
        ivd_lbl = VD_IVD_BASE + base
        if ivd_lbl not in vert_labels: continue
        t = mask_to_mesh3d(vert_iso == ivd_lbl, origin_mm,
                           f'IVD below {VERIDAH_NAMES.get(base, str(base))}',
                           col, 0.50, smooth, True)
        if t: traces.append(t)

    # TSS selective
    if show_tss and tss_iso is not None:
        for lbl, name, col, op in TSS_RENDER:
            if lbl not in tss_labels: continue
            t = mask_to_mesh3d(tss_iso == lbl, origin_mm, name, col, op,
                               0.8 if lbl in (TSS_CORD, TSS_CANAL) else smooth,
                               lbl not in (TSS_CORD, TSS_CANAL))
            if t: traces.append(t)

    # Annotation traces
    if tv_label and tv_label in vert_labels:
        traces += tv_plane_traces(vert_iso, tv_label, origin_mm, tv_name, phenotype)
    tv_shape_dict = morpho.get('tv_shape') or {}
    if tv_label and tv_label in vert_labels and tv_shape_dict:
        traces += tv_body_annotation_traces(vert_iso, tv_label, origin_mm,
                                             tv_shape_dict, phenotype)
    traces += tp_ruler_traces(tp_L, origin_mm, '#ff3333', 'Left',  span_L)
    traces += tp_ruler_traces(tp_R, origin_mm, '#00ccff', 'Right', span_R)
    traces += gap_ruler_traces(tp_L, sac_iso, origin_mm, '#ff8800', 'Left',  dist_L)
    traces += gap_ruler_traces(tp_R, sac_iso, origin_mm, '#00aaff', 'Right', dist_R)

    if not any(isinstance(tr, go.Mesh3d) for tr in traces):
        logger.error(f"[{study_id}] Zero meshes generated"); return None

    lumbar_count = morpho.get('lumbar_count_consensus', '?')
    score        = result.get('pathology_score', 0)
    lstv_flag    = '⚠ LSTV' if result.get('lstv_detected') else '✓ Normal'

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f"<b>{study_id}</b>  ·  "
                  f"<span style='color:{cfg['color']}'>{cfg['label']}</span>  ·  "
                  f"Castellvi <b>{castellvi}</b>  ·  "
                  f"TV: <b>{tv_name}</b>  ·  "
                  f"Lumbar: <b>{lumbar_count}</b>  ·  "
                  f"{lstv_flag}  ·  Score: <b>{score:.0f}</b>"),
            font=dict(size=11, color='#e8e8f0'), x=0.01),
        paper_bgcolor='#0a0a18', plot_bgcolor='#0a0a18',
        scene=dict(
            bgcolor='#0a0a18',
            xaxis=_axis('X'), yaxis=_axis('Y'), zaxis=_axis('Z (SI)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6, y=0.2, z=0.3), up=dict(x=0, y=0, z=1))),
        legend=dict(font=dict(color='#e8e8f0', size=8),
                    bgcolor='rgba(10,10,24,0.88)', bordercolor='#222244', borderwidth=1,
                    x=0.01, y=0.97, itemsizing='constant'),
        margin=dict(l=0, r=0, t=36, b=0),
    )

    return (fig, castellvi, tv_name, cls_L, cls_R,
            span_L, span_R, dist_L, dist_R, morpho, cfg)


def _axis(title: str) -> dict:
    return dict(title=title, showgrid=True, gridcolor='#1a1a3a',
                showbackground=True, backgroundcolor='#0a0a18',
                tickfont=dict(color='#6666aa'), titlefont=dict(color='#6666aa'),
                zeroline=False)


# ── HTML template ──────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LSTV 3D — {study_id}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0a0a18;--sf:#11112a;--bd:#222244;--tx:#e8e8f0;--mu:#5566aa}}
html,body{{background:var(--bg);color:var(--tx);font-family:'JetBrains Mono',monospace;
           height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{display:flex;align-items:center;flex-wrap:wrap;gap:6px;padding:5px 10px;
        border-bottom:1px solid var(--bd);background:var(--sf);flex-shrink:0}}
h1{{font-family:'Syne',sans-serif;font-size:.75rem;font-weight:700}}
.b{{display:inline-block;padding:2px 7px;border-radius:14px;font-size:.60rem;font-weight:600;white-space:nowrap}}
.bs{{background:#222244;color:var(--mu)}}
.bc{{background:{status_bg};color:{status_color};border:1px solid {status_border};font-size:.72rem;font-weight:700;padding:3px 10px;border-radius:4px}}
.bct{{background:#1a2a3a;color:#5599cc}} .bch{{background:#2a1a0a;color:#cc8833}}
.bln{{background:#0a2a0a;color:#44cc66}} .ble{{background:#2a0a0a;color:#ff3333}}
.bnl{{background:#0a1a0a;color:#55cc77;font-weight:600}}
.bsc{{background:#1a1a3a;color:#aaaaff;font-weight:700}}
.tb{{display:flex;gap:4px;align-items:center;margin-left:auto;flex-wrap:wrap}}
.tb span{{font-size:.55rem;color:var(--mu);text-transform:uppercase}}
button{{background:var(--bg);border:1px solid var(--bd);color:var(--tx);
        font-family:inherit;font-size:.60rem;padding:2px 7px;border-radius:4px;cursor:pointer}}
button:hover{{background:var(--bd)}} button.on{{background:#3366ff;border-color:#3366ff;color:#fff}}
.mt{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:3px 10px;
     border-bottom:1px solid var(--bd);flex-shrink:0;font-size:.60rem}}
.m{{display:flex;align-items:center;gap:3px;color:var(--mu)}}
.v{{color:var(--tx);font-weight:600}}
.ok{{color:#2dc653!important}} .wn{{color:#ff8800!important}} .cr{{color:#ff3333!important}}
.note{{font-size:.50rem;color:#3a3a5a;margin-left:auto}}
.main-row{{display:flex;flex:1;min-height:0;overflow:hidden}}
#pl{{flex:1;min-width:0;min-height:0;overflow:hidden}}
#pl .js-plotly-plot,#pl .plot-container{{height:100%!important}}
#mp{{width:250px;flex-shrink:0;background:rgba(10,10,24,0.94);
     border-left:1px solid var(--bd);overflow-y:auto;overflow-x:hidden;
     padding:0 0 12px 0;font-size:.59rem;scrollbar-width:thin;scrollbar-color:#222244 var(--bg)}}
#mp::-webkit-scrollbar{{width:3px}}
#mp::-webkit-scrollbar-thumb{{background:#222244;border-radius:2px}}
.pstatus{{text-align:center;font-family:'Syne',sans-serif;font-size:.80rem;font-weight:700;
  letter-spacing:.04em;text-transform:uppercase;padding:8px 10px;margin-bottom:2px}}
.pconf{{text-align:center;font-size:.55rem;font-weight:600;padding:2px 10px 4px;opacity:.85}}
.ps{{font-family:'Syne',sans-serif;font-size:.55rem;font-weight:700;color:#5566aa;
     text-transform:uppercase;letter-spacing:.06em;padding:5px 8px 2px;margin-top:4px;
     border-top:1px solid #161630}}
.ps:first-child{{border-top:none;margin-top:0}}
.pr{{display:flex;justify-content:space-between;align-items:baseline;padding:1px 8px;gap:4px}}
.pk{{color:#5566aa;white-space:nowrap;flex-shrink:0;max-width:52%}}
.pv{{text-align:right;color:var(--tx);font-weight:600;word-break:break-word}}
.pv.ok{{color:#2dc653}} .pv.wn{{color:#ff8800}} .pv.cr{{color:#ff3333}} .pv.pm{{color:#6677aa;font-weight:400}}
.pc{{padding:2px 10px;color:#8899bb;font-size:.56rem;line-height:1.4}}
.preason{{padding:3px 8px 4px;color:#aabbcc;font-size:.55rem;line-height:1.5;
          border-left:2px solid #3344aa;margin:2px 6px}}
.pwarning{{padding:2px 8px;color:#ffaa33;font-size:.55rem}}
.lg{{display:flex;gap:6px;flex-wrap:wrap;align-items:center;padding:2px 10px;
     border-bottom:1px solid var(--bd);flex-shrink:0;font-size:.58rem}}
.li{{display:flex;align-items:center;gap:3px;color:var(--mu)}}
.sw{{width:8px;height:8px;border-radius:2px;flex-shrink:0}}
</style></head><body>
<header>
  <h1>LSTV 3D</h1>
  <span class="b bs">{study_id}</span>
  <span class="b bc">{status_emoji}&nbsp;{status_label}</span>
  <span class="b bct">Castellvi: {castellvi}</span>
  <span class="b bch">TV: {tv_name}</span>
  <span class="b {lc_badge}">Lumbar: {lumbar_count}</span>
  {rib_badge}
  {lstv_badge}
  <span class="b bsc">Score: {score:.0f}</span>
  <div class="tb">
    <span>View</span>
    <button onclick="sv('oblique')" id="b-oblique" class="on">Oblique</button>
    <button onclick="sv('lat')"   id="b-lat">Lat</button>
    <button onclick="sv('post')"  id="b-post">Post</button>
    <button onclick="sv('ant')"   id="b-ant">Ant</button>
    <button onclick="sv('axial')" id="b-axial">Axial</button>
  </div>
</header>
<div class="mt">
  <div class="m">TP-L <span class="v {tpl_c}">{span_L:.1f}mm</span></div>
  <div class="m">TP-R <span class="v {tpr_c}">{span_R:.1f}mm</span></div>
  <div class="m">Gap-L <span class="v {gl_c}">{gap_L}</span></div>
  <div class="m">Gap-R <span class="v {gr_c}">{gap_R}</span></div>
  <div class="m">H/AP <span class="v {hap_c}">{hap_disp}</span></div>
  <div class="m">DHI-below <span class="v {dhi_c}">{dhi_disp}</span></div>
  <span class="note">drag=rotate · scroll=zoom · click legend=toggle</span>
</div>
<div class="lg">
  <div class="li"><div class="sw" style="background:#ff3333"></div>TP-Left</div>
  <div class="li"><div class="sw" style="background:#00ccff"></div>TP-Right</div>
  <div class="li"><div class="sw" style="background:#ff8c00"></div>Sacrum</div>
  <div class="li"><div class="sw" style="background:{tv_col};opacity:.9"></div>TV ({tv_name})</div>
  <div class="li"><div class="sw" style="background:#7744bb;opacity:.7"></div>Arcus</div>
  <div class="li"><div class="sw" style="background:#ffe066;opacity:.8"></div>Cord</div>
  <div class="li"><div class="sw" style="background:#00ffb3;opacity:.4"></div>Canal</div>
</div>
<div class="main-row">
  <div id="pl">{plotly_div}</div>
  <div id="mp">{metrics_panel}</div>
</div>
<script>
const CAM={{
  oblique:{{eye:{{x:1.6,y:0.3,z:0.35}},up:{{x:0,y:0,z:1}}}},
  lat:    {{eye:{{x:2.5,y:0.0,z:0.0}},up:{{x:0,y:0,z:1}}}},
  post:   {{eye:{{x:0.0,y:2.5,z:0.0}},up:{{x:0,y:0,z:1}}}},
  ant:    {{eye:{{x:0.0,y:-2.5,z:0.0}},up:{{x:0,y:0,z:1}}}},
  axial:  {{eye:{{x:0.0,y:0.0,z:3.0}},up:{{x:0,y:1,z:0}}}},
}};
function sv(n){{
  const pd=document.querySelector('#pl .js-plotly-plot');
  if(!pd)return;
  Plotly.relayout(pd,{{'scene.camera.eye':CAM[n].eye,'scene.camera.up':CAM[n].up}});
  document.querySelectorAll('.tb button').forEach(b=>b.classList.remove('on'));
  const b=document.getElementById('b-'+n);if(b)b.classList.add('on');
}}
window.addEventListener('resize',()=>{{
  const pd=document.querySelector('#pl .js-plotly-plot');
  if(pd)Plotly.Plots.resize(pd);
}});
</script>
</body></html>"""


def save_html(fig, study_id: str, output_dir: Path,
              castellvi: str, tv_name: str, cls_L: str, cls_R: str,
              span_L: float, span_R: float, dist_L: float, dist_R: float,
              morpho: dict, cfg: dict, result: dict) -> Path:
    from plotly.io import to_html

    plotly_div = to_html(fig, full_html=False, include_plotlyjs='cdn',
                          config=dict(responsive=True, displayModeBar=True, displaylogo=False))

    def _f(v): return f'{v:.1f}mm' if (v is not None and np.isfinite(v)) else 'N/A'
    def _gc(v): return 'cr' if (np.isfinite(v) and v <= CONTACT_DIST_MM) else 'ok'
    def _hc(v): return 'cr' if v >= TP_HEIGHT_MM else 'ok'

    phenotype    = morpho.get('lstv_phenotype', 'normal')
    lumbar_count = morpho.get('lumbar_count_consensus', '?')
    lc_anomaly   = morpho.get('lumbar_count_anomaly', False)
    score        = result.get('pathology_score', 0)
    rib          = morpho.get('rib_anomaly') or {}
    tv_shape     = morpho.get('tv_shape') or {}
    disc_below   = morpho.get('disc_below') or {}
    h_ap         = tv_shape.get('h_ap_ratio')
    shpc         = tv_shape.get('shape_class', '')
    dhi_b        = disc_below.get('dhi_pct')

    status_cfg    = PHENOTYPE_CONFIG.get(phenotype, PHENOTYPE_CONFIG['normal'])
    lc_badge      = 'ble' if lc_anomaly else 'bln'
    rib_badge     = ('<span class="b ble">Rib anomaly ⚠</span>'
                     if rib.get('any_anomaly') else '')
    lstv_detected = result.get('lstv_detected', False)
    lstv_badge    = ('<span class="b ble">⚠ LSTV</span>'
                     if lstv_detected
                     else '<span class="b bnl">✓ Normal</span>')
    hap_c         = ('cr' if shpc == 'sacral-like' else
                     'wn' if shpc == 'transitional' else 'ok')
    hap_disp      = f'{h_ap:.2f} ({shpc})' if h_ap else 'N/A'
    dhi_c         = ('cr' if dhi_b and dhi_b < DHI_REDUCED_PCT else
                     'wn' if dhi_b and dhi_b < DHI_MODERATE_PCT else 'ok')
    dhi_disp      = f'{dhi_b:.0f}%' if dhi_b else 'N/A'
    tv_col        = status_cfg['color']

    html = _HTML_TEMPLATE.format(
        study_id      = study_id,
        status_label  = status_cfg['label'],
        status_emoji  = status_cfg['emoji'],
        status_color  = status_cfg['color'],
        status_bg     = status_cfg['bg'],
        status_border = status_cfg['border'],
        castellvi     = castellvi,
        tv_name       = tv_name,
        lumbar_count  = lumbar_count,
        lc_badge      = lc_badge,
        rib_badge     = rib_badge,
        lstv_badge    = lstv_badge,
        score         = score,
        span_L        = span_L, span_R = span_R,
        tpl_c=_hc(span_L), tpr_c=_hc(span_R),
        gap_L=_f(dist_L),  gap_R=_f(dist_R),
        gl_c=_gc(dist_L),  gr_c=_gc(dist_R),
        hap_disp=hap_disp, hap_c=hap_c,
        dhi_disp=dhi_disp, dhi_c=dhi_c,
        tv_col=tv_col,
        metrics_panel = build_metrics_panel(result),
        plotly_div    = plotly_div,
    )

    out_path = output_dir / f"{study_id}_lstv_3d.html"
    out_path.write_text(html, encoding='utf-8')
    logger.info(f"  → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# ── Study ranking ──────────────────────────────────────────────────────────────

def rank_studies(results: List[dict],
                 n_pathologic: int,
                 n_normal: int) -> Tuple[List[str], List[str]]:
    """
    Partition studies into pathologic (top N by score) and strict-normal (bottom N).
    A study is STRICTLY NORMAL iff:
      • lstv_detected = False
      • lumbar_count_consensus = 5
      • pathology_score = 0
    Falls back to lowest-scored if not enough strict normals.
    """
    scored = sorted(
        ((r['study_id'], r.get('pathology_score') or 0) for r in results),
        key=lambda t: t[1], reverse=True,
    )
    pathologic_ids = {sid for sid, _ in scored[:n_pathologic]}
    pathologic     = [sid for sid, _ in scored[:n_pathologic]]
    result_by_id   = {r['study_id']: r for r in results}

    def _is_normal(sid: str) -> bool:
        r = result_by_id.get(sid, {})
        if r.get('lstv_detected'): return False
        morpho = r.get('lstv_morphometrics') or {}
        cnt    = morpho.get('lumbar_count_consensus')
        if cnt is not None and cnt != EXPECTED_LUMBAR: return False
        if (r.get('pathology_score') or 0) > 0: return False
        return True

    normal_pool = [(sid, sc) for sid, sc in reversed(scored)
                   if sid not in pathologic_ids and _is_normal(sid)]

    if len(normal_pool) < n_normal:
        logger.warning(f"Only {len(normal_pool)} strict-normal studies; falling back to lowest-scored")
        extra = [(sid, sc) for sid, sc in reversed(scored)
                 if sid not in pathologic_ids and sid not in {s for s, _ in normal_pool}]
        normal_pool = (normal_pool + extra)[:n_normal]

    normal = [sid for sid, _ in normal_pool[:n_normal]]

    for sid in pathologic:
        sc = next(s for i, s in scored if i == sid)
        r  = result_by_id.get(sid, {})
        ph = (r.get('lstv_morphometrics') or {}).get('lstv_phenotype', '?')
        logger.info(f"  PATHOLOGIC  {sid}: score={sc:.1f}  phenotype={ph}  "
                    f"reasons={r.get('lstv_reason', [])}")
    for sid in normal:
        sc = next((s for i, s in scored if i == sid), 0)
        logger.info(f"  NORMAL      {sid}: score={sc:.1f}")

    return pathologic, normal


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description='LSTV-focused 3D spine visualiser with pathology ranking')
    parser.add_argument('--spineps_dir',    required=True)
    parser.add_argument('--totalspine_dir', required=True)
    parser.add_argument('--output_dir',     required=True)
    parser.add_argument('--lstv_json',      default=None)
    parser.add_argument('--study_id',       default=None)
    parser.add_argument('--all',            action='store_true')
    parser.add_argument('--rank_by',        default='lstv')
    parser.add_argument('--top_n',          type=int, default=5)
    parser.add_argument('--top_normal',     type=int, default=2)
    parser.add_argument('--smooth',         type=float, default=2.0)
    parser.add_argument('--no_tss',         action='store_true')
    args = parser.parse_args()

    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_root = spineps_dir / 'segmentations'

    all_results: List[dict] = []
    result_by_id: Dict[str, dict] = {}
    if args.lstv_json:
        p = Path(args.lstv_json)
        if p.exists():
            with open(p) as fh: all_results = json.load(fh)
            result_by_id = {str(r['study_id']): r for r in all_results}
            logger.info(f"Loaded {len(result_by_id)} results from {p.name}")
        else:
            logger.warning(f"lstv_json not found: {p}")

    if args.study_id:
        study_ids = [args.study_id]
    elif args.all or args.rank_by == 'all':
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        logger.info(f"ALL mode: {len(study_ids)} studies")
    elif args.rank_by == 'lstv':
        if not all_results:
            parser.error("--rank_by lstv requires --lstv_json")
        pathologic, normal = rank_studies(all_results, args.top_n, args.top_normal)
        seen = set(pathologic)
        study_ids = pathologic + [s for s in normal if s not in seen]
        study_ids = [s for s in study_ids if (seg_root / s).is_dir()]
    else:
        parser.error("--rank_by must be 'lstv' or 'all', or use --study_id / --all")

    logger.info(f"Rendering {len(study_ids)} studies → {output_dir}")

    ok = 0
    for sid in study_ids:
        logger.info(f"\n{'='*60}\n[{sid}]")
        try:
            result = result_by_id.get(sid, {
                'study_id': sid, 'lstv_detected': False, 'lstv_reason': [],
                'castellvi_type': None, 'left': {}, 'right': {},
                'lstv_morphometrics': None, 'pathology_score': 0,
            })
            if result.get('pathology_score') is None:
                result['pathology_score'] = compute_lstv_pathology_score(
                    result, result.get('lstv_morphometrics'))

            out = build_3d_figure(
                sid, spineps_dir, totalspine_dir, result,
                smooth=args.smooth, show_tss=not args.no_tss,
            )
            if out is None: continue

            (fig, castellvi, tv_name, cls_L, cls_R,
             span_L, span_R, dist_L, dist_R, morpho, cfg) = out

            save_html(fig, sid, output_dir,
                      castellvi, tv_name, cls_L, cls_R,
                      span_L, span_R, dist_L, dist_R, morpho, cfg, result)
            ok += 1

        except Exception as exc:
            logger.error(f"  [{sid}] Failed: {exc}")
            logger.debug(traceback.format_exc())

    logger.info(f"\n{'='*60}\nDone. {ok}/{len(study_ids)} HTMLs → {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
