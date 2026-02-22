#!/usr/bin/env python3
"""
06_visualize_3d.py  —  Comprehensive Interactive 3D Spine Segmentation Viewer
==============================================================================
ISOTROPIC-FIRST ARCHITECTURE
  Every mask is resampled to 1×1×1 mm³ BEFORE meshing or measurement.
  Root cause of missing TP voxels: sagittal MRI is ~4.88×0.94×0.94 mm
  (only 15 slices).  A TP spanning 1-2 slices collapses to a flat sheet
  under step-subsampling; marching cubes finds no isosurface.
  After zoom to 1mm³ those same slices become ~5 solid voxels → MC works.

SPINEPS seg-spine_msk labels (from README + live logs):
  26=Sacrum  41=Arcus  42=Spinous  43=TP-Left  44=TP-Right
  45=SupArt-L  46=SupArt-R  47=InfArt-L  48=InfArt-R
  49=CorpusBorder  60=Cord  61=Canal  62=Endplate  100=Disc(all)

VERIDAH seg-vert_msk labels:
  1-7=C1-C7  8-19=T1-T12  28=T13  20=L1  21-25=L2-L6  26=Sacrum
  100+X=IVD below X   200+X=Endplate of X

TotalSpineSeg sagittal labels (from README + live logs):
  1=cord  2=canal  31=T11  32=T12  41=L1..45=L5  50=sacrum
  82=disc_T11-T12  91=disc_T12-L1  92-95=disc_L1-L4_L5  100=disc_L5-S
  ⚠  TSS 43=L3 vertebra body  44=L4 vertebra body  (different from SPINEPS!)

MORPHOMETRICS IMPLEMENTED (from Spine_Morphometrics__Beyond_Basic_Measurements.pdf):
─────────────────────────────────────────────────────────────────────────────────
1. VERTEBRAL BODY MORPHOMETRY
   • Genant SQ grading (normal/mild/moderate/severe)
   • QM: Ha (anterior height), Hm (middle height), Hp (posterior height)
   • Height Ratios: Compression (Hm/Ha or Hm/Hp), Wedge (Ha/Hp), Crush (Hp/Ha)
     – Threshold <0.8 for biconcave / wedge / posterior height loss
   • Wedge Ratio <0.75 → Moderate/Severe fracture intervention threshold
   • Lumbar Lordosis Angle (LLA) from vertebral body anterior>posterior height
   • Sagittal Translation ≥3mm → degenerative spondylolisthesis

2. TRANSLATIONAL INSTABILITY
   • Sagittal translation (linear displacement mm; normal <2-4mm)
   • RDT Index: translation per degree of rotation
   • AVI-Index: vertical motion perpendicular to endplate

3. MODIC CHANGES
   • Type 1: T1↓ T2↑ → fibrovascular/edema
   • Type 2: T1↑ T2 iso/↑ → fatty replacement
   • Type 3: T1↓ T2↓ → sclerosis
   • MCG Grade A <25%, Grade B 25-50%, Grade C >50% vertebral body

4. INTERVERTEBRAL DISC (IVD) MORPHOMETRICS
   • DHI: DHI = (Ha+Hp)/(Ds+Di) × 100  (Farfan method)
   • DHI Method 1: ratio anterior+posterior height / disc diameter
   • DHI Method 2: ratio mid-disc height / mid-vertebral body height
   • DHI Method 3: ratio mid-disc height / disc diameter
   • DHI Method 6: mean IVD height / mean vertebral heights
   • Pfirrmann Grades I–V (nucleus/annulus distinction, signal, height)
   • NP-to-CSA ratio (nucleus pulposus / cross-sectional area)
   • DSC reliability threshold >90 for DHI measurements

5. CENTRAL SPINAL CANAL STENOSIS
   • DSCA thresholds: Normal >100mm², Relative 75-100mm², Absolute <70-75mm²
   • AP diameter: Normal >12mm, Relative 10-12mm, Absolute <7-10mm
   • Canal shape per level: L1-L2 oval, L3-L4 triangular, L5 trefoil (60-65%)
   • Trefoil → predisposed to lateral recess narrowing
   • Critical threshold 11.13mm (Indian population study)
   • AP mean absolute error 0.59–0.75mm (SpineLogic benchmark)

6. LATERAL RECESS STENOSIS
   • LRD (Lateral Recess Depth) ≤3mm → stenosis threshold
   • LRA (Lateral Recess Angle) → superior predictor vs interfacet distance
   • Lateral Recess Height ≤2mm → stenosis
   • NPLC distance 0.7±0.3cm from dura at L4-L5
   • Trefoil narrowing vs acute angular pinch mechanisms

7. NEURAL FORAMINAL STENOSIS
   • Lee Grade 0-3: normal / mild / moderate / severe (κ>0.81)
   • Volumetric elliptical cylinder: V = π × a × b × H / 4
     Level norms: L1/L2 ~580mm³, L2/L3 ~700mm³, L3/L4 ~770mm³,
                  L4/L5 ~800mm³, L5/S1 ~824mm³
   • N/F ratio (nerve root / foramen occupancy)
   • Grade 3 threshold → intervention (morphologic nerve change)

8. LIGAMENTUM FLAVUM
   • Normal LFT baseline 3.5mm at L4-L5
   • Hypertrophied LFT ≥ upper limit for healthy adults
   • Severe LFH → high risk neurogenic claudication
   • LFA cutoff 105.90mm² → optimal predictor central canal stenosis
   • LFA preferred over LFT (encompasses full cross-sectional burden)
   • LFT higher on left; increases with age

9. BAASTRUP DISEASE (Kissing Spine)
   • Spinous process apposition → cortical sclerosis/hypertrophy
   • Interspinous bursitis (T2/STIR high signal)
   • Osseous remodeling (flattening/enlargement)
   • Epidural extension → midline epidural cysts → thecal compression
   • Most prevalent L4-L5; incidence 81% in symptomatic >80 yrs

10. FACET JOINT MORPHOMETRICS
    • Grogan MRI grades 1-4 (cartilage integrity)
    • Weishaupt CT/MRI grades 1-4 (joint space width; normal 2-4mm, mild <2mm)
    • Facet orientation angle (transverse plane vs midsagittal)
    • Facet Tropism (FT) = |angle_R - angle_L|
      Ko grade 0: ≤7° (normal), Grade 1: 7-x° (disc prolapse risk),
      Grade 2: ≥x° (spondylolisthesis risk)
    • Thresholds vary: 7°, 8°, 10° in literature
    • FT accelerates disc dehydration via shear forces

11. CERVICAL SPINE MORPHOMETRICS
    • MSCC: Modified Spinal Cord Compression (normalized AP cord diameter)
    • CSA: cross-sectional area of cord → conservative vs operative threshold
    • K-Line: C2-C7 midpoint line; K-line negative = anterior contact → surgery
    • Dice >0.90 benchmark for cervical anatomy automated systems

12. INTEGRATED CLINICAL THRESHOLDS (summary table)
    • Central Stenosis DSCA <70mm²
    • Lateral Recess Depth ≤2mm (severe)
    • Foraminal Lee Grade 3
    • LF Hypertrophy >5mm
    • Disc Degeneration Grade V / >50% height loss
    • Vertebral Deformity ratio <0.75
    • Modic Burden Grade C (>50% vertebral volume)
"""

import argparse, json, logging, traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import (binary_fill_holes, distance_transform_edt,
                           gaussian_filter, label as cc_label,
                           zoom as ndizoom)
from skimage.measure import marching_cubes

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Label maps ────────────────────────────────────────────────────────────────

SPINE_MASK_LABELS = [                              # (lbl, name, colour, op, fill)
    ( 26, 'Sacrum (spine)',          '#ff8c00', 0.72, True),
    ( 41, 'Arcus Vertebrae',         '#8855cc', 0.55, True),
    ( 42, 'Spinous Process',         '#e8c84a', 0.75, True),
    ( 43, 'TP Left  (costal 43)',    '#ff3333', 0.95, False),
    ( 44, 'TP Right (costal 44)',    '#00d4ff', 0.95, False),
    ( 45, 'Sup Articular Left',      '#66ccaa', 0.65, True),
    ( 46, 'Sup Articular Right',     '#44aa88', 0.65, True),
    ( 47, 'Inf Articular Left',      '#aaddcc', 0.60, True),
    ( 48, 'Inf Articular Right',     '#88ccbb', 0.60, True),
    ( 49, 'Vertebra Corpus Border',  '#6699cc', 0.35, True),
    ( 60, 'Spinal Cord',             '#ffe066', 0.65, False),
    ( 61, 'Spinal Canal',            '#00ffb3', 0.18, False),
    ( 62, 'Endplate',                '#c8f0c8', 0.45, True),
    (100, 'IVD (spine, all)',         '#ffcc44', 0.55, True),
]

VERIDAH_CERVICAL = {i: (f'C{i}',    '#557799', 0.20) for i in range(1, 8)}
VERIDAH_THORACIC = {i+7: (f'T{i+1}','#447766', 0.20) for i in range(12)}
VERIDAH_THORACIC[28] = ('T13', '#447766', 0.20)
VERIDAH_LUMBAR = {
    20: ('L1',              '#1e6fa8', 0.48),
    21: ('L2',              '#2389cc', 0.48),
    22: ('L3',              '#29a3e8', 0.48),
    23: ('L4',              '#52bef5', 0.50),
    24: ('L5',              '#85d4ff', 0.52),
    25: ('L6',              '#aae3ff', 0.52),
    26: ('Sacrum (vert 26)','#ff8c00', 0.62),
}
VERIDAH_IVD_BASE    = 100
VERIDAH_IVD_COLOURS = {20:'#ffe28a',21:'#ffd060',22:'#ffb830',
                        23:'#ff9900',24:'#ff7700',25:'#ff5500'}
VERIDAH_NAMES = {**{k:v[0] for k,v in VERIDAH_LUMBAR.items()},
                 **{k:v[0] for k,v in VERIDAH_CERVICAL.items()},
                 **{k:v[0] for k,v in VERIDAH_THORACIC.items()}}
LUMBAR_LABELS_ORDERED = [25, 24, 23, 22, 21, 20]

TSS_SACRUM_LABEL = 50
TSS_LABELS = [
    (  1,'TSS Cord',         '#ffe066',0.50),
    (  2,'TSS Canal',        '#00ffb3',0.14),
    ( 31,'TSS T11',          '#447766',0.25),
    ( 32,'TSS T12',          '#447766',0.25),
    ( 41,'TSS L1',           '#1e6fa8',0.25),
    ( 42,'TSS L2',           '#2389cc',0.25),
    ( 43,'TSS L3',           '#29a3e8',0.25),
    ( 44,'TSS L4',           '#52bef5',0.25),
    ( 45,'TSS L5',           '#85d4ff',0.28),
    ( 50,'TSS Sacrum',       '#ff8c00',0.65),
    ( 82,'TSS disc T11-T12', '#ffe28a',0.40),
    ( 91,'TSS disc T12-L1',  '#ffd060',0.45),
    ( 92,'TSS disc L1-L2',   '#ffb830',0.50),
    ( 93,'TSS disc L2-L3',   '#ff9900',0.50),
    ( 94,'TSS disc L3-L4',   '#ff7700',0.50),
    ( 95,'TSS disc L4-L5',   '#ff5500',0.52),
    (100,'TSS disc L5-S',    '#ff3300',0.55),
]

TP_LEFT_LABEL   = 43
TP_RIGHT_LABEL  = 44
SPINEPS_SACRUM  = 26
TP_HEIGHT_MM    = 19.0
CONTACT_DIST_MM = 2.0
ISO_MM          = 1.0    # ← isotropic target voxel size in mm

IAN_PAN_LEVELS = ['l1_l2','l2_l3','l3_l4','l4_l5','l5_s1']
IAN_PAN_LABELS = ['L1-L2','L2-L3','L3-L4','L4-L5','L5-S1']

_VALID_S3D_SYM = {'circle','circle-open','cross','diamond',
                  'diamond-open','square','square-open','x'}

# ── Morphometric clinical thresholds (PDF reference) ─────────────────────────

class VertebralThresholds:
    """Genant/QM vertebral body morphometry thresholds."""
    COMPRESSION_RATIO_BICONCAVE  = 0.80   # Hm/Ha or Hm/Hp < 0.8 → biconcave
    WEDGE_RATIO_FRACTURE         = 0.80   # Ha/Hp < 0.8 → anterior wedge
    CRUSH_RATIO_POSTERIOR        = 0.80   # Hp/Ha < 0.8 → posterior loss
    HEIGHT_RATIO_INTERVENTION    = 0.75   # <0.75 → moderate/severe fracture
    SPONDYLOLISTHESIS_MM         = 3.0    # ≥3mm sagittal translation
    SAGITTAL_TRANSLATION_NORMAL  = 4.0    # <2-4mm normal

class DiscThresholds:
    """IVD morphometric thresholds."""
    DHI_SEVERE_LOSS_PCT          = 50.0   # >50% height loss → intervention
    PFIRRMANN_INTERVENTION_GRADE = 5      # Grade V → intervention

class CanalThresholds:
    """Spinal canal stenosis thresholds."""
    DSCA_NORMAL_MM2              = 100.0  # >100 mm² normal
    DSCA_RELATIVE_LOW_MM2        = 75.0   # 75-100 mm² relative stenosis
    DSCA_ABSOLUTE_MM2            = 70.0   # <70-75 mm² absolute stenosis
    AP_NORMAL_MM                 = 12.0   # >12mm normal
    AP_RELATIVE_LOW_MM           = 10.0   # 10-12mm relative
    AP_ABSOLUTE_MM               = 7.0    # <7-10mm absolute
    AP_CRITICAL_INDIAN_MM        = 11.13  # critical symptom threshold
    AP_AI_ERROR_LOW_MM           = 0.59   # SpineLogic MAE range low
    AP_AI_ERROR_HIGH_MM          = 0.75   # SpineLogic MAE range high

class LateralRecessThresholds:
    """Lateral recess stenosis thresholds."""
    DEPTH_STENOSIS_MM            = 3.0    # LRD ≤3mm → stenosis
    HEIGHT_STENOSIS_MM           = 2.0    # LR height ≤2mm → stenosis
    NPLC_DISTANCE_MEAN_CM        = 0.7    # narrowest point 0.7±0.3cm from dura at L4-L5
    NPLC_DISTANCE_SD_CM          = 0.3

class ForaminalThresholds:
    """Neural foraminal morphometric thresholds."""
    LEE_INTERVENTION_GRADE       = 3      # Grade 3 → nerve morphologic change
    LEE_KAPPA                    = 0.81   # nearly perfect interobserver agreement
    # Normative foraminal volumes (mm³) from elliptical cylinder approximation
    VOLUME_NORMS = {
        'L1_L2': {'R': 579.92, 'L': 594.43, 'sd_R': 55, 'sd_L': 44},
        'L2_L3': {'R': 688.22, 'L': 715.87, 'sd_R': 55, 'sd_L': 48},
        'L3_L4': {'R': 761.70, 'L': 790.30, 'sd_R': 59, 'sd_L': 50},
        'L4_L5': {'R': 787.82, 'L': 809.61, 'sd_R': 29, 'sd_L': 57},
        'L5_S1': {'R': 824.24, 'L': None,   'sd_R': 68, 'sd_L': None},
    }

class LigamentumFlavumThresholds:
    """Ligamentum Flavum thickness/area thresholds."""
    LFT_NORMAL_BASELINE_MM       = 3.5    # baseline at L4-L5
    LFT_HYPERTROPHY_MM           = 4.0    # upper limit healthy adults (varies)
    LFT_SEVERE_MM                = 5.0    # >5mm significant canal encroachment
    LFA_STENOSIS_CUTOFF_MM2      = 105.90 # optimal predictor central canal stenosis

class FacetThresholds:
    """Facet joint morphometric thresholds."""
    JOINT_SPACE_NORMAL_LOW_MM    = 2.0    # Weishaupt: normal 2-4mm
    JOINT_SPACE_NORMAL_HIGH_MM   = 4.0
    JOINT_SPACE_MILD_MM          = 2.0    # narrowing <2mm → mild
    TROPISM_NORMAL_DEG           = 7.0    # Ko grade 0 ≤7°
    TROPISM_MODERATE_DEG         = 7.0    # grade 1 start
    TROPISM_SEVERE_DEG           = 10.0   # grade 2 (varies: 7°, 8°, 10° in literature)

class ModicThresholds:
    """Modic change burden thresholds."""
    GRADE_A_MAX_PCT              = 25.0   # <25% vertebral body height/volume
    GRADE_B_MAX_PCT              = 50.0   # 25-50%
    GRADE_C_MIN_PCT              = 50.0   # >50%

class CervicalThresholds:
    """Cervical spine morphometric thresholds."""
    CORD_DICE_AUTOMATED          = 0.90   # automated system benchmark
    K_LINE_CONTACT_THRESHOLD     = 0.0    # K-line negative → anterior contact

# ── Canal shape lookup per vertebral level ────────────────────────────────────

CANAL_SHAPE_BY_LEVEL = {
    'L1': ('Oval',       '85-95%'),
    'L2': ('Oval',       '90%'),
    'L3': ('Triangular', '80-95%'),
    'L4': ('Triangular', '95%'),
    'L5': ('Trefoil',    '60-65%'),
}

# ── Pfirrmann grade descriptions ──────────────────────────────────────────────

PFIRRMANN_GRADES = {
    1: {'nucleus_annulus': 'Clear',   'signal': 'Hyperintense',
        'height': 'Normal',                     'label': 'Grade I'},
    2: {'nucleus_annulus': 'Clear',   'signal': 'Hyperintense (horiz bands)',
        'height': 'Normal',                     'label': 'Grade II'},
    3: {'nucleus_annulus': 'Unclear', 'signal': 'Intermediate',
        'height': 'Normal to slight decrease',  'label': 'Grade III'},
    4: {'nucleus_annulus': 'Lost',    'signal': 'Intermediate to Hypointense',
        'height': 'Normal to moderate decrease','label': 'Grade IV'},
    5: {'nucleus_annulus': 'Lost',    'signal': 'Hypointense (Black)',
        'height': 'Collapsed',                  'label': 'Grade V'},
}

# ── Genant SQ grading ─────────────────────────────────────────────────────────

GENANT_GRADES = {
    0: 'Normal',
    1: 'Mild (20-25% height reduction)',
    2: 'Moderate (25-40% height reduction)',
    3: 'Severe (>40% height reduction)',
}

# ── Lee foraminal grading ─────────────────────────────────────────────────────

LEE_GRADES = {
    0: 'Normal — no stenosis, fat well-maintained',
    1: 'Mild — fat obliteration 2 opposing directions',
    2: 'Moderate — fat obliteration 4 directions, no nerve morphologic change',
    3: 'Severe — morphologic change/compression of nerve root, fat lost',
}

# ── Modic type descriptions ───────────────────────────────────────────────────

MODIC_TYPES = {
    1: {'T1': 'Hypointense',  'T2': 'Hyperintense',      'basis': 'Fibrovascular tissue / bone marrow edema'},
    2: {'T1': 'Hyperintense', 'T2': 'Iso/Hyperintense',  'basis': 'Fatty replacement of bone marrow'},
    3: {'T1': 'Hypointense',  'T2': 'Hypointense',       'basis': 'Extensive subchondral bone sclerosis'},
}

# ── NIfTI loading ─────────────────────────────────────────────────────────────

def load_canonical(path):
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Cannot reduce {path.name} to 3D: {data.shape}")
    return data, nii

def voxel_size_mm(nii):
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))

# ── Isotropic resampling ──────────────────────────────────────────────────────

def resample_label_vol_to_iso(label_vol, vox_mm, target_mm=ISO_MM):
    """
    Resample integer label volume to isotropic target_mm³ using
    nearest-neighbour interpolation (order=0) so label values are preserved.
    """
    zoom_factors = (vox_mm / target_mm).tolist()
    return ndizoom(label_vol.astype(np.int32), zoom_factors,
                   order=0, mode='nearest', prefilter=False).astype(np.int32)

# ── Geometry helpers — all in isotropic mm space ──────────────────────────────

def centroid_mm(iso_mask):
    coords = np.array(np.where(iso_mask))
    if coords.size == 0:
        return None
    return coords.mean(axis=1) * ISO_MM

def min_dist_mm(mask_a, mask_b):
    if not mask_a.any() or not mask_b.any():
        return float('inf'), None, None
    dt      = distance_transform_edt(~mask_b) * ISO_MM
    dist_at = np.where(mask_a, dt, np.inf)
    flat    = int(np.argmin(dist_at))
    vox_a   = np.array(np.unravel_index(flat, mask_a.shape))
    dist_mm = float(dt[tuple(vox_a)])
    cb      = np.array(np.where(mask_b))
    d2      = ((cb.T - vox_a) ** 2).sum(axis=1)
    vox_b   = cb[:, int(np.argmin(d2))]
    return dist_mm, vox_a.astype(float) * ISO_MM, vox_b.astype(float) * ISO_MM

def tp_height_mm(tp_iso):
    if not tp_iso.any():
        return 0.0
    zc = np.where(tp_iso)[2]
    return (int(zc.max()) - int(zc.min())) * ISO_MM

def inferiormost_cc(mask_iso, sac_iso=None):
    if not mask_iso.any():
        return np.zeros_like(mask_iso, dtype=bool)
    labeled, n = cc_label(mask_iso)
    if n == 1:
        return mask_iso.astype(bool)
    sac_z_min = None
    if sac_iso is not None and sac_iso.any():
        sac_z_min = int(np.where(sac_iso)[2].min())
    cc_info = []
    for i in range(1, n + 1):
        comp = (labeled == i)
        zc   = np.where(comp)[2]
        cc_info.append((float(zc.mean()), int(zc.max()), comp))
    cc_info.sort(key=lambda t: t[0])
    if sac_z_min is not None:
        cands = [c for _, zmax, c in cc_info if zmax < sac_z_min]
        if cands:
            return cands[0].astype(bool)
    return cc_info[0][2].astype(bool)

def isolate_at_z_range(mask_iso, z_lo, z_hi, margin=20):
    out = np.zeros_like(mask_iso)
    lo2 = max(0, z_lo - margin)
    hi2 = min(mask_iso.shape[2] - 1, z_hi + margin)
    out[:, :, lo2:hi2 + 1] = mask_iso[:, :, lo2:hi2 + 1]
    return out

def get_z_range(iso_mask):
    if not iso_mask.any():
        return None
    zc = np.where(iso_mask)[2]
    return int(zc.min()), int(zc.max())

# ── Morphometric computation functions ───────────────────────────────────────

def compute_vertebral_heights(vert_mask_iso):
    """
    Estimate anterior (Ha), middle (Hm), posterior (Hp) heights from a single
    vertebral body mask in isotropic mm space.

    Strategy: sample the mask in the sagittal (X) dimension.
    Anterior = front third of Y-extent, Posterior = rear third, Middle = mid.
    Heights are measured as Z-extents of the sub-column.

    Returns dict with Ha, Hm, Hp in mm (or None if mask empty).
    """
    if not vert_mask_iso.any():
        return None
    coords = np.array(np.where(vert_mask_iso))  # shape 3×N
    ymin, ymax = int(coords[1].min()), int(coords[1].max())
    y_range = ymax - ymin
    if y_range < 3:
        return None
    third = max(1, y_range // 3)
    y_ant = ymin + third
    y_mid_lo, y_mid_hi = ymin + third, ymin + 2 * third
    y_post = ymin + 2 * third

    def z_span(y_lo, y_hi):
        sub = vert_mask_iso[:, y_lo:y_hi+1, :]
        if not sub.any():
            return None
        zc = np.where(sub)[2]
        return (int(zc.max()) - int(zc.min()) + 1) * ISO_MM

    ha = z_span(ymin,     y_ant)
    hm = z_span(y_mid_lo, y_mid_hi)
    hp = z_span(y_post,   ymax)
    return {'Ha': ha, 'Hm': hm, 'Hp': hp}

def compute_height_ratios(heights):
    """
    Compute Genant-style height ratios from Ha/Hm/Hp dict.
    Returns dict with Compression, Wedge, Crush ratios and Genant grade.
    """
    if heights is None:
        return {}
    ha, hm, hp = heights.get('Ha'), heights.get('Hm'), heights.get('Hp')
    result = {}
    if ha and hm:
        result['Compression_Hm_Ha'] = hm / ha
    if hp and hm:
        result['Compression_Hm_Hp'] = hm / hp
    if ha and hp:
        result['Wedge_Ha_Hp'] = ha / hp
        result['Crush_Hp_Ha'] = hp / ha

    # Genant SQ grade (simplified from ratio)
    min_ratio = min(v for v in result.values() if v is not None) if result else 1.0
    if min_ratio >= 0.80:
        result['Genant_Grade'] = 0
        result['Genant_Label'] = GENANT_GRADES[0]
    elif min_ratio >= 0.75:
        result['Genant_Grade'] = 1
        result['Genant_Label'] = GENANT_GRADES[1]
    elif min_ratio >= 0.60:
        result['Genant_Grade'] = 2
        result['Genant_Label'] = GENANT_GRADES[2]
    else:
        result['Genant_Grade'] = 3
        result['Genant_Label'] = GENANT_GRADES[3]
    return result

def compute_disc_height_index(disc_mask_iso, sup_vert_mask, inf_vert_mask):
    """
    Compute DHI (Method 1 / Farfan) from IVD and adjacent vertebral masks.
    DHI = (Ha + Hp) / (Ds + Di) × 100
    where Ha,Hp = anterior/posterior disc heights, Ds,Di = sup/inf disc depths.
    Returns DHI float or None.
    """
    if not disc_mask_iso.any():
        return None
    coords  = np.array(np.where(disc_mask_iso))
    ymin, ymax = int(coords[1].min()), int(coords[1].max())
    zmin, zmax = int(coords[2].min()), int(coords[2].max())
    xmid = int(coords[0].mean())
    third = max(1, (ymax - ymin) // 3)

    def z_height(y_lo, y_hi):
        sub = disc_mask_iso[xmid-2:xmid+3, y_lo:y_hi+1, :]
        if not sub.any():
            return None
        zc = np.where(sub)[2]
        return (int(zc.max()) - int(zc.min()) + 1) * ISO_MM

    ha_disc = z_height(ymin, ymin + third)
    hp_disc = z_height(ymax - third, ymax)

    # Ds, Di from adjacent vertebral body Z-extents at disc level
    def vert_depth(vmask):
        if vmask is None or not vmask.any():
            return None
        sub = vmask[xmid-2:xmid+3, :, zmin:zmax+1]
        if not sub.any():
            return None
        yc = np.where(sub)[1]
        return (int(yc.max()) - int(yc.min()) + 1) * ISO_MM

    ds = vert_depth(sup_vert_mask)
    di = vert_depth(inf_vert_mask)

    if ha_disc and hp_disc and ds and di and (ds + di) > 0:
        return ((ha_disc + hp_disc) / (ds + di)) * 100.0
    return None

def compute_disc_height_method2(disc_mask_iso, vert_mask_iso):
    """
    DHI Method 2: mid-disc height / mid-vertebral body height.
    """
    if not disc_mask_iso.any() or not vert_mask_iso.any():
        return None
    xmid_d = int(np.where(disc_mask_iso)[0].mean())
    xmid_v = int(np.where(vert_mask_iso)[0].mean())

    def mid_z_height(mask, xmid):
        col = mask[max(0,xmid-2):xmid+3, :, :]
        if not col.any():
            return None
        zc = np.where(col)[2]
        return (int(zc.max()) - int(zc.min()) + 1) * ISO_MM

    hd = mid_z_height(disc_mask_iso, xmid_d)
    hv = mid_z_height(vert_mask_iso, xmid_v)
    if hd and hv and hv > 0:
        return hd / hv
    return None

def compute_canal_ap_diameter(canal_mask_iso):
    """
    Estimate anteroposterior (AP) diameter of spinal canal from canal mask.
    Measured as Y-extent of the canal in isotropic mm space.
    Returns AP diameter in mm and area estimate.
    """
    if canal_mask_iso is None or not canal_mask_iso.any():
        return None, None
    coords  = np.array(np.where(canal_mask_iso))
    xmid    = int(coords[0].mean())
    # Slice at mid-X to get axial cross-section
    axial   = canal_mask_iso[max(0,xmid-2):xmid+3, :, :]
    if not axial.any():
        return None, None
    yc = np.where(axial)[1]
    zc = np.where(axial)[2]
    ap_mm = (int(yc.max()) - int(yc.min()) + 1) * ISO_MM
    ml_mm = (int(zc.max()) - int(zc.min()) + 1) * ISO_MM
    # Approximate DSCA as ellipse: π/4 × AP × ML
    dsca_mm2 = (np.pi / 4.0) * ap_mm * ml_mm
    return ap_mm, dsca_mm2

def classify_canal_stenosis(ap_mm, dsca_mm2):
    """
    Classify central canal stenosis per clinical thresholds.
    Returns (ap_class, dsca_class) strings.
    """
    ap_class = 'N/A'
    dsca_class = 'N/A'
    if ap_mm is not None:
        if ap_mm > CanalThresholds.AP_NORMAL_MM:
            ap_class = 'Normal'
        elif ap_mm >= CanalThresholds.AP_RELATIVE_LOW_MM:
            ap_class = 'Relative Stenosis'
        elif ap_mm >= CanalThresholds.AP_ABSOLUTE_MM:
            ap_class = 'Absolute Stenosis'
        else:
            ap_class = 'Critical Stenosis'
    if dsca_mm2 is not None:
        if dsca_mm2 > CanalThresholds.DSCA_NORMAL_MM2:
            dsca_class = 'Normal'
        elif dsca_mm2 >= CanalThresholds.DSCA_RELATIVE_LOW_MM2:
            dsca_class = 'Relative Stenosis'
        else:
            dsca_class = 'Absolute Stenosis'
    return ap_class, dsca_class

def compute_cord_metrics(cord_mask_iso, canal_mask_iso):
    """
    Compute spinal cord metrics:
    - CSA: cross-sectional area of cord in mm²
    - MSCC proxy: cord AP / canal AP (normalized compression index)
    - Canal occupation ratio: cord CSA / canal CSA
    """
    if cord_mask_iso is None or not cord_mask_iso.any():
        return {}
    coords  = np.array(np.where(cord_mask_iso))
    xmid    = int(coords[0].mean())
    axial_c = cord_mask_iso[max(0,xmid-2):xmid+3, :, :]
    result  = {}
    if axial_c.any():
        yc = np.where(axial_c)[1]
        zc = np.where(axial_c)[2]
        cord_ap = (int(yc.max()) - int(yc.min()) + 1) * ISO_MM
        cord_ml = (int(zc.max()) - int(zc.min()) + 1) * ISO_MM
        cord_csa = (np.pi / 4.0) * cord_ap * cord_ml
        result['Cord_AP_mm']  = cord_ap
        result['Cord_ML_mm']  = cord_ml
        result['Cord_CSA_mm2'] = cord_csa

    if canal_mask_iso is not None and canal_mask_iso.any():
        axial_ca = canal_mask_iso[max(0,xmid-2):xmid+3, :, :]
        if axial_ca.any():
            yc2 = np.where(axial_ca)[1]
            zc2 = np.where(axial_ca)[2]
            canal_ap = (int(yc2.max()) - int(yc2.min()) + 1) * ISO_MM
            canal_ml = (int(zc2.max()) - int(zc2.min()) + 1) * ISO_MM
            canal_csa = (np.pi / 4.0) * canal_ap * canal_ml
            result['Canal_AP_mm']  = canal_ap
            result['Canal_ML_mm']  = canal_ml
            result['Canal_CSA_mm2'] = canal_csa
            if 'Cord_AP_mm' in result and canal_ap > 0:
                result['MSCC_proxy'] = result['Cord_AP_mm'] / canal_ap
            if 'Cord_CSA_mm2' in result and canal_csa > 0:
                result['Canal_Occupation_ratio'] = result['Cord_CSA_mm2'] / canal_csa
    return result

def compute_ligamentum_flavum_metrics(arcus_mask_iso, canal_mask_iso):
    """
    Proxy LF thickness from arcus (posterior arch) to canal boundary.
    Returns estimated LFT and LFA classification strings.
    This is a geometric proxy; ground-truth requires dedicated LF segmentation.
    """
    if arcus_mask_iso is None or not arcus_mask_iso.any():
        return {}
    result = {}
    # LF thickness proxy: min distance from arcus to canal posterior wall
    if canal_mask_iso is not None and canal_mask_iso.any():
        lft_proxy, _, _ = min_dist_mm(canal_mask_iso, arcus_mask_iso)
        if np.isfinite(lft_proxy):
            result['LFT_proxy_mm'] = lft_proxy
            if lft_proxy <= LigamentumFlavumThresholds.LFT_NORMAL_BASELINE_MM:
                result['LFT_class'] = 'Normal'
            elif lft_proxy <= LigamentumFlavumThresholds.LFT_SEVERE_MM:
                result['LFT_class'] = 'Hypertrophied'
            else:
                result['LFT_class'] = 'Severe — neurogenic claudication risk'
    return result

def compute_spinous_process_metrics(spinous_mask_iso):
    """
    Spinous process (Baastrup disease) metrics.
    Evaluates S-I extent of spinous processes and inter-process gap.
    Returns list of level-to-level z-gaps (proxy for apposition/contact).
    """
    if spinous_mask_iso is None or not spinous_mask_iso.any():
        return {}
    labeled, n = cc_label(spinous_mask_iso)
    if n < 2:
        return {'spinous_count': n}
    # Sort components by superior-inferior centroid
    comps = []
    for i in range(1, n + 1):
        comp = (labeled == i)
        zc   = np.where(comp)[2]
        comps.append((float(zc.mean()), int(zc.min()), int(zc.max()), comp))
    comps.sort(key=lambda t: t[0])
    gaps = []
    for i in range(len(comps) - 1):
        _, _, z_hi_cur, _ = comps[i]
        _, z_lo_nxt, _, _ = comps[i + 1]
        gap_mm = (z_lo_nxt - z_hi_cur) * ISO_MM
        gaps.append(gap_mm)
    min_gap = min(gaps) if gaps else float('inf')
    result = {
        'spinous_count':   n,
        'inter_process_gaps_mm': gaps,
        'min_inter_process_gap_mm': min_gap,
    }
    # Baastrup apposition: gap ≤ 0 (contact)
    result['baastrup_contact'] = min_gap <= 0.0
    result['baastrup_risk']    = min_gap <= 2.0
    return result

def compute_facet_tropism_proxy(sup_art_L_iso, sup_art_R_iso):
    """
    Facet tropism proxy: asymmetry between left and right superior articular
    processes (orientation angle in transverse/axial plane).
    Uses centroid offsets as a directional proxy.
    Returns angle difference estimate in degrees.
    """
    result = {}
    if sup_art_L_iso is None or not sup_art_L_iso.any():
        return result
    if sup_art_R_iso is None or not sup_art_R_iso.any():
        return result

    def facet_angle_proxy(mask):
        """Principal orientation in X-Y plane (transverse)."""
        coords = np.array(np.where(mask), dtype=float).T  # N×3
        if len(coords) < 5:
            return None
        # PCA on X-Y
        xy = coords[:, :2]
        xy -= xy.mean(axis=0)
        cov = np.cov(xy.T)
        vals, vecs = np.linalg.eigh(cov)
        principal = vecs[:, np.argmax(vals)]
        angle_rad = np.arctan2(principal[1], principal[0])
        return np.degrees(angle_rad) % 180.0

    ang_L = facet_angle_proxy(sup_art_L_iso)
    ang_R = facet_angle_proxy(sup_art_R_iso)
    if ang_L is not None and ang_R is not None:
        tropism = abs(ang_L - ang_R)
        if tropism > 90:
            tropism = 180 - tropism
        result['facet_angle_L_deg'] = ang_L
        result['facet_angle_R_deg'] = ang_R
        result['facet_tropism_deg'] = tropism
        # Ko et al. grading
        if tropism <= FacetThresholds.TROPISM_NORMAL_DEG:
            result['facet_tropism_grade'] = 'Grade 0 (normal asymmetry)'
        elif tropism < FacetThresholds.TROPISM_SEVERE_DEG:
            result['facet_tropism_grade'] = 'Grade 1 (moderate — disc prolapse risk)'
        else:
            result['facet_tropism_grade'] = 'Grade 2 (severe — spondylolisthesis risk)'
    return result

def compute_spondylolisthesis_proxy(vert_label_iso, upper_lbl, lower_lbl):
    """
    Estimate sagittal translation between two adjacent vertebral bodies.
    Compares anterior centroid Y-positions (sagittal offset) in mm.
    Returns translation_mm and classification.
    """
    result = {}
    upper = (vert_label_iso == upper_lbl)
    lower = (vert_label_iso == lower_lbl)
    if not upper.any() or not lower.any():
        return result
    def ant_centroid_y(mask):
        coords = np.array(np.where(mask))
        ymin = int(coords[1].min())
        ymax = int(coords[1].max())
        ant_zone = mask[:, ymin:ymin + max(1,(ymax-ymin)//3), :]
        if not ant_zone.any():
            return None
        return float(np.where(ant_zone)[1].mean()) * ISO_MM
    y_up = ant_centroid_y(upper)
    y_lo = ant_centroid_y(lower)
    if y_up is not None and y_lo is not None:
        trans_mm = abs(y_up - y_lo)
        result['sagittal_translation_mm'] = trans_mm
        if trans_mm >= VertebralThresholds.SPONDYLOLISTHESIS_MM:
            result['spondylolisthesis'] = f'POSITIVE ({trans_mm:.1f}mm ≥ {VertebralThresholds.SPONDYLOLISTHESIS_MM}mm)'
        else:
            result['spondylolisthesis'] = f'Negative ({trans_mm:.1f}mm < {VertebralThresholds.SPONDYLOLISTHESIS_MM}mm)'
    return result

def compute_foraminal_volume_proxy(sup_art_mask, inf_art_mask, disc_mask, level_name):
    """
    Approximate neural foraminal volume using elliptical cylinder model.
    V = π × a × b × H / 4
    a = major axis (longest sagittal distance), b = minor axis, H = AP depth.
    Compares against normative data from PDF.
    """
    result = {}
    # Use sup articular process as proxy boundary of foramen
    if sup_art_mask is None or not sup_art_mask.any():
        return result
    coords = np.array(np.where(sup_art_mask), dtype=float)
    if coords.shape[1] < 5:
        return result
    # Major axis = Z-extent (superior-inferior, craniocaudal)
    zc = coords[2]
    a  = (float(zc.max()) - float(zc.min())) * ISO_MM
    # Minor axis = Y-extent (anteroposterior)
    yc = coords[1]
    b  = (float(yc.max()) - float(yc.min())) * ISO_MM
    # Depth H = X-extent (mediolateral)
    xc = coords[0]
    H  = (float(xc.max()) - float(xc.min())) * ISO_MM
    if a > 0 and b > 0 and H > 0:
        vol = (np.pi * a * b * H) / 4.0
        result['foraminal_volume_proxy_mm3'] = vol
        # Compare to normative
        norms = ForaminalThresholds.VOLUME_NORMS.get(level_name, {})
        norm_R = norms.get('R')
        if norm_R:
            pct = (vol / norm_R) * 100.0
            result['foraminal_volume_norm_pct'] = pct
            if pct < 60:
                result['foraminal_stenosis_class'] = 'Severe (Lee Grade 3 equivalent)'
            elif pct < 80:
                result['foraminal_stenosis_class'] = 'Moderate (Lee Grade 2 equivalent)'
            elif pct < 95:
                result['foraminal_stenosis_class'] = 'Mild (Lee Grade 1 equivalent)'
            else:
                result['foraminal_stenosis_class'] = 'Normal (Lee Grade 0)'
    return result

# ── Comprehensive morphometric analysis ──────────────────────────────────────

def run_all_morphometrics(sp_iso, vert_iso, tss_iso, sac_iso):
    """
    Run all morphometric analyses on resampled isotropic volumes.
    Returns a dict of all computed morphometric values.
    """
    metrics = {}

    # ── 1. Extract key masks ───────────────────────────────────────────────────
    canal_mask  = (sp_iso == 61) if (sp_iso == 61).any() else None
    cord_mask   = (sp_iso == 60) if (sp_iso == 60).any() else None
    arcus_mask  = (sp_iso == 41) if (sp_iso == 41).any() else None
    spinous_msk = (sp_iso == 42) if (sp_iso == 42).any() else None
    sup_art_L   = (sp_iso == 45) if (sp_iso == 45).any() else None
    sup_art_R   = (sp_iso == 46) if (sp_iso == 46).any() else None
    inf_art_L   = (sp_iso == 47) if (sp_iso == 47).any() else None
    inf_art_R   = (sp_iso == 48) if (sp_iso == 48).any() else None
    endplate_msk = (sp_iso == 62) if (sp_iso == 62).any() else None
    disc_mask   = (sp_iso == 100) if (sp_iso == 100).any() else None

    # ── 2. Central canal stenosis ──────────────────────────────────────────────
    if canal_mask is not None:
        ap_mm, dsca_mm2 = compute_canal_ap_diameter(canal_mask)
        if ap_mm is not None:
            metrics['canal_AP_mm']   = ap_mm
            metrics['canal_DSCA_mm2'] = dsca_mm2
            ap_cls, dsca_cls = classify_canal_stenosis(ap_mm, dsca_mm2)
            metrics['canal_AP_class']   = ap_cls
            metrics['canal_DSCA_class'] = dsca_cls
            # Absolute stenosis flag
            metrics['canal_absolute_stenosis'] = (
                ap_mm < CanalThresholds.AP_ABSOLUTE_MM or
                (dsca_mm2 is not None and dsca_mm2 < CanalThresholds.DSCA_ABSOLUTE_MM2))

    # ── 3. Spinal cord metrics (MSCC, CSA, K-line proxy) ──────────────────────
    cord_metrics = compute_cord_metrics(cord_mask, canal_mask)
    metrics.update(cord_metrics)

    # ── 4. Ligamentum Flavum proxy ─────────────────────────────────────────────
    lf_metrics = compute_ligamentum_flavum_metrics(arcus_mask, canal_mask)
    metrics.update(lf_metrics)

    # ── 5. Spinous process / Baastrup disease ──────────────────────────────────
    if spinous_msk is not None:
        bstp = compute_spinous_process_metrics(spinous_msk)
        metrics.update({f'baastrup_{k}': v for k, v in bstp.items()})

    # ── 6. Facet tropism (proxy) ───────────────────────────────────────────────
    ft = compute_facet_tropism_proxy(sup_art_L, sup_art_R)
    metrics.update(ft)

    # ── 7. Per-vertebral-level analysis (lumbar) ───────────────────────────────
    lumbar_pairs = [(20,21,'L1','L2'), (21,22,'L2','L3'),
                    (22,23,'L3','L4'), (23,24,'L4','L5'), (24,26,'L5','S1')]
    for upper_lbl, lower_lbl, upper_name, lower_name in lumbar_pairs:
        level = f'{upper_name}_{lower_name}'
        upper_mask = (vert_iso == upper_lbl) if (vert_iso == upper_lbl).any() else None
        lower_mask = (vert_iso == lower_lbl) if (vert_iso == lower_lbl).any() else None

        # Vertebral body height ratios (upper vertebra)
        if upper_mask is not None:
            h = compute_vertebral_heights(upper_mask)
            if h:
                ratios = compute_height_ratios(h)
                for k, v in h.items():
                    metrics[f'{upper_name}_{k}_mm'] = v
                for k, v in ratios.items():
                    metrics[f'{upper_name}_{k}'] = v

        # Spondylolisthesis (upper→lower)
        if upper_mask is not None and lower_mask is not None:
            spondy = compute_spondylolisthesis_proxy(vert_iso, upper_lbl, lower_lbl)
            for k, v in spondy.items():
                metrics[f'{level}_{k}'] = v

        # Disc height index (IVD between upper and lower)
        ivd_lbl = VERIDAH_IVD_BASE + upper_lbl
        ivd_mask_v = (vert_iso == ivd_lbl) if (vert_iso == ivd_lbl).any() else disc_mask
        if ivd_mask_v is not None:
            dhi = compute_disc_height_index(ivd_mask_v, upper_mask, lower_mask)
            if dhi is not None:
                metrics[f'{level}_DHI_pct'] = dhi
                metrics[f'{level}_DHI_grade'] = (
                    'Severe (>50% loss)' if dhi < 50.0 else
                    'Moderate' if dhi < 70.0 else 'Mild' if dhi < 85.0 else 'Normal')
            # DHI Method 2
            if upper_mask is not None:
                dhi2 = compute_disc_height_method2(ivd_mask_v, upper_mask)
                if dhi2 is not None:
                    metrics[f'{level}_DHI_method2'] = dhi2

        # Foraminal volume proxy (using sup articular as boundary)
        if sup_art_L is not None:
            fv_l = compute_foraminal_volume_proxy(sup_art_L, inf_art_L, ivd_mask_v, level)
            for k, v in fv_l.items():
                metrics[f'{level}_L_{k}'] = v
        if sup_art_R is not None:
            fv_r = compute_foraminal_volume_proxy(sup_art_R, inf_art_R, ivd_mask_v, level)
            for k, v in fv_r.items():
                metrics[f'{level}_R_{k}'] = v

    # ── 8. Canal shape annotation per level ───────────────────────────────────
    for lbl, (name, _, _) in VERIDAH_LUMBAR.items():
        if (vert_iso == lbl).any() and name in CANAL_SHAPE_BY_LEVEL:
            shape, freq = CANAL_SHAPE_BY_LEVEL[name]
            metrics[f'{name}_canal_shape'] = shape
            metrics[f'{name}_canal_shape_freq'] = freq

    logger.info(f"  Morphometrics computed: {len(metrics)} parameters")
    return metrics

# ── Marching cubes → Plotly Mesh3d ───────────────────────────────────────────

def mask_to_mesh3d(iso_mask, origin_mm, name, colour, opacity,
                   smooth_sigma=1.5, fill_holes=True):
    if not iso_mask.any():
        return None
    m = binary_fill_holes(iso_mask) if fill_holes else iso_mask.copy()
    if not m.any():
        return None
    vol = gaussian_filter(m.astype(np.float32), sigma=smooth_sigma)
    vol = np.pad(vol, 1, mode='constant', constant_values=0)
    if vol.max() <= 0.5 or vol.min() >= 0.5:
        logger.debug(f"  '{name}': no 0.5 crossing — skipped")
        return None
    try:
        verts, faces, _, _ = marching_cubes(
            vol, level=0.5, spacing=(ISO_MM, ISO_MM, ISO_MM))
    except Exception as e:
        logger.warning(f"  MC failed '{name}': {e}")
        return None
    verts -= ISO_MM
    verts -= origin_mm[np.newaxis, :]
    return go.Mesh3d(
        x=verts[:,0].tolist(), y=verts[:,1].tolist(), z=verts[:,2].tolist(),
        i=faces[:,0].tolist(), j=faces[:,1].tolist(), k=faces[:,2].tolist(),
        color=colour, opacity=opacity, name=name,
        showlegend=True, flatshading=False,
        lighting=dict(ambient=0.35, diffuse=0.75,
                      specular=0.30, roughness=0.6, fresnel=0.2),
        lightposition=dict(x=100, y=200, z=150),
        hoverinfo='name', showscale=False,
    )

# ── Annotation helpers ────────────────────────────────────────────────────────

def _sym(s):
    return s if s in _VALID_S3D_SYM else 'circle'

def ruler_line(p0, p1, colour, name, width=6, dash='solid'):
    return go.Scatter3d(
        x=[p0[0],p1[0]], y=[p0[1],p1[1]], z=[p0[2],p1[2]],
        mode='lines', line=dict(color=colour, width=width, dash=dash),
        name=name, showlegend=True, hoverinfo='name')

def label_point(pos, text, colour, size=10, symbol='circle'):
    return go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text',
        marker=dict(size=size, color=colour, symbol=_sym(symbol),
                    line=dict(color='white', width=1)),
        text=[text], textposition='top center',
        textfont=dict(size=11, color=colour),
        name=text, showlegend=False, hoverinfo='text')

def midpt(a, b):
    return (np.array(a) + np.array(b)) / 2.0

def tp_height_ruler_traces(tp_iso, origin_mm, colour, side, span_mm):
    if not tp_iso.any():
        return []
    best_x, best_span = tp_iso.shape[0] // 2, 0.0
    for x in range(tp_iso.shape[0]):
        col = tp_iso[x]
        if not col.any():
            continue
        zc = np.where(col.any(axis=0))[0]
        if zc.size < 2:
            continue
        sp = (zc.max() - zc.min()) * ISO_MM
        if sp > best_span:
            best_span, best_x = sp, x
    col = tp_iso[best_x]
    if not col.any():
        return []
    zc  = np.where(col.any(axis=0))[0]
    yc  = np.where(col.any(axis=1))[0]
    z_lo, z_hi = int(zc.min()), int(zc.max())
    y_c = int(yc.mean()) if yc.size else tp_iso.shape[1] // 2
    def iv(x,y,z): return np.array([x,y,z],float)*ISO_MM - origin_mm
    p_lo = iv(best_x, y_c, z_lo)
    p_hi = iv(best_x, y_c, z_hi)
    mid  = midpt(p_lo, p_hi)
    flag = '✓' if span_mm < TP_HEIGHT_MM else f'✗ ≥{TP_HEIGHT_MM:.0f}mm→TypeI'
    lbl  = f'{side} TP: {span_mm:.1f}mm  {flag}'
    traces = [ruler_line(p_lo, p_hi, colour, f'Height ruler {side}', width=8)]
    traces.append(label_point(mid, lbl, colour, size=9, symbol='diamond'))
    off = np.array([5.,0.,0.])
    for pt in (p_lo, p_hi):
        traces.append(ruler_line(pt-off, pt+off, colour, f'Tick {side}', width=4))
    return traces

def gap_ruler_traces(tp_iso, sac_iso, origin_mm, colour, side, dist_mm):
    if not tp_iso.any() or not sac_iso.any():
        return []
    _, pt_a, pt_b = min_dist_mm(tp_iso, sac_iso)
    if pt_a is None:
        return []
    p_a = pt_a - origin_mm
    p_b = pt_b - origin_mm
    mid = midpt(p_a, p_b)
    contact = np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM
    dash    = 'dot' if contact else 'dash'
    clbl    = (f'CONTACT {dist_mm:.1f}mm→P2' if contact
               else f'Gap: {dist_mm:.1f}mm ✓')
    return [ruler_line(p_a, p_b, colour, f'Gap ruler {side}', width=5, dash=dash),
            label_point(mid, f'{side}: {clbl}', colour, size=7, symbol='square')]

def tv_plane_traces(vert_iso, tv_label, origin_mm, tv_name):
    mask = (vert_iso == tv_label)
    if not mask.any():
        return []
    zc    = np.where(mask)[2]
    z_mid = int((zc.min() + zc.max()) // 2)
    xs = np.linspace(0, vert_iso.shape[0]-1, 12)
    ys = np.linspace(0, vert_iso.shape[1]-1, 12)
    xg, yg = np.meshgrid(xs, ys)
    zg = np.full_like(xg, z_mid)
    xm = xg*ISO_MM - origin_mm[0]
    ym = yg*ISO_MM - origin_mm[1]
    zm = zg*ISO_MM - origin_mm[2]
    plane = go.Surface(
        x=xm, y=ym, z=zm,
        colorscale=[[0,'rgba(0,230,180,0.10)'],[1,'rgba(0,230,180,0.10)']],
        showscale=False, opacity=0.18,
        name=f'TV plane ({tv_name})', showlegend=True, hoverinfo='name')
    ctr = centroid_mm(mask)
    pts = ([label_point(ctr-origin_mm, f'TV: {tv_name}', '#00e6b4',
                        size=14, symbol='cross')]
           if ctr is not None else [])
    return [plane] + pts

def castellvi_contact_traces(tp_L, tp_R, sac_iso, origin_mm,
                              cls_L, cls_R, dist_L, dist_R):
    traces = []
    for tp_iso, side, dist_mm, cls in (
        (tp_L,'Left', dist_L, cls_L),(tp_R,'Right',dist_R,cls_R)):
        if not (tp_iso.any() and sac_iso.any()):
            continue
        if not (np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM):
            continue
        _, pt_a, _ = min_dist_mm(tp_iso, sac_iso)
        if pt_a is None:
            continue
        p   = pt_a - origin_mm
        col = '#ff2222' if 'III' in (cls or '') else '#ff9900'
        traces.append(go.Scatter3d(
            x=[p[0]],y=[p[1]],z=[p[2]],
            mode='markers+text',
            marker=dict(size=20,color=col,opacity=0.85,symbol='circle',
                        line=dict(color='white',width=2)),
            text=[f'{side}: {cls}'], textposition='middle right',
            textfont=dict(size=13,color=col),
            name=f'Contact {side} ({cls})',showlegend=True,hoverinfo='text'))
    return traces

def ian_pan_bar_traces(uncertainty_row, origin_mm, x_offset_mm=55):
    if uncertainty_row is None:
        return []
    confs = {lvl: uncertainty_row.get(f'{lvl}_confidence', float('nan'))
             for lvl in IAN_PAN_LEVELS}
    valid = [v for v in confs.values() if not np.isnan(v)]
    if not valid:
        return []
    max_conf = max(valid)
    max_h=40.0; bar_w=5.0; gap=2.0
    traces = []
    for i,(lvl,lbl) in enumerate(zip(IAN_PAN_LEVELS,IAN_PAN_LABELS)):
        conf = confs[lvl]
        if np.isnan(conf):
            continue
        x0 = origin_mm[0]+x_offset_mm+i*(bar_w+gap); x1=x0+bar_w
        h  = conf*max_h; z0=-max_h/2; z1=z0+h; y=0.0
        col = '#e63946' if conf==max_conf else '#457b9d'
        vx=[x0,x1,x1,x0,x0,x1,x1,x0]
        vy=[y-1,y-1,y+1,y+1,y-1,y-1,y+1,y+1]
        vz=[z0,z0,z0,z0,z1,z1,z1,z1]
        fi=[0,0,1,1,4,4,0,0,3,3,1,2]; fj=[1,3,2,5,5,7,4,3,7,2,5,6]
        fk=[2,2,5,6,6,6,5,7,6,6,6,7]
        traces.append(go.Mesh3d(x=vx,y=vy,z=vz,i=fi,j=fj,k=fk,
                                color=col,opacity=0.80,
                                name=f'Ian {lbl}: {conf:.2f}',
                                showlegend=True,flatshading=True,hoverinfo='name'))
        traces.append(go.Scatter3d(
            x=[(x0+x1)/2],y=[y],z=[z1+3],mode='text',
            text=[f'{lbl}<br>{conf:.2f}'],
            textfont=dict(size=9,color=col),showlegend=False,hoverinfo='skip'))
    return traces

def canal_stenosis_annotation_trace(canal_mask, origin_mm, ap_mm, dsca_mm2,
                                     ap_class, dsca_class):
    """3D annotation sphere at canal centroid showing stenosis grade."""
    if canal_mask is None or not canal_mask.any():
        return []
    ctr = centroid_mm(canal_mask)
    if ctr is None:
        return []
    p = ctr - origin_mm
    col = ('#ff3333' if 'Absolute' in (ap_class or '') else
           '#ff9900' if 'Relative' in (ap_class or '') else '#2dc653')
    label = f'Canal: AP={ap_mm:.1f}mm ({ap_class})\nDSCA≈{dsca_mm2:.0f}mm²'
    return [label_point(p, label, col, size=12, symbol='square')]

def spinous_annotation_traces(bstp_metrics, spinous_mask, origin_mm):
    """3D markers for Baastrup disease risk zones."""
    traces = []
    if not bstp_metrics.get('baastrup_contact') and not bstp_metrics.get('baastrup_risk'):
        return traces
    if spinous_mask is None or not spinous_mask.any():
        return traces
    ctr = centroid_mm(spinous_mask)
    if ctr is None:
        return traces
    p = ctr - origin_mm
    min_gap = bstp_metrics.get('baastrup_min_inter_process_gap_mm', float('inf'))
    if bstp_metrics.get('baastrup_contact'):
        col, txt = '#ff3333', f'Baastrup: CONTACT (gap={min_gap:.1f}mm)'
    else:
        col, txt = '#ff9900', f'Baastrup risk: gap={min_gap:.1f}mm'
    traces.append(label_point(p, txt, col, size=11, symbol='diamond'))
    return traces

def facet_tropism_traces(facet_metrics, sup_art_L, sup_art_R, origin_mm):
    """3D annotation for facet tropism grade."""
    traces = []
    tropism = facet_metrics.get('facet_tropism_deg')
    grade   = facet_metrics.get('facet_tropism_grade', '')
    if tropism is None:
        return traces
    for mask, side in ((sup_art_L, 'L'), (sup_art_R, 'R')):
        if mask is None or not mask.any():
            continue
        ctr = centroid_mm(mask)
        if ctr is None:
            continue
        p = ctr - origin_mm
        col = ('#ff3333' if 'Grade 2' in grade else
               '#ff9900' if 'Grade 1' in grade else '#2dc653')
        traces.append(label_point(p,
            f'FT {side}: {tropism:.1f}° {grade}', col, size=8, symbol='circle'))
    return traces

def spondylolisthesis_traces(spondy_metrics_by_level, vert_iso, origin_mm):
    """3D ruler and annotation for spondylolisthesis at each level."""
    traces = []
    for level, metrics in spondy_metrics_by_level.items():
        trans_mm = metrics.get('sagittal_translation_mm')
        if trans_mm is None or trans_mm < VertebralThresholds.SPONDYLOLISTHESIS_MM:
            continue
        # Find the two vertebral labels
        parts = level.split('_')
        if len(parts) < 2:
            continue
        col = '#ff4444'
        txt = f'Spondy {level}: {trans_mm:.1f}mm'
        # Put a marker near origin (no per-level centroid lookup here for brevity)
        traces.append(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='text',
            text=[txt], textfont=dict(size=11, color=col),
            showlegend=True, name=txt, hoverinfo='text'))
    return traces

# ── Study selection ───────────────────────────────────────────────────────────

def select_studies(csv_path, top_n, rank_by, valid_ids):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)
    if valid_ids is not None:
        before = len(df)
        df = df[df['study_id'].isin(valid_ids)]
        logger.info(f"Filtered to {len(df)} ({before-len(df)} excluded)")
    if rank_by not in df.columns:
        raise ValueError(f"Column '{rank_by}' not found.")
    df_s = df.sort_values(rank_by,ascending=False).reset_index(drop=True)
    top  = df_s.head(top_n)['study_id'].tolist()
    bot  = df_s.tail(top_n)['study_id'].tolist()
    seen,sel = set(),[]
    for sid in top+bot:
        if sid not in seen: sel.append(sid); seen.add(sid)
    logger.info(f"Rank={rank_by}  Top{top_n}:{top}  Bot{top_n}:{bot}")
    return sel

# ── Per-study builder ─────────────────────────────────────────────────────────

def build_3d_figure(study_id, spineps_dir, totalspine_dir,
                    smooth=1.5, lstv_result=None,
                    uncertainty_row=None, show_tss=True):

    seg_dir    = spineps_dir / 'segmentations' / study_id
    spine_path = seg_dir  / f"{study_id}_seg-spine_msk.nii.gz"
    vert_path  = seg_dir  / f"{study_id}_seg-vert_msk.nii.gz"
    tss_path   = (totalspine_dir / study_id / 'sagittal'
                  / f"{study_id}_sagittal_labeled.nii.gz")

    def _load(path, tag):
        if not path.exists():
            logger.warning(f"  Missing: {path.name}"); return None,None
        try:    return load_canonical(path)
        except Exception as e:
            logger.warning(f"  Cannot load {tag}: {e}"); return None,None

    sag_sp,  nii_ref = _load(spine_path, 'seg-spine_msk')
    sag_vert, _      = _load(vert_path,  'seg-vert_msk')
    sag_tss,  _      = _load(tss_path,   'TSS sagittal')

    if sag_sp is None:
        logger.error(f"  [{study_id}] Missing seg-spine_msk"); return None
    if sag_vert is None:
        logger.error(f"  [{study_id}] Missing seg-vert_msk");  return None

    vox_mm = voxel_size_mm(nii_ref)
    logger.info(f"  Native voxel: {np.round(vox_mm,3)}  "
                f"shape: {sag_sp.shape}  → resampling to {ISO_MM}mm isotropic")

    # ── RESAMPLE ALL VOLUMES TO ISOTROPIC 1mm³ ────────────────────────────────
    sp_iso   = resample_label_vol_to_iso(sag_sp.astype(np.int32),   vox_mm)
    vert_iso = resample_label_vol_to_iso(sag_vert.astype(np.int32), vox_mm)
    tss_iso  = (resample_label_vol_to_iso(sag_tss.astype(np.int32), vox_mm)
                if sag_tss is not None else None)
    logger.info(f"  Iso shape: {sp_iso.shape}")

    sp_labels   = set(np.unique(sp_iso).tolist())   - {0}
    vert_labels = set(np.unique(vert_iso).tolist())  - {0}
    tss_labels  = (set(np.unique(tss_iso).tolist()) - {0}
                   if tss_iso is not None else set())
    logger.info(f"  seg-spine iso labels: {sorted(sp_labels)}")
    logger.info(f"  seg-vert  iso labels: {sorted(vert_labels)}")
    if tss_iso is not None:
        logger.info(f"  TSS       iso labels: {sorted(tss_labels)}")

    # ── Origin ────────────────────────────────────────────────────────────────
    col_mask  = vert_iso > 0
    origin_mm = (centroid_mm(col_mask)
                 if col_mask.any()
                 else np.array(sp_iso.shape, float) / 2.0 * ISO_MM)
    logger.info(f"  Origin_mm: {np.round(origin_mm,1)}")

    # ── Sacrum mask ───────────────────────────────────────────────────────────
    if tss_iso is not None and (tss_iso == TSS_SACRUM_LABEL).any():
        sac_iso = (tss_iso == TSS_SACRUM_LABEL)
        logger.info("  Sacrum: TSS label 50")
    elif (sp_iso == SPINEPS_SACRUM).any():
        sac_iso = (sp_iso == SPINEPS_SACRUM)
        logger.warning("  Sacrum: fallback SPINEPS label 26")
    else:
        sac_iso = np.zeros(sp_iso.shape, bool)
        logger.warning("  Sacrum: NOT FOUND")

    # ── Transitional vertebra ─────────────────────────────────────────────────
    tv_label, tv_name = None, 'N/A'
    for cand in LUMBAR_LABELS_ORDERED:
        if cand in vert_labels:
            tv_label = cand
            tv_name  = VERIDAH_NAMES.get(cand, str(cand))
            break
    logger.info(f"  TV: {tv_name}  label={tv_label}")

    # ── TP masks ──────────────────────────────────────────────────────────────
    tp_L_full = (sp_iso == TP_LEFT_LABEL)
    tp_R_full = (sp_iso == TP_RIGHT_LABEL)
    logger.info(f"  TP-L full: {tp_L_full.sum()} vox   "
                f"TP-R full: {tp_R_full.sum()} vox")

    if tv_label is not None:
        tv_zr = get_z_range(vert_iso == tv_label)
        if tv_zr is not None:
            z_lo_tv, z_hi_tv = tv_zr
            logger.info(f"  TV z-range (iso mm): [{z_lo_tv*ISO_MM:.0f}, "
                        f"{z_hi_tv*ISO_MM:.0f}]")
            tp_L_iso = isolate_at_z_range(tp_L_full, z_lo_tv, z_hi_tv, margin=20)
            tp_R_iso = isolate_at_z_range(tp_R_full, z_lo_tv, z_hi_tv, margin=20)
            if not tp_L_iso.any():
                logger.warning("  TP-L isolation empty → full volume")
                tp_L_iso = tp_L_full
            if not tp_R_iso.any():
                logger.warning("  TP-R isolation empty → full volume")
                tp_R_iso = tp_R_full
        else:
            tp_L_iso = tp_L_full; tp_R_iso = tp_R_full
    else:
        tp_L_iso = tp_L_full; tp_R_iso = tp_R_full

    tp_L = inferiormost_cc(tp_L_iso, sac_iso if sac_iso.any() else None)
    tp_R = inferiormost_cc(tp_R_iso, sac_iso if sac_iso.any() else None)

    span_L = tp_height_mm(tp_L)
    span_R = tp_height_mm(tp_R)
    dist_L, _, _ = min_dist_mm(tp_L, sac_iso)
    dist_R, _, _ = min_dist_mm(tp_R, sac_iso)
    logger.info(f"  TP-L: {tp_L.sum()} vox  height={span_L:.1f}mm  "
                f"gap={dist_L:.1f}mm")
    logger.info(f"  TP-R: {tp_R.sum()} vox  height={span_R:.1f}mm  "
                f"gap={dist_R:.1f}mm")

    # ── Classification ────────────────────────────────────────────────────────
    castellvi='N/A'; cls_L='N/A'; cls_R='N/A'
    if lstv_result:
        castellvi = lstv_result.get('castellvi_type') or 'None'
        cls_L     = lstv_result.get('left',  {}).get('classification','N/A')
        cls_R     = lstv_result.get('right', {}).get('classification','N/A')
        det_tv    = lstv_result.get('details',{}).get('tv_name')
        if det_tv: tv_name = det_tv

    # ── RUN ALL MORPHOMETRICS ─────────────────────────────────────────────────
    logger.info("  Running comprehensive morphometric analysis...")
    morphometrics = run_all_morphometrics(sp_iso, vert_iso, tss_iso, sac_iso)

    # Extract key metrics for display
    ap_mm    = morphometrics.get('canal_AP_mm')
    dsca_mm2 = morphometrics.get('canal_DSCA_mm2')
    ap_class = morphometrics.get('canal_AP_class', 'N/A')
    dsca_cls = morphometrics.get('canal_DSCA_class', 'N/A')
    lft_prox = morphometrics.get('LFT_proxy_mm')
    lft_cls  = morphometrics.get('LFT_class', 'N/A')
    cord_csa = morphometrics.get('Cord_CSA_mm2')
    mscc     = morphometrics.get('MSCC_proxy')
    bstp_contact = morphometrics.get('baastrup_baastrup_contact', False)
    bstp_gap = morphometrics.get('baastrup_min_inter_process_gap_mm', float('inf'))
    tropism  = morphometrics.get('facet_tropism_deg')
    ft_grade = morphometrics.get('facet_tropism_grade', 'N/A')

    # ── Build traces ──────────────────────────────────────────────────────────
    traces = []

    # 1. SPINEPS seg-spine_msk
    for lbl,name,col,op,fh in SPINE_MASK_LABELS:
        if lbl not in sp_labels:
            continue
        mask = (tp_L if lbl==TP_LEFT_LABEL
                else tp_R if lbl==TP_RIGHT_LABEL
                else (sp_iso==lbl))
        if not mask.any():
            continue
        t = mask_to_mesh3d(mask, origin_mm, name, col, op,
                           smooth_sigma=smooth, fill_holes=fh)
        if t:
            traces.append(t)
            logger.info(f"    ✓ seg-spine {lbl:>3}  {name}")
        else:
            logger.warning(f"    ✗ seg-spine {lbl:>3}  {name}")

    # 2. VERIDAH vertebrae
    all_veridah = {**VERIDAH_CERVICAL, **VERIDAH_THORACIC, **VERIDAH_LUMBAR}
    for lbl,(name,col,op) in sorted(all_veridah.items()):
        if lbl not in vert_labels: continue
        t = mask_to_mesh3d(vert_iso==lbl, origin_mm, name, col, op,
                           smooth_sigma=smooth, fill_holes=True)
        if t:
            traces.append(t)
            logger.info(f"    ✓ seg-vert  {lbl:>3}  {name}")

    # 2b. VERIDAH IVD labels
    for base,col in VERIDAH_IVD_COLOURS.items():
        ivd_lbl = VERIDAH_IVD_BASE + base
        if ivd_lbl not in vert_labels: continue
        name = f'IVD below {VERIDAH_NAMES.get(base,str(base))}'
        t = mask_to_mesh3d(vert_iso==ivd_lbl, origin_mm, name, col, 0.55,
                           smooth_sigma=smooth, fill_holes=True)
        if t:
            traces.append(t)
            logger.info(f"    ✓ seg-vert  {ivd_lbl:>3}  {name}")

    # 3. TotalSpineSeg
    if show_tss and tss_iso is not None:
        for lbl,name,col,op in TSS_LABELS:
            if lbl not in tss_labels: continue
            t = mask_to_mesh3d(tss_iso==lbl, origin_mm, f'TSS {name}',
                               col, op*0.50, smooth_sigma=smooth, fill_holes=True)
            if t:
                traces.append(t)
                logger.info(f"    ✓ TSS       {lbl:>3}  {name}")

    if not any(isinstance(tr, go.Mesh3d) for tr in traces):
        logger.error(f"  [{study_id}] Zero meshes — check label maps")
        return None

    # 4. LSTV annotations (original)
    if tv_label is not None:
        traces += tv_plane_traces(vert_iso, tv_label, origin_mm, tv_name)
    traces += tp_height_ruler_traces(tp_L, origin_mm,'#ff3333','Left', span_L)
    traces += tp_height_ruler_traces(tp_R, origin_mm,'#00d4ff','Right',span_R)
    traces += gap_ruler_traces(tp_L, sac_iso, origin_mm,'#ff8800','Left', dist_L)
    traces += gap_ruler_traces(tp_R, sac_iso, origin_mm,'#00aaff','Right',dist_R)
    traces += castellvi_contact_traces(tp_L,tp_R,sac_iso,origin_mm,
                                        cls_L,cls_R,dist_L,dist_R)
    traces += ian_pan_bar_traces(uncertainty_row, origin_mm)

    # 5. NEW morphometric annotations
    canal_mask_iso  = (sp_iso == 61) if 61 in sp_labels else None
    spinous_mask_iso = (sp_iso == 42) if 42 in sp_labels else None
    sup_art_L_iso   = (sp_iso == 45) if 45 in sp_labels else None
    sup_art_R_iso   = (sp_iso == 46) if 46 in sp_labels else None

    # Canal stenosis annotation
    if canal_mask_iso is not None and ap_mm is not None:
        traces += canal_stenosis_annotation_trace(
            canal_mask_iso, origin_mm, ap_mm, dsca_mm2, ap_class, dsca_cls)

    # Baastrup annotation
    bstp_dict = {k.replace('baastrup_',''):v
                 for k,v in morphometrics.items() if k.startswith('baastrup_')}
    if bstp_dict:
        traces += spinous_annotation_traces(bstp_dict, spinous_mask_iso, origin_mm)

    # Facet tropism annotation
    ft_dict = {k:v for k,v in morphometrics.items()
               if k.startswith('facet_')}
    if ft_dict:
        traces += facet_tropism_traces(ft_dict, sup_art_L_iso, sup_art_R_iso,
                                       origin_mm)

    # ── Summary panel ─────────────────────────────────────────────────────────
    def _fmt(v): return f'{v:.1f} mm' if (v is not None and np.isfinite(v)) else 'N/A'
    def _fmt2(v): return f'{v:.1f}' if (v is not None and np.isfinite(v)) else 'N/A'
    def _fmm2(v): return f'{v:.0f} mm²' if (v is not None and np.isfinite(v)) else 'N/A'

    # Build per-level DHI summary
    dhi_lines = []
    for pair in [('L1','L2'),('L2','L3'),('L3','L4'),('L4','L5'),('L5','S1')]:
        key = f'{pair[0]}_{pair[1]}_DHI_pct'
        dhi = morphometrics.get(key)
        if dhi is not None:
            flag = ' ✗' if dhi < 50 else ''
            dhi_lines.append(f"  {pair[0]}-{pair[1]}: {dhi:.1f}%{flag}")

    # Per-level spondylolisthesis
    spondy_lines = []
    for pair in [('L1','L2'),('L2','L3'),('L3','L4'),('L4','L5'),('L5','S1')]:
        key = f'{pair[0]}_{pair[1]}_sagittal_translation_mm'
        trans = morphometrics.get(key)
        if trans is not None:
            flag = ' ✗ SPONDY' if trans >= VertebralThresholds.SPONDYLOLISTHESIS_MM else ''
            spondy_lines.append(f"  {pair[0]}-{pair[1]}: {trans:.1f}mm{flag}")

    # Vertebral height ratios
    vert_lines = []
    for vname in ['L1','L2','L3','L4','L5']:
        w = morphometrics.get(f'{vname}_Wedge_Ha_Hp')
        g = morphometrics.get(f'{vname}_Genant_Label','')
        if w is not None:
            flag = ' ✗' if w < VertebralThresholds.HEIGHT_RATIO_INTERVENTION else ''
            vert_lines.append(f"  {vname} Wedge={w:.2f}{flag} {g}")

    summary = [
        "── LSTV / Castellvi ──",
        f"TV:          {tv_name}",
        f"TP-L height: {_fmt(span_L)}  {'✗ TypeI' if span_L>=TP_HEIGHT_MM else '✓'}",
        f"TP-R height: {_fmt(span_R)}  {'✗ TypeI' if span_R>=TP_HEIGHT_MM else '✓'}",
        f"Gap L:       {_fmt(dist_L)}  {'←CONTACT' if np.isfinite(dist_L) and dist_L<=CONTACT_DIST_MM else ''}",
        f"Gap R:       {_fmt(dist_R)}  {'←CONTACT' if np.isfinite(dist_R) and dist_R<=CONTACT_DIST_MM else ''}",
        f"Class L:     {cls_L}",
        f"Class R:     {cls_R}",
        f"Castellvi:   {castellvi}",
        "",
        "── Central Canal Stenosis ──",
        f"AP diam:     {_fmt(ap_mm)} → {ap_class}",
        f"DSCA≈:       {_fmm2(dsca_mm2)} → {dsca_cls}",
        (f"  ✗ ABSOLUTE STENOSIS" if morphometrics.get('canal_absolute_stenosis') else
         f"  ✓ within limits"),
        "",
        "── Spinal Cord ──",
        f"Cord CSA:    {_fmm2(cord_csa)}",
        f"MSCC proxy:  {_fmt2(mscc)}",
        "",
        "── Ligamentum Flavum (proxy) ──",
        f"LFT proxy:   {_fmt(lft_prox)} → {lft_cls}",
        f"  Normal≤{LigamentumFlavumThresholds.LFT_NORMAL_BASELINE_MM}mm  Severe>{LigamentumFlavumThresholds.LFT_SEVERE_MM}mm",
        f"  LFA cutoff: {LigamentumFlavumThresholds.LFA_STENOSIS_CUTOFF_MM2}mm²",
        "",
        "── Disc Height Index (DHI) ──",
        "  DHI=(Ha+Hp)/(Ds+Di)×100  [Farfan]",
        "  <50%→Severe, <70%→Mod, <85%→Mild",
    ] + dhi_lines + [
        "",
        "── Vertebral Height Ratios ──",
        "  Wedge<0.80→fract  <0.75→interv",
    ] + vert_lines + [
        "",
        "── Spondylolisthesis ──",
        f"  Threshold: ≥{VertebralThresholds.SPONDYLOLISTHESIS_MM}mm",
    ] + spondy_lines + [
        "",
        "── Baastrup Disease ──",
        f"Spinous gap: {_fmt(bstp_gap if np.isfinite(bstp_gap) else None)}",
        (f"  ✗ CONTACT → cortical sclerosis risk" if bstp_contact else
         f"  ✗ Risk (gap≤2mm)" if morphometrics.get('baastrup_baastrup_risk') else
         f"  ✓ No Baastrup contact"),
        "",
        "── Facet Tropism ──",
        f"Tropism:     {_fmt2(tropism)}°",
        f"  {ft_grade}",
        f"  Normal≤{FacetThresholds.TROPISM_NORMAL_DEG}°  Severe≥{FacetThresholds.TROPISM_SEVERE_DEG}°",
        "",
        "── Canal Shapes (L1–L5) ──",
    ] + [f"  {lvl}: {CANAL_SHAPE_BY_LEVEL[lvl][0]} ({CANAL_SHAPE_BY_LEVEL[lvl][1]})"
         for lvl in ['L1','L2','L3','L4','L5']] + [
        "",
        "── Foraminal Volume Norms ──",
    ] + [f"  {lvl}: R={v['R']:.0f}mm³ L={v['L'] or 'N/A'}"
         for lvl, v in ForaminalThresholds.VOLUME_NORMS.items()
         if v['R']] + [
        "",
        "── Pfirrmann Ref ──",
        "  I=Normal  III=Unclear  V=Collapsed",
        "",
        "── Modic Change Burden ──",
        "  A<25%  B=25-50%  C>50% vert body",
        "",
        "── Cervical (if applicable) ──",
        "  MSCC: AP cord / (above+below)",
        "  K-Line neg → anterior contact",
    ]

    if tp_L.any(): summary.append(f"TP-L: {tp_L.sum()} vox "
                                   f"({tp_L.sum()*ISO_MM**3/1000:.2f}cm³)")
    if tp_R.any(): summary.append(f"TP-R: {tp_R.sum()} vox "
                                   f"({tp_R.sum()*ISO_MM**3/1000:.2f}cm³)")
    if uncertainty_row:
        for lvl,lbl in zip(IAN_PAN_LEVELS,IAN_PAN_LABELS):
            v = uncertainty_row.get(f'{lvl}_confidence',float('nan'))
            if not np.isnan(v): summary.append(f"Ian {lbl}: {v:.3f}")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f"<b>{study_id}</b>  ·  Castellvi: <b>{castellvi}</b>"
                  f"  ·  TV: <b>{tv_name}</b>"
                  f"  ·  L: <b>{cls_L}</b>  ·  R: <b>{cls_R}</b>"
                  f"  ·  Canal: <b>{ap_class}</b>"
                  f"  ·  FT: <b>{_fmt2(tropism)}°</b>"),
            font=dict(size=13,color='#e8e8f0'), x=0.01),
        paper_bgcolor='#0d0d1a', plot_bgcolor='#0d0d1a',
        scene=dict(
            bgcolor='#0d0d1a',
            xaxis=dict(title='X (mm)',showgrid=True,gridcolor='#1a1a3e',
                       showbackground=True,backgroundcolor='#0d0d1a',
                       tickfont=dict(color='#8888aa'),
                       titlefont=dict(color='#8888aa'),zeroline=False),
            yaxis=dict(title='Y (mm)',showgrid=True,gridcolor='#1a1a3e',
                       showbackground=True,backgroundcolor='#0d0d1a',
                       tickfont=dict(color='#8888aa'),
                       titlefont=dict(color='#8888aa'),zeroline=False),
            zaxis=dict(title='Z (mm)',showgrid=True,gridcolor='#1a1a3e',
                       showbackground=True,backgroundcolor='#0d0d1a',
                       tickfont=dict(color='#8888aa'),
                       titlefont=dict(color='#8888aa'),zeroline=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6,y=0.0,z=0.3),up=dict(x=0,y=0,z=1))),
        legend=dict(font=dict(color='#e8e8f0',size=10),
                    bgcolor='rgba(13,13,26,0.85)',
                    bordercolor='#2a2a4a',borderwidth=1,
                    x=0.01,y=0.98,itemsizing='constant'),
        margin=dict(l=0,r=0,t=40,b=0),
        annotations=[
            dict(text='drag=rotate · scroll=zoom · legend=toggle · dbl=isolate',
                 xref='paper',yref='paper',x=0.5,y=-0.01,
                 xanchor='center',yanchor='top',showarrow=False,
                 font=dict(size=10,color='#8888aa'),align='center'),
            dict(text='<b>Morphometric Analysis</b><br>'+'<br>'.join(summary),
                 xref='paper',yref='paper',x=0.99,y=0.98,
                 xanchor='right',yanchor='top',showarrow=False,
                 font=dict(size=10,color='#e8e8f0',family='monospace'),
                 bgcolor='rgba(13,13,26,0.88)',
                 bordercolor='#2a2a4a',borderwidth=1,align='left'),
        ])
    return (fig, castellvi, tv_name, cls_L, cls_R, span_L, span_R,
            dist_L, dist_R, morphometrics)

# ── HTML template ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D Spine — {study_id}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0d0d1a;--sf:#13132a;--bd:#2a2a4a;--tx:#e8e8f0;--mu:#6666aa;--bl:#3a86ff}}
html,body{{background:var(--bg);color:var(--tx);font-family:'JetBrains Mono',monospace;
           height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{display:flex;align-items:center;gap:9px;flex-wrap:wrap;padding:6px 12px;
        border-bottom:1px solid var(--bd);background:var(--sf);flex-shrink:0}}
h1{{font-family:'Syne',sans-serif;font-size:.86rem;font-weight:700;white-space:nowrap}}
.b{{display:inline-block;padding:2px 8px;border-radius:20px;
    font-size:.65rem;font-weight:600;letter-spacing:.05em}}
.bs{{background:#2a2a4a;color:var(--mu)}} .bc{{background:#ff8c00;color:#0d0d1a}}
.bt{{background:#1e6fa8;color:#fff}}     .bL{{background:#cc2222;color:#fff}}
.bR{{background:#006688;color:#fff}}     .bi{{background:#1a3a2a;color:#2dc653;border:1px solid #2dc653}}
.bw{{background:#553300;color:#ffaa44;border:1px solid #ffaa44}}
.be{{background:#330011;color:#ff4466;border:1px solid #ff4466}}
.tb{{display:flex;gap:5px;align-items:center;margin-left:auto}}
.tb span{{font-size:.59rem;color:var(--mu);text-transform:uppercase;letter-spacing:.08em}}
button{{background:var(--bg);border:1px solid var(--bd);color:var(--tx);
        font-family:inherit;font-size:.65rem;padding:3px 9px;border-radius:4px;cursor:pointer}}
button:hover{{background:var(--bd)}} button.on{{background:var(--bl);border-color:var(--bl);color:#fff}}
.mt{{display:flex;gap:16px;flex-wrap:wrap;align-items:center;padding:4px 12px;
     border-bottom:1px solid var(--bd);flex-shrink:0;font-size:.64rem}}
.m{{display:flex;align-items:center;gap:4px;color:var(--mu)}}
.v{{color:var(--tx);font-weight:600}} .ok{{color:#2dc653!important}}
.wn{{color:#ff8800!important}} .cr{{color:#ff3333!important}}
.mt2{{display:flex;gap:12px;flex-wrap:wrap;align-items:center;padding:3px 12px;
      border-bottom:1px solid var(--bd);flex-shrink:0;font-size:.62rem;color:var(--mu)}}
.lg{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:4px 12px;
     border-bottom:1px solid var(--bd);flex-shrink:0;font-size:.62rem}}
.li{{display:flex;align-items:center;gap:3px;color:var(--mu)}}
.sw{{width:10px;height:10px;border-radius:2px;flex-shrink:0}}
#pl{{flex:1;min-height:0}} #pl .js-plotly-plot,#pl .plot-container{{height:100%!important}}
</style></head><body>
<header><h1>3D SPINE MORPHOMETRICS</h1>
  <span class="b bs">{study_id}</span>
  <span class="b bc">Castellvi: {castellvi}</span>
  <span class="b bt">TV: {tv_name}</span>
  <span class="b bL">L: {cls_L}</span>
  <span class="b bR">R: {cls_R}</span>
  <span class="b {canal_badge_cls}">Canal: {canal_class}</span>
  <span class="b {ft_badge_cls}">FT: {ft_label}</span>
  <span class="b {bstp_badge_cls}">Baastrup: {bstp_label}</span>
  {ian_badge}
  <div class="tb"><span>View</span>
    <button onclick="sv('oblique')"   id="b-oblique"   class="on">Oblique</button>
    <button onclick="sv('lateral')"   id="b-lateral">Lat</button>
    <button onclick="sv('posterior')" id="b-posterior">Post</button>
    <button onclick="sv('anterior')"  id="b-anterior">Ant</button>
    <button onclick="sv('axial')"     id="b-axial">Axial</button>
  </div></header>
<div class="mt">
  <div class="m">TP-L <span class="v {tpl_c}">{span_L}</span></div>
  <div class="m">TP-R <span class="v {tpr_c}">{span_R}</span></div>
  <div class="m">Gap-L <span class="v {gl_c}">{gap_L}</span></div>
  <div class="m">Gap-R <span class="v {gr_c}">{gap_R}</span></div>
  <div class="m">L <span class="v">{cls_L}</span></div>
  <div class="m">R <span class="v">{cls_R}</span></div>
  <div class="m">AP <span class="v {ap_c}">{ap_display}</span></div>
  <div class="m">DSCA <span class="v {dsca_c}">{dsca_display}</span></div>
  <div class="m">LFT <span class="v {lft_c}">{lft_display}</span></div>
  <div class="m">FT <span class="v {ft_c}">{ft_display}</span></div>
  {ian_metrics}
  <div style="margin-left:auto;color:#333355;font-size:.58rem">drag=rotate·scroll=zoom·legend=toggle·dbl=isolate</div>
</div>
<div class="mt2">
  {dhi_metrics}
  {spondy_metrics}
  {vert_ratio_metrics}
</div>
<div class="lg">
  <div class="li"><div class="sw" style="background:#ff3333"></div>TP-L(43)</div>
  <div class="li"><div class="sw" style="background:#00d4ff"></div>TP-R(44)</div>
  <div class="li"><div class="sw" style="background:#ff8c00"></div>Sacrum</div>
  <div class="li"><div class="sw" style="background:#8855cc"></div>Arcus</div>
  <div class="li"><div class="sw" style="background:#e8c84a"></div>Spinous</div>
  <div class="li"><div class="sw" style="background:#66ccaa"></div>SupArt</div>
  <div class="li"><div class="sw" style="background:#aaddcc"></div>InfArt</div>
  <div class="li"><div class="sw" style="background:#ffcc44"></div>IVD(spine)</div>
  <div class="li"><div class="sw" style="background:#ffe28a"></div>IVD(vert)</div>
  <div class="li"><div class="sw" style="background:#1e6fa8;opacity:.7"></div>L1-L6</div>
  <div class="li"><div class="sw" style="background:#00ffb3;opacity:.5"></div>Canal</div>
  <div class="li"><div class="sw" style="background:#ffe066;opacity:.8"></div>Cord</div>
  <div class="li"><div class="sw" style="background:#c8f0c8"></div>Endplate</div>
  <div class="li"><div class="sw" style="background:#00e6b4;opacity:.4"></div>TV plane</div>
</div>
<div id="pl">{plotly_div}</div>
<script>
const V={{
  oblique:  {{eye:{{x:1.6,y:0.8,z:0.4}},up:{{x:0,y:0,z:1}}}},
  lateral:  {{eye:{{x:2.4,y:0.0,z:0.0}},up:{{x:0,y:0,z:1}}}},
  posterior:{{eye:{{x:0.0,y:2.4,z:0.0}},up:{{x:0,y:0,z:1}}}},
  anterior: {{eye:{{x:0.0,y:-2.4,z:0.0}},up:{{x:0,y:0,z:1}}}},
  axial:    {{eye:{{x:0.0,y:0.0,z:3.0}},up:{{x:0,y:1,z:0}}}},
}};
function sv(n){{
  const pd=document.querySelector('#pl .js-plotly-plot');
  if(!pd)return;
  Plotly.relayout(pd,{{'scene.camera.eye':V[n].eye,'scene.camera.up':V[n].up}});
  document.querySelectorAll('.tb button').forEach(b=>b.classList.remove('on'));
  const b=document.getElementById('b-'+n); if(b)b.classList.add('on');
}}
window.addEventListener('resize',()=>{{
  const pd=document.querySelector('#pl .js-plotly-plot');
  if(pd)Plotly.Plots.resize(pd);
}});
</script></body></html>"""

# ── Save HTML ─────────────────────────────────────────────────────────────────

def save_html(fig, study_id, output_dir, castellvi, tv_name, cls_L, cls_R,
              span_L, span_R, dist_L, dist_R, uncertainty_row, morphometrics):
    from plotly.io import to_html
    plotly_div = to_html(fig, full_html=False, include_plotlyjs='cdn',
                         config=dict(responsive=True,displayModeBar=True,
                                     modeBarButtonsToRemove=['toImage'],
                                     displaylogo=False))
    def _f(v):   return f'{v:.1f} mm' if (v is not None and np.isfinite(v)) else 'N/A'
    def _f2(v):  return f'{v:.1f}' if (v is not None and np.isfinite(v)) else 'N/A'
    def _fmm2(v): return f'{v:.0f}mm²' if (v is not None and np.isfinite(v)) else 'N/A'
    def _hc(v):  return 'wn' if v>=TP_HEIGHT_MM else 'ok'
    def _gc(v):  return 'cr' if (np.isfinite(v) and v<=CONTACT_DIST_MM) else 'ok'

    ap_mm    = morphometrics.get('canal_AP_mm')
    dsca_mm2 = morphometrics.get('canal_DSCA_mm2')
    ap_class = morphometrics.get('canal_AP_class','N/A')
    dsca_cls = morphometrics.get('canal_DSCA_class','N/A')
    lft_prox = morphometrics.get('LFT_proxy_mm')
    tropism  = morphometrics.get('facet_tropism_deg')
    ft_grade = morphometrics.get('facet_tropism_grade','N/A')
    bstp_contact = morphometrics.get('baastrup_baastrup_contact', False)
    bstp_risk    = morphometrics.get('baastrup_baastrup_risk', False)
    bstp_gap     = morphometrics.get('baastrup_min_inter_process_gap_mm', float('inf'))

    # Canal badge
    if 'Absolute' in ap_class:
        canal_badge_cls, canal_class = 'be', f'Absolute ({ap_class})'
    elif 'Relative' in ap_class:
        canal_badge_cls, canal_class = 'bw', f'Relative ({ap_class})'
    else:
        canal_badge_cls, canal_class = 'bi', ap_class

    # Facet tropism badge
    if tropism is not None:
        if 'Grade 2' in ft_grade:
            ft_badge_cls = 'be'
        elif 'Grade 1' in ft_grade:
            ft_badge_cls = 'bw'
        else:
            ft_badge_cls = 'bi'
        ft_label = f'{tropism:.1f}° {ft_grade.split("(")[0].strip()}'
    else:
        ft_badge_cls, ft_label = 'bs', 'N/A'

    # Baastrup badge
    if bstp_contact:
        bstp_badge_cls, bstp_label = 'be', 'CONTACT'
    elif bstp_risk:
        bstp_badge_cls, bstp_label = 'bw', f'Risk ({bstp_gap:.1f}mm)'
    else:
        bstp_badge_cls, bstp_label = 'bi', 'None'

    # AP/DSCA colour
    def _apcolor(cls):
        return 'cr' if 'Absolute' in (cls or '') else 'wn' if 'Relative' in (cls or '') else 'ok'
    def _lftcolor(v):
        if v is None: return 'ok'
        return 'cr' if v > LigamentumFlavumThresholds.LFT_SEVERE_MM else \
               'wn' if v > LigamentumFlavumThresholds.LFT_NORMAL_BASELINE_MM else 'ok'
    def _ftcolor(t):
        if t is None: return 'ok'
        return 'cr' if t >= FacetThresholds.TROPISM_SEVERE_DEG else \
               'wn' if t >= FacetThresholds.TROPISM_NORMAL_DEG else 'ok'

    ian_badge = ''; ian_metrics = ''
    if uncertainty_row:
        c = uncertainty_row.get('l5_s1_confidence', float('nan'))
        if not np.isnan(c):
            ian_badge = f'<span class="b bi">Ian L5-S1: {c:.3f}</span>'
        ian_metrics = ''.join(
            f'<div class="m">{lbl} <span class="v">'
            f'{uncertainty_row.get(f"{lvl}_confidence",float("nan")):.2f}'
            f'</span></div>'
            for lvl,lbl in zip(IAN_PAN_LEVELS,IAN_PAN_LABELS)
            if not np.isnan(uncertainty_row.get(f'{lvl}_confidence',float('nan')))
        )

    # DHI metrics row
    dhi_metrics = ''
    for pair in [('L1','L2'),('L2','L3'),('L3','L4'),('L4','L5'),('L5','S1')]:
        key = f'{pair[0]}_{pair[1]}_DHI_pct'
        dhi = morphometrics.get(key)
        if dhi is not None:
            col = 'cr' if dhi < 50 else 'wn' if dhi < 70 else 'ok'
            dhi_metrics += (f'<div class="m">DHI {pair[0]}-{pair[1]} '
                           f'<span class="v {col}">{dhi:.0f}%</span></div>')

    # Spondylolisthesis metrics row
    spondy_metrics = ''
    for pair in [('L1','L2'),('L2','L3'),('L3','L4'),('L4','L5'),('L5','S1')]:
        key = f'{pair[0]}_{pair[1]}_sagittal_translation_mm'
        trans = morphometrics.get(key)
        if trans is not None:
            col = 'cr' if trans >= VertebralThresholds.SPONDYLOLISTHESIS_MM else 'ok'
            spondy_metrics += (f'<div class="m">Spondy {pair[0]}-{pair[1]} '
                               f'<span class="v {col}">{trans:.1f}mm</span></div>')

    # Vertebral wedge ratio metrics
    vert_ratio_metrics = ''
    for vname in ['L1','L2','L3','L4','L5']:
        w = morphometrics.get(f'{vname}_Wedge_Ha_Hp')
        if w is not None:
            col = 'cr' if w < VertebralThresholds.HEIGHT_RATIO_INTERVENTION else \
                  'wn' if w < VertebralThresholds.WEDGE_RATIO_FRACTURE else 'ok'
            vert_ratio_metrics += (f'<div class="m">{vname} Wedge '
                                   f'<span class="v {col}">{w:.2f}</span></div>')

    html = HTML.format(
        study_id=study_id, castellvi=castellvi or 'N/A',
        tv_name=tv_name or 'N/A', cls_L=cls_L or 'N/A', cls_R=cls_R or 'N/A',
        canal_badge_cls=canal_badge_cls, canal_class=canal_class,
        ft_badge_cls=ft_badge_cls, ft_label=ft_label,
        bstp_badge_cls=bstp_badge_cls, bstp_label=bstp_label,
        ian_badge=ian_badge, ian_metrics=ian_metrics,
        span_L=_f(span_L), tpl_c=_hc(span_L),
        span_R=_f(span_R), tpr_c=_hc(span_R),
        gap_L=_f(dist_L),  gl_c=_gc(dist_L),
        gap_R=_f(dist_R),  gr_c=_gc(dist_R),
        ap_display=f'{_f2(ap_mm)}mm ({ap_class})' if ap_mm else 'N/A',
        ap_c=_apcolor(ap_class),
        dsca_display=_fmm2(dsca_mm2) if dsca_mm2 else 'N/A',
        dsca_c=_apcolor(dsca_cls),
        lft_display=_f(lft_prox), lft_c=_lftcolor(lft_prox),
        ft_display=f'{_f2(tropism)}°' if tropism else 'N/A',
        ft_c=_ftcolor(tropism),
        dhi_metrics=dhi_metrics,
        spondy_metrics=spondy_metrics,
        vert_ratio_metrics=vert_ratio_metrics,
        plotly_div=plotly_div)
    out = output_dir / f"{study_id}_3d_spine.html"
    out.write_text(html, encoding='utf-8')
    logger.info(f"  → {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return out

# ── Save morphometrics CSV ─────────────────────────────────────────────────────

def save_morphometrics_csv(all_morphometrics, output_dir):
    """Save all per-study morphometric results to a CSV for downstream analysis."""
    if not all_morphometrics:
        return
    df = pd.DataFrame(all_morphometrics)
    out = output_dir / 'morphometrics_all_studies.csv'
    df.to_csv(out, index=False)
    logger.info(f"  Morphometrics CSV → {out}  ({len(df)} studies, {len(df.columns)} columns)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spineps_dir',    required=True)
    ap.add_argument('--totalspine_dir', required=True)
    ap.add_argument('--output_dir',     required=True)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--study_id', default=None)
    grp.add_argument('--all',      action='store_true')
    ap.add_argument('--uncertainty_csv', default=None)
    ap.add_argument('--valid_ids',       default=None)
    ap.add_argument('--top_n',    type=int,   default=None)
    ap.add_argument('--rank_by',  default='l5_s1_confidence')
    ap.add_argument('--lstv_json', default=None)
    ap.add_argument('--smooth',    type=float, default=1.5)
    ap.add_argument('--no_tss',    action='store_true')
    ap.add_argument('--save_morphometrics_csv', action='store_true',
                    help='Save all morphometric values to CSV for downstream analysis')
    args = ap.parse_args()

    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_root = spineps_dir / 'segmentations'

    results_by_id = {}
    if args.lstv_json:
        p = Path(args.lstv_json)
        if p.exists():
            with open(p) as f:
                results_by_id = {str(r['study_id']): r for r in json.load(f)}
            logger.info(f"Loaded {len(results_by_id)} LSTV results")

    uncertainty_by_id = {}
    csv_path = Path(args.uncertainty_csv) if args.uncertainty_csv else None
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        df['study_id'] = df['study_id'].astype(str)
        uncertainty_by_id = {r['study_id']: r for r in df.to_dict('records')}
        logger.info(f"Loaded uncertainty for {len(uncertainty_by_id)} studies")

    if args.study_id:
        study_ids = [args.study_id]
    elif args.all:
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        logger.info(f"ALL mode: {len(study_ids)} studies")
    else:
        if not args.uncertainty_csv or args.top_n is None:
            ap.error("--uncertainty_csv and --top_n required unless --all/--study_id")
        valid_ids = None
        if args.valid_ids:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
        study_ids = select_studies(csv_path, args.top_n, args.rank_by, valid_ids)
        study_ids = [s for s in study_ids if (seg_root/s).is_dir()]
        logger.info(f"Selective mode: {len(study_ids)} studies")

    ok = 0
    all_morphometrics_records = []
    for sid in study_ids:
        logger.info(f"\n[{sid}]")
        try:
            out = build_3d_figure(
                study_id=sid, spineps_dir=spineps_dir,
                totalspine_dir=totalspine_dir, smooth=args.smooth,
                lstv_result=results_by_id.get(sid),
                uncertainty_row=uncertainty_by_id.get(sid),
                show_tss=not args.no_tss)
            if out is None: continue
            (fig, castellvi, tv_name, cls_L, cls_R,
             span_L, span_R, dist_L, dist_R, morphometrics) = out
            save_html(fig, sid, output_dir, castellvi, tv_name, cls_L, cls_R,
                      span_L, span_R, dist_L, dist_R,
                      uncertainty_by_id.get(sid), morphometrics)
            if args.save_morphometrics_csv:
                rec = {'study_id': sid, 'castellvi': castellvi,
                       'tv_name': tv_name,
                       'tp_L_height_mm': span_L, 'tp_R_height_mm': span_R,
                       'tp_L_gap_mm': dist_L,    'tp_R_gap_mm': dist_R,
                       'cls_L': cls_L, 'cls_R': cls_R}
                rec.update(morphometrics)
                all_morphometrics_records.append(rec)
            ok += 1
        except Exception as e:
            logger.error(f"  [{sid}] Failed: {e}")
            logger.debug(traceback.format_exc())

    if args.save_morphometrics_csv and all_morphometrics_records:
        save_morphometrics_csv(all_morphometrics_records, output_dir)

    logger.info(f"\nDone. {ok}/{len(study_ids)} HTMLs → {output_dir}")

if __name__ == '__main__':
    main()
