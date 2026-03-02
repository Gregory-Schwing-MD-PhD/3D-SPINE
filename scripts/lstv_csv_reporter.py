
#!/usr/bin/env python3
"""
lstv_csv_reporter.py — LSTV Results → Presentation-Ready CSV
=============================================================
v1.0

Converts the list of per-study classification dicts (from classify_study)
plus their AlignmentResult objects into two CSV files:

  1. lstv_per_study.csv     — one row per study, all fields
  2. lstv_alignment.csv     — one row per study, alignment-focused columns
  3. lstv_cohort_summary.csv — single-row cohort statistics

All columns are named for direct use in paper tables.
Numeric fields are rounded to a consistent precision.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional

from vertebral_alignment import AlignmentResult, CohortAlignmentStats

logger = logging.getLogger(__name__)

# ─── Column definitions ───────────────────────────────────────────────────────

PER_STUDY_COLUMNS = [
    # Identifiers
    'study_id',

    # ── Primary LSTV classification ───────────────────────────────────────────
    'lstv_detected',
    'castellvi_type',
    'lstv_phenotype',
    'phenotype_confidence',
    'pathology_score',

    # ── Alignment analysis ────────────────────────────────────────────────────
    'alignment_preferred',         # H0_aligned | H1_shifted | insufficient_data
    'alignment_confidence',        # high | moderate | low | insufficient_data
    'alignment_score_h0',          # mean Dice under H0 (TSS=ground truth)
    'alignment_score_h1',          # mean Dice under H1 (VERIDAH=ground truth)
    'alignment_score_margin',      # score_h1 - score_h0 (+ve = H1 better)
    'alignment_consistency_h1',    # fraction of pairs where H1 > H0
    'vd_l6_present',               # VERIDAH found an L6 label

    # ── Per-level H0 Dice (alignment quality under null hypothesis) ───────────
    'h0_dice_L1', 'h0_dice_L2', 'h0_dice_L3', 'h0_dice_L4', 'h0_dice_L5',

    # ── Per-level H1 Dice (alignment quality under shift hypothesis) ──────────
    'h1_dice_L1', 'h1_dice_L2', 'h1_dice_L3', 'h1_dice_L4', 'h1_dice_L5toL6',

    # ── Classification under each hypothesis ──────────────────────────────────
    'castellvi_h0',
    'castellvi_h1',
    'phenotype_h0',
    'phenotype_h1',
    'lstv_detected_h0',
    'lstv_detected_h1',
    'classification_changes_with_alignment',   # True if H0≠H1 classification

    # ── Morphometrics ─────────────────────────────────────────────────────────
    'lumbar_count_tss',
    'lumbar_count_veridah',
    'lumbar_count_consensus',
    'tv_name',
    'tv_body_shape_class',
    'tv_h_ap_ratio',
    'tv_norm_ratio_vs_l4',

    # ── Disc metrics ──────────────────────────────────────────────────────────
    'disc_above_level',
    'disc_above_dhi_pct',
    'disc_below_level',
    'disc_below_dhi_pct',
    'disc_below_grade',
    'relative_disc_ratio',

    # ── Surgical relevance ────────────────────────────────────────────────────
    'wrong_level_risk',
    'wrong_level_risk_pct',
    'nerve_root_ambiguity',
    'bertolotti_probability',

    # ── Bayesian probabilities ────────────────────────────────────────────────
    'p_sacralization',
    'p_lumbarization',
    'p_normal',

    # ── Phase 1 geometric ─────────────────────────────────────────────────────
    'left_tp_height_mm',
    'left_dist_mm',
    'left_tp_axis_deg',
    'right_tp_height_mm',
    'right_dist_mm',
    'right_tp_axis_deg',

    # ── Cross-validation ──────────────────────────────────────────────────────
    'xval_sacrum_dice',
    'xval_l5_centroid_dist_mm',
    'xval_warnings',

    # ── Ground truth (reserved) ───────────────────────────────────────────────
    'ground_truth_label',       # for future annotation
    'ground_truth_method',

    # ── Notes ─────────────────────────────────────────────────────────────────
    'alignment_summary',
    'errors',
]

ALIGNMENT_COLUMNS = [
    'study_id',
    'vd_l6_present',
    'alignment_preferred',
    'alignment_confidence',
    'alignment_score_h0',
    'alignment_score_h1',
    'alignment_score_margin',
    'alignment_consistency_h1',
    'h0_dice_L1', 'h0_dice_L2', 'h0_dice_L3', 'h0_dice_L4', 'h0_dice_L5',
    'h1_dice_L1', 'h1_dice_L2', 'h1_dice_L3', 'h1_dice_L4', 'h1_dice_L5toL6',
    'tss_labels_present',
    'vd_labels_present',
    'castellvi_h0',
    'castellvi_h1',
    'phenotype_h0',
    'phenotype_h1',
    'classification_changes_with_alignment',
    'ground_truth_label',
    'ground_truth_method',
    'alignment_summary',
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _r(v, decimals: int = 3):
    """Round a numeric value; return '' for None/nan."""
    if v is None: return ''
    try:
        f = float(v)
        import math
        if math.isnan(f) or math.isinf(f): return ''
        return round(f, decimals)
    except (TypeError, ValueError):
        return v


def _bool(v) -> str:
    if v is None: return ''
    return 'TRUE' if v else 'FALSE'


def _extract_h_dice(h_result, level_names: List[str]) -> Dict[str, str]:
    """Extract per-level Dice from a HypothesisResult into a flat dict."""
    out: Dict[str, str] = {}
    if h_result is None:
        for n in level_names:
            out[n] = ''
        return out
    for pair, col in zip(h_result.pairs, level_names):
        out[col] = _r(pair.dice, 3) if pair.valid else ''
    return out


def _flatten_study(
        r:   dict,
        ar:  Optional[AlignmentResult],
) -> Dict[str, str]:
    """Flatten one study result dict + AlignmentResult into a flat CSV row dict."""
    row: Dict[str, str] = {col: '' for col in PER_STUDY_COLUMNS}

    # ── Identifiers ────────────────────────────────────────────────────────────
    row['study_id'] = str(r.get('study_id', ''))

    # ── Primary classification ─────────────────────────────────────────────────
    row['lstv_detected']        = _bool(r.get('lstv_detected'))
    row['castellvi_type']       = str(r.get('castellvi_type') or '')
    row['pathology_score']      = _r(r.get('pathology_score'), 1)

    morpho = r.get('lstv_morphometrics') or {}
    row['lstv_phenotype']       = str(morpho.get('lstv_phenotype') or '')
    row['phenotype_confidence'] = str(morpho.get('phenotype_confidence') or '')

    # ── Alignment ──────────────────────────────────────────────────────────────
    if ar is not None:
        row['alignment_preferred']      = ar.preferred_hypothesis
        row['alignment_confidence']     = ar.confidence
        row['alignment_score_h0']       = _r(ar.score_h0, 3)
        row['alignment_score_h1']       = _r(ar.score_h1, 3)
        row['alignment_score_margin']   = _r(ar.score_margin, 3)
        row['alignment_consistency_h1'] = _r(ar.consistency_frac_h1, 3)
        row['vd_l6_present']            = _bool(ar.vd_l6_present)
        row['alignment_summary']        = ar.summary

        # Per-level H0 dice
        h0_cols  = ['h0_dice_L1', 'h0_dice_L2', 'h0_dice_L3', 'h0_dice_L4', 'h0_dice_L5']
        h0_dice  = _extract_h_dice(ar.h0, h0_cols)
        row.update(h0_dice)

        # Per-level H1 dice
        h1_cols  = ['h1_dice_L1', 'h1_dice_L2', 'h1_dice_L3', 'h1_dice_L4', 'h1_dice_L5toL6']
        h1_dice  = _extract_h_dice(ar.h1, h1_cols)
        row.update(h1_dice)

        # Ensemble downstream classifications
        row['castellvi_h0']   = str(ar.castellvi_h0 or '')
        row['castellvi_h1']   = str(ar.castellvi_h1 or '')
        row['phenotype_h0']   = str(ar.phenotype_h0  or '')
        row['phenotype_h1']   = str(ar.phenotype_h1  or '')
        row['lstv_detected_h0'] = _bool(ar.lstv_detected_h0)
        row['lstv_detected_h1'] = _bool(ar.lstv_detected_h1)

        ct_changed = (ar.castellvi_h0 != ar.castellvi_h1 or
                      ar.phenotype_h0  != ar.phenotype_h1)
        row['classification_changes_with_alignment'] = _bool(ct_changed)

        row['ground_truth_label']  = str(ar.ground_truth_label  or '')
        row['ground_truth_method'] = str(ar.ground_truth_method or '')

    else:
        row['vd_l6_present'] = ''

    # ── Morphometrics ──────────────────────────────────────────────────────────
    row['lumbar_count_tss']       = str(morpho.get('lumbar_count_tss')       or '')
    row['lumbar_count_veridah']   = str(morpho.get('lumbar_count_veridah')   or '')
    row['lumbar_count_consensus'] = str(morpho.get('lumbar_count_consensus') or '')
    row['tv_name']                = str(morpho.get('tv_name') or '')

    tv_shape = morpho.get('tv_shape') or {}
    row['tv_body_shape_class']   = str(tv_shape.get('shape_class') or '')
    row['tv_h_ap_ratio']         = _r(tv_shape.get('h_ap_ratio'), 3)
    row['tv_norm_ratio_vs_l4']   = _r(tv_shape.get('norm_ratio'),  3)

    da = morpho.get('disc_above') or {}
    db = morpho.get('disc_below') or {}
    row['disc_above_level']   = str(da.get('level') or '')
    row['disc_above_dhi_pct'] = _r(da.get('dhi_pct'), 1)
    row['disc_below_level']   = str(db.get('level') or '')
    row['disc_below_dhi_pct'] = _r(db.get('dhi_pct'), 1)
    row['disc_below_grade']   = str(db.get('grade') or '')
    row['relative_disc_ratio']= _r(morpho.get('relative_disc_ratio'), 3)

    # ── Surgical relevance ─────────────────────────────────────────────────────
    sr = morpho.get('surgical_relevance') or {}
    row['wrong_level_risk']     = str(sr.get('wrong_level_risk') or '')
    row['wrong_level_risk_pct'] = _r(sr.get('wrong_level_risk_pct'), 2)
    row['nerve_root_ambiguity'] = _bool(sr.get('nerve_root_ambiguity'))
    row['bertolotti_probability']= _r(sr.get('bertolotti_probability'), 2)

    # ── Probabilities ──────────────────────────────────────────────────────────
    probs = morpho.get('probabilities') or {}
    row['p_sacralization'] = _r(probs.get('p_sacralization'), 3)
    row['p_lumbarization'] = _r(probs.get('p_lumbarization'), 3)
    row['p_normal']        = _r(probs.get('p_normal'),        3)

    # ── Phase 1 per side ───────────────────────────────────────────────────────
    left  = r.get('left')  or {}
    right = r.get('right') or {}
    row['left_tp_height_mm']  = _r(left.get('tp_height_mm'),              1)
    row['left_dist_mm']       = _r(left.get('dist_mm'),                   1)
    row['left_tp_axis_deg']   = _r(left.get('tp_axis_deg_from_segmental'),1)
    row['right_tp_height_mm'] = _r(right.get('tp_height_mm'),             1)
    row['right_dist_mm']      = _r(right.get('dist_mm'),                  1)
    row['right_tp_axis_deg']  = _r(right.get('tp_axis_deg_from_segmental'),1)

    # ── Cross-validation ───────────────────────────────────────────────────────
    xval = r.get('cross_validation') or {}
    row['xval_sacrum_dice']          = _r(xval.get('sacrum_dice'), 3)
    row['xval_l5_centroid_dist_mm']  = _r(xval.get('l5_centroid_dist_mm'), 1)
    row['xval_warnings']             = '; '.join(xval.get('warnings') or [])

    # ── Errors ────────────────────────────────────────────────────────────────
    row['errors'] = '; '.join(r.get('errors') or [])

    return row


def _flatten_alignment_only(ar: AlignmentResult) -> Dict[str, str]:
    """Flatten an AlignmentResult into alignment-focused CSV columns."""
    row: Dict[str, str] = {col: '' for col in ALIGNMENT_COLUMNS}

    row['study_id']                   = ar.study_id
    row['vd_l6_present']              = _bool(ar.vd_l6_present)
    row['alignment_preferred']        = ar.preferred_hypothesis
    row['alignment_confidence']       = ar.confidence
    row['alignment_score_h0']         = _r(ar.score_h0,        3)
    row['alignment_score_h1']         = _r(ar.score_h1,        3)
    row['alignment_score_margin']     = _r(ar.score_margin,    3)
    row['alignment_consistency_h1']   = _r(ar.consistency_frac_h1, 3)
    row['tss_labels_present']         = str(ar.tss_labels_present)
    row['vd_labels_present']          = str(ar.vd_labels_present)
    row['castellvi_h0']               = str(ar.castellvi_h0   or '')
    row['castellvi_h1']               = str(ar.castellvi_h1   or '')
    row['phenotype_h0']               = str(ar.phenotype_h0   or '')
    row['phenotype_h1']               = str(ar.phenotype_h1   or '')
    row['classification_changes_with_alignment'] = _bool(
        ar.castellvi_h0 != ar.castellvi_h1 or ar.phenotype_h0 != ar.phenotype_h1)
    row['ground_truth_label']         = str(ar.ground_truth_label  or '')
    row['ground_truth_method']        = str(ar.ground_truth_method or '')
    row['alignment_summary']          = ar.summary

    h0_cols = ['h0_dice_L1','h0_dice_L2','h0_dice_L3','h0_dice_L4','h0_dice_L5']
    h1_cols = ['h1_dice_L1','h1_dice_L2','h1_dice_L3','h1_dice_L4','h1_dice_L5toL6']
    row.update(_extract_h_dice(ar.h0, h0_cols))
    row.update(_extract_h_dice(ar.h1, h1_cols))

    return row


# ─── Main writer ──────────────────────────────────────────────────────────────

def write_csv_reports(
        results:           List[dict],
        alignment_results: List[AlignmentResult],
        cohort_stats:      CohortAlignmentStats,
        output_dir:        Path,
) -> None:
    """
    Write all three CSV files to output_dir.

    Parameters
    ----------
    results           : list of dicts from classify_study()
    alignment_results : list of AlignmentResult, same order as results
    cohort_stats      : CohortAlignmentStats from compute_cohort_stats()
    output_dir        : destination directory (must exist)
    """
    ar_by_id: Dict[str, AlignmentResult] = {ar.study_id: ar for ar in alignment_results}

    # ── 1. Per-study full table ────────────────────────────────────────────────
    per_study_path = output_dir / 'lstv_per_study.csv'
    with open(per_study_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_STUDY_COLUMNS,
                                extrasaction='ignore')
        writer.writeheader()
        for r in results:
            sid = str(r.get('study_id', ''))
            ar  = ar_by_id.get(sid)
            row = _flatten_study(r, ar)
            writer.writerow(row)
    logger.info(f"  CSV: {per_study_path}  ({len(results)} rows)")

    # ── 2. Alignment-focused table ─────────────────────────────────────────────
    alignment_path = output_dir / 'lstv_alignment.csv'
    with open(alignment_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=ALIGNMENT_COLUMNS,
                                extrasaction='ignore')
        writer.writeheader()
        for ar in alignment_results:
            row = _flatten_alignment_only(ar)
            writer.writerow(row)
    logger.info(f"  CSV: {alignment_path}  ({len(alignment_results)} rows)")

    # ── 3. Cohort summary ─────────────────────────────────────────────────────
    summary_path = output_dir / 'lstv_cohort_summary.csv'
    stats_dict   = cohort_stats.to_dict()
    with open(summary_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(stats_dict.keys()))
        writer.writeheader()
        writer.writerow(stats_dict)
    logger.info(f"  CSV: {summary_path}")

    # ── 4. Alignment subgroup: studies with VD L6 present (most interesting) ──
    l6_studies   = [r for r in results
                    if ar_by_id.get(str(r.get('study_id', ''))) is not None
                    and ar_by_id[str(r.get('study_id', ''))].vd_l6_present]
    if l6_studies:
        l6_path = output_dir / 'lstv_l6_subgroup.csv'
        with open(l6_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=ALIGNMENT_COLUMNS,
                                    extrasaction='ignore')
            writer.writeheader()
            for r in l6_studies:
                sid = str(r.get('study_id', ''))
                ar  = ar_by_id[sid]
                writer.writerow(_flatten_alignment_only(ar))
        logger.info(f"  CSV: {l6_path}  ({len(l6_studies)} L6-present studies)")
