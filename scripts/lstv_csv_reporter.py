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

    # ── Alignment analysis (generalised offset model) ───────────────────────
    'alignment_preferred',         # aligned | shifted_plus_1 | shifted_minus_1 | ...
    'alignment_confidence',        # high | moderate | low | ambiguous | insufficient_data
    'alignment_best_offset',       # integer: 0=aligned, +1=lumbarization, -1=sacralization
    'alignment_best_score',        # mean Dice at accepted offset
    'alignment_second_best_score', # mean Dice at runner-up offset
    'alignment_score_margin',      # best - second_best (+ve = clear winner)
    'alignment_consistency_frac',  # fraction of individual pairs favouring best offset
    'alignment_consistency_beats', # numerator: pairs where best > zero
    'alignment_consistency_total', # denominator: comparable pairs
    'tss_lumbar_count',            # number of TSS lumbar labels present
    'vd_lumbar_count',             # number of VERIDAH lumbar labels present
    'vd_l6_present',               # VERIDAH found a label 25 (L6)

    # ── Per-offset Dice scores (offset -2 .. +2) ──────────────────────────────
    'dice_offset_minus2',
    'dice_offset_minus1',
    'dice_offset_0',
    'dice_offset_plus1',
    'dice_offset_plus2',

    # ── Per-level Dice at offset=0 (alignment quality at null hypothesis) ─────
    'dice0_L1', 'dice0_L2', 'dice0_L3', 'dice0_L4', 'dice0_L5',

    # ── Per-level Dice at best offset ─────────────────────────────────────────
    'best_dice_L1', 'best_dice_L2', 'best_dice_L3', 'best_dice_L4', 'best_dice_L5',

    # ── Classification at offset=0 (TSS-guided TV = primary) ──────────────────
    'castellvi_at_zero',       # Castellvi using TSS L5 as TV
    'phenotype_at_zero',       # LSTV phenotype using TSS L5 as TV
    'lstv_detected_at_zero',

    # ── Classification at best offset (alternative TV) ────────────────────────
    'castellvi_at_best',       # Castellvi using offset-guided TV
    'tv_at_best',              # which VD label was used as TV for at_best
    'phenotype_at_best',       # LSTV phenotype using offset-guided TV
    'lstv_detected_at_best',

    # ── Disagreement flag ─────────────────────────────────────────────────────
    'castellvi_disagrees',     # True when at_zero != at_best (and both not None)
    'phenotype_disagrees',

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
    'tss_lumbar_count',
    'vd_lumbar_count',
    'vd_l6_present',
    'alignment_preferred',
    'alignment_confidence',
    'alignment_best_offset',
    'alignment_best_score',
    'alignment_second_best_score',
    'alignment_score_margin',
    'alignment_consistency_frac',
    'alignment_consistency_beats',
    'alignment_consistency_total',
    'dice_offset_minus2',
    'dice_offset_minus1',
    'dice_offset_0',
    'dice_offset_plus1',
    'dice_offset_plus2',
    'dice0_L1', 'dice0_L2', 'dice0_L3', 'dice0_L4', 'dice0_L5',
    'best_dice_L1', 'best_dice_L2', 'best_dice_L3', 'best_dice_L4', 'best_dice_L5',
    'tss_labels_present',
    'vd_labels_present',
    'castellvi_at_zero',
    'castellvi_at_best',
    'tv_at_best',
    'phenotype_at_zero',
    'phenotype_at_best',
    'castellvi_disagrees',
    'phenotype_disagrees',
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
        row['alignment_preferred']         = ar.preferred_hypothesis
        row['alignment_confidence']        = ar.confidence
        row['alignment_best_offset']       = (str(ar.best_offset)
                                               if ar.best_offset is not None else '')
        row['alignment_best_score']        = _r(ar.best_score, 3)
        row['alignment_second_best_score'] = _r(ar.second_best_score, 3)
        row['alignment_score_margin']      = _r(ar.score_margin, 3)
        row['alignment_consistency_frac']  = _r(ar.consistency_frac, 3)
        row['alignment_consistency_beats'] = str(ar.consistency_beats)
        row['alignment_consistency_total'] = str(ar.consistency_total)
        row['tss_lumbar_count']            = str(ar.tss_count)
        row['vd_lumbar_count']             = str(ar.vd_count)
        row['vd_l6_present']               = _bool(25 in ar.vd_labels_present)
        row['alignment_summary']           = ar.summary

        # Per-offset mean Dice scores
        for k, col in [(-2,'dice_offset_minus2'), (-1,'dice_offset_minus1'),
                        (0,'dice_offset_0'), (1,'dice_offset_plus1'), (2,'dice_offset_plus2')]:
            s = ar.offset_scores.get(k)
            row[col] = _r(s.mean_dice if s else None, 3)

        # Per-level Dice at offset=0
        s0 = ar.offset_scores.get(0)
        for i, col in enumerate(['dice0_L1','dice0_L2','dice0_L3','dice0_L4','dice0_L5']):
            d = s0.pairs[i].dice if (s0 and i < len(s0.pairs)) else None
            row[col] = _r(d, 3)

        # Per-level Dice at best offset
        sb = ar.offset_scores.get(ar.best_offset) if ar.best_offset is not None else None
        for i, col in enumerate(['best_dice_L1','best_dice_L2','best_dice_L3',
                                  'best_dice_L4','best_dice_L5']):
            d = sb.pairs[i].dice if (sb and i < len(sb.pairs)) else None
            row[col] = _r(d, 3)

        # Classification at offset=0 (primary / TSS-guided)
        row['castellvi_at_zero']      = str(ar.castellvi_at_zero  or '')
        row['phenotype_at_zero']      = str(ar.phenotype_at_zero   or '')
        row['lstv_detected_at_zero']  = _bool(ar.lstv_detected_at_zero)

        # Classification at best offset (alternative TV)
        row['castellvi_at_best']      = str(ar.castellvi_at_best  or '')
        row['phenotype_at_best']      = str(ar.phenotype_at_best   or '')
        row['lstv_detected_at_best']  = _bool(ar.lstv_detected_at_best)

        # Compute which VD label was used as TV at best offset
        # (at_zero TV = VD label matching tss_highest_lumbar; at_best = that + offset)
        if ar.best_offset is not None and ar.best_offset != 0:
            from vertebral_alignment import VD_LUMBAR_BASE, VD_LUMBAR_MAX
            VD_NAMES = {20:'L1',21:'L2',22:'L3',23:'L4',24:'L5',25:'L6'}
            # at_zero VD = highest VD lumbar label at offset=0 that was a valid pair
            s0_valid = [p for p in (s0.pairs if s0 else []) if p.valid]
            at_zero_vd = max((p.vd_label for p in s0_valid), default=None)
            if at_zero_vd is not None:
                at_best_vd = at_zero_vd + ar.best_offset
                row['tv_at_best'] = VD_NAMES.get(at_best_vd, f'VD{at_best_vd}')
            else:
                row['tv_at_best'] = ''
        else:
            row['tv_at_best'] = ''

        # Disagreement flags
        ct_zero = ar.castellvi_at_zero;  ct_best = ar.castellvi_at_best
        ph_zero = ar.phenotype_at_zero;  ph_best = ar.phenotype_at_best
        ct_disagree = (ct_zero != ct_best and None not in (ct_zero, ct_best))
        ph_disagree = (ph_zero != ph_best and None not in (ph_zero, ph_best))
        row['castellvi_disagrees'] = _bool(ct_disagree)
        row['phenotype_disagrees'] = _bool(ph_disagree)

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
    row['tss_lumbar_count']           = str(ar.tss_count)
    row['vd_lumbar_count']            = str(ar.vd_count)
    row['vd_l6_present']              = _bool(25 in ar.vd_labels_present)
    row['alignment_preferred']        = ar.preferred_hypothesis
    row['alignment_confidence']       = ar.confidence
    row['alignment_best_offset']      = (str(ar.best_offset)
                                         if ar.best_offset is not None else '')
    row['alignment_best_score']       = _r(ar.best_score, 3)
    row['alignment_second_best_score']= _r(ar.second_best_score, 3)
    row['alignment_score_margin']     = _r(ar.score_margin, 3)
    row['alignment_consistency_frac'] = _r(ar.consistency_frac, 3)
    row['alignment_consistency_beats']= str(ar.consistency_beats)
    row['alignment_consistency_total']= str(ar.consistency_total)
    row['tss_labels_present']         = str(ar.tss_labels_present)
    row['vd_labels_present']          = str(ar.vd_labels_present)

    # Per-offset scores
    for k, col in [(-2,'dice_offset_minus2'), (-1,'dice_offset_minus1'),
                    (0,'dice_offset_0'), (1,'dice_offset_plus1'), (2,'dice_offset_plus2')]:
        s = ar.offset_scores.get(k)
        row[col] = _r(s.mean_dice if s else None, 3)

    # Per-level Dice at offset=0
    s0 = ar.offset_scores.get(0)
    for i, col in enumerate(['dice0_L1','dice0_L2','dice0_L3','dice0_L4','dice0_L5']):
        d = s0.pairs[i].dice if (s0 and i < len(s0.pairs)) else None
        row[col] = _r(d, 3)

    # Per-level Dice at best offset
    sb = ar.offset_scores.get(ar.best_offset) if ar.best_offset is not None else None
    for i, col in enumerate(['best_dice_L1','best_dice_L2','best_dice_L3',
                              'best_dice_L4','best_dice_L5']):
        d = sb.pairs[i].dice if (sb and i < len(sb.pairs)) else None
        row[col] = _r(d, 3)

    # Ensemble classifications
    row['castellvi_at_zero']   = str(ar.castellvi_at_zero  or '')
    row['castellvi_at_best']   = str(ar.castellvi_at_best  or '')
    row['phenotype_at_zero']   = str(ar.phenotype_at_zero   or '')
    row['phenotype_at_best']   = str(ar.phenotype_at_best   or '')

    # TV used for at_best classification
    VD_NAMES = {20:'L1',21:'L2',22:'L3',23:'L4',24:'L5',25:'L6'}
    if ar.best_offset is not None and ar.best_offset != 0 and s0:
        s0_valid = [p for p in s0.pairs if p.valid]
        at_zero_vd = max((p.vd_label for p in s0_valid), default=None)
        if at_zero_vd is not None:
            at_best_vd = at_zero_vd + ar.best_offset
            row['tv_at_best'] = VD_NAMES.get(at_best_vd, f'VD{at_best_vd}')

    ct_zero = ar.castellvi_at_zero; ct_best = ar.castellvi_at_best
    ph_zero = ar.phenotype_at_zero; ph_best = ar.phenotype_at_best
    row['castellvi_disagrees'] = _bool(ct_zero != ct_best
                                        and None not in (ct_zero, ct_best))
    row['phenotype_disagrees'] = _bool(ph_zero != ph_best
                                        and None not in (ph_zero, ph_best))
    row['ground_truth_label']  = str(ar.ground_truth_label  or '')
    row['ground_truth_method'] = str(ar.ground_truth_method or '')
    row['alignment_summary']   = ar.summary

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
                    and (25 in ar_by_id[str(r.get('study_id', ''))].vd_labels_present)]
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
