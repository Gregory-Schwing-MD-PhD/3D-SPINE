#!/usr/bin/env python3
"""
vertebral_alignment.py — Rigorous TSS / VERIDAH Vertebral Alignment Analyser
=============================================================================
v1.0

PURPOSE
-------
Determine whether TSS (TotalSpineSeg) and VERIDAH (SPINEPS seg-vert_msk) label
the same vertebral levels with the same indices, or whether one is shifted by
exactly one level relative to the other ("off-by-one error").

This is CRITICAL for LSTV classification because:
  • TSS labels lumbar vertebrae L1–L5 (integer labels 41–45).
  • VERIDAH labels lumbar vertebrae L1–L6 (integer labels 20–25).
  • When VERIDAH detects an "L6", this may represent:
      (A) TRUE lumbarization — a genuine 6th mobile lumbar segment.
      (B) OFF-BY-ONE labelling shift — VERIDAH has counted one more segment
          than TSS because TSS has incorporated the uppermost sacral segment
          into its sacrum label (label 50), while VERIDAH has split it out
          as "L6" (label 25) or "L5" (label 24).

DETECTION METHODOLOGY — Sequence Dice Alignment Scoring
--------------------------------------------------------
We do NOT use a single centroid distance threshold (which is fragile and
magnitude-dependent). Instead we test two explicit hypotheses across the
ENTIRE lumbar vertebral sequence:

  H0 (ALIGNED): VD_Lk corresponds to TSS_Lk for k = 1..5
      Pairs tested: (TSS 41, VD 20), (TSS 42, VD 21), (TSS 43, VD 22),
                    (TSS 44, VD 23), (TSS 45, VD 24)

  H1 (SHIFTED +1): VERIDAH is shifted one level cranially relative to TSS.
      VD_Lk corresponds to TSS_L(k+1), i.e. VERIDAH has an extra caudal level.
      Pairs tested: (TSS 41, VD 21), (TSS 42, VD 22), (TSS 43, VD 23),
                    (TSS 44, VD 24), (TSS 45, VD 25-or-nothing)

For each hypothesis the "alignment score" is the mean Dice coefficient across
ALL testable vertebral pairs where both masks are non-empty. A minimum of
MIN_PAIRS_REQUIRED pairs must be available to call a hypothesis.

The preferred hypothesis is the one with the higher alignment score.
H1 is accepted (off-by-one declared) only when:
  1. score_H1 > score_H0 + SCORE_MARGIN  (H1 clearly beats H0 globally)
  2. At least MIN_PAIRS_REQUIRED pairs available for H1
  3. VD_L6 (label 25) must be present (otherwise there is nothing to shift into)
  4. Per-level consistency check: at least CONSISTENCY_FRAC of individual
     H1 pairs must beat their corresponding H0 pair Dice. This guards against
     a single very-high-overlap pair dragging up the H1 mean.

ENSEMBLE OUTPUT
---------------
Both hypotheses are always evaluated and reported. Downstream callers receive:
  • preferred_hypothesis: 'H0_aligned' | 'H1_shifted'
  • alignment_scores: {H0: float, H1: float}
  • per_level_dice: {H0: {level: dice}, H1: {level: dice}}
  • classification_h0: Castellvi/LSTV result assuming TSS is ground truth
  • classification_h1: Castellvi/LSTV result assuming VERIDAH is ground truth
  • confidence: 'high' | 'moderate' | 'low' | 'insufficient_data'

This module is the stepping stone toward a YOLO/CNN ground-truth pipeline that
will use axial nerve morphology and iliolumbar ligament landmarks to provide
scan-level ground truth for training.

FUTURE EXTENSIBILITY
--------------------
  • ground_truth_label field is reserved for prospective annotation
  • All per-level Dice values are exported for feature engineering
  • The AlignmentResult dataclass is JSON-serialisable via .to_dict()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Label constants (must match lstv_engine.py) ───────────────────────────────
TSS_LUMBAR_LABELS : Dict[int, str] = {41: 'L1', 42: 'L2', 43: 'L3', 44: 'L4', 45: 'L5'}
TSS_SACRUM_LABEL  = 50

VD_LUMBAR_LABELS  : Dict[int, str] = {20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6'}
VD_SACRUM_LABEL   = 26

# ── Sequence pairing definitions ──────────────────────────────────────────────
# Each entry: (tss_label, vd_label, level_name)
H0_PAIRS: List[Tuple[int, int, str]] = [
    (41, 20, 'L1'), (42, 21, 'L2'), (43, 22, 'L3'), (44, 23, 'L4'), (45, 24, 'L5'),
]
# H1: VERIDAH shifted +1 caudally (VERIDAH has an extra vertebra below TSS L5)
H1_PAIRS: List[Tuple[int, int, str]] = [
    (41, 21, 'L1'), (42, 22, 'L2'), (43, 23, 'L3'), (44, 24, 'L4'), (45, 25, 'L5→L6'),
]

# ── Decision thresholds ───────────────────────────────────────────────────────
# Minimum Dice margin by which H1 must beat H0 to be accepted.
# This guards against noise: a tie or marginal difference does NOT overturn TSS.
SCORE_MARGIN       : float = 0.08   # H1 mean Dice must exceed H0 by ≥8 pp
MIN_PAIRS_REQUIRED : int   = 3      # Need at least 3 valid pairs to score a hypothesis
CONSISTENCY_FRAC   : float = 0.60   # ≥60% of individual H1 pairs must beat their H0 counterpart
MIN_PAIR_DICE      : float = 0.10   # Pairs below this are considered "not present" for scoring


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PairDice:
    """Dice overlap between one TSS and one VERIDAH vertebral mask."""
    tss_label:   int
    vd_label:    int
    level_name:  str
    dice:        Optional[float]  # None if either mask is empty
    tss_voxels:  int
    vd_voxels:   int
    overlap_voxels: int
    valid: bool  # True when both masks non-empty


@dataclass
class HypothesisResult:
    """Summary statistics for one alignment hypothesis."""
    name:             str   # 'H0_aligned' or 'H1_shifted'
    pairs:            List[PairDice]
    n_valid_pairs:    int
    mean_dice:        Optional[float]
    min_dice:         Optional[float]
    max_dice:         Optional[float]
    std_dice:         Optional[float]
    sufficient_data:  bool


@dataclass
class AlignmentResult:
    """
    Full output of the TSS/VERIDAH alignment analysis for one study.

    preferred_hypothesis:
        'H0_aligned'  — TSS and VERIDAH agree; off-by-one NOT detected.
        'H1_shifted'  — VERIDAH is shifted +1 level relative to TSS;
                        likely lumbarization FP in VERIDAH.
        'insufficient_data' — fewer than MIN_PAIRS_REQUIRED valid pairs.

    confidence:
        'high'       — score margin ≥ 2× SCORE_MARGIN
        'moderate'   — score margin ≥ SCORE_MARGIN
        'low'        — score margin < SCORE_MARGIN (borderline)
        'insufficient_data'

    ground_truth_label:
        Reserved for prospective annotation (YOLO/nerve morphology pipeline).
        None until annotated.
    """
    study_id:             str
    preferred_hypothesis: str          # 'H0_aligned' | 'H1_shifted' | 'insufficient_data'
    confidence:           str          # 'high' | 'moderate' | 'low' | 'insufficient_data'
    score_h0:             Optional[float]
    score_h1:             Optional[float]
    score_margin:         Optional[float]   # score_h1 - score_h0 (positive → H1 better)
    consistency_frac_h1:  Optional[float]   # fraction of H1 pairs beating H0

    h0:  HypothesisResult
    h1:  HypothesisResult

    vd_l6_present:        bool
    tss_labels_present:   List[int]
    vd_labels_present:    List[int]

    # Downstream classification under each hypothesis
    # Populated externally (by classify_study) after this module runs
    castellvi_h0:         Optional[str] = None
    castellvi_h1:         Optional[str] = None
    phenotype_h0:         Optional[str] = None
    phenotype_h1:         Optional[str] = None
    lstv_detected_h0:     Optional[bool] = None
    lstv_detected_h1:     Optional[bool] = None

    # Reserved for prospective ground truth annotation
    ground_truth_label:   Optional[str] = None   # 'H0_correct' | 'H1_correct' | 'ambiguous'
    ground_truth_method:  Optional[str] = None   # 'nerve_morphology' | 'iliolumbar_lig' | 'full_spine_ct'

    # Human-readable summary
    summary:              str = ''

    def to_dict(self) -> dict:
        return asdict(self)


# ═════════════════════════════════════════════════════════════════════════════
# CORE COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def _dice(a: np.ndarray, b: np.ndarray) -> Tuple[float, int, int, int]:
    """
    Compute Dice coefficient between two boolean masks.
    Returns (dice, n_a, n_b, n_overlap).
    """
    a = a.astype(bool)
    b = b.astype(bool)
    n_a       = int(a.sum())
    n_b       = int(b.sum())
    n_overlap = int((a & b).sum())
    denom     = n_a + n_b
    dice      = float(2 * n_overlap / denom) if denom > 0 else float('nan')
    return dice, n_a, n_b, n_overlap


def _score_pairs(
        sag_tss:  np.ndarray,
        sag_vert: np.ndarray,
        pairs:    List[Tuple[int, int, str]],
) -> HypothesisResult:
    """
    Compute per-pair and summary Dice for a list of (tss_label, vd_label, name) pairs.
    """
    pair_results: List[PairDice] = []

    for tss_lbl, vd_lbl, name in pairs:
        tss_mask = (sag_tss  == tss_lbl)
        vd_mask  = (sag_vert == vd_lbl)

        if not tss_mask.any() or not vd_mask.any():
            pair_results.append(PairDice(
                tss_label=tss_lbl, vd_label=vd_lbl, level_name=name,
                dice=None, tss_voxels=int(tss_mask.sum()),
                vd_voxels=int(vd_mask.sum()), overlap_voxels=0, valid=False))
            continue

        d, n_t, n_v, n_o = _dice(tss_mask, vd_mask)
        pair_results.append(PairDice(
            tss_label=tss_lbl, vd_label=vd_lbl, level_name=name,
            dice=round(d, 4), tss_voxels=n_t, vd_voxels=n_v,
            overlap_voxels=n_o, valid=True))

    valid_dice = [p.dice for p in pair_results
                  if p.valid and p.dice is not None and p.dice >= MIN_PAIR_DICE]
    n_valid    = len(valid_dice)
    sufficient = n_valid >= MIN_PAIRS_REQUIRED

    return HypothesisResult(
        name            = '',  # filled by caller
        pairs           = pair_results,
        n_valid_pairs   = n_valid,
        mean_dice       = round(float(np.mean(valid_dice)), 4)  if valid_dice else None,
        min_dice        = round(float(np.min(valid_dice)),  4)  if valid_dice else None,
        max_dice        = round(float(np.max(valid_dice)),  4)  if valid_dice else None,
        std_dice        = round(float(np.std(valid_dice)),  4)  if len(valid_dice) > 1 else None,
        sufficient_data = sufficient,
    )


def _consistency_fraction(h0_result: HypothesisResult,
                           h1_result: HypothesisResult) -> Tuple[float, int, int]:
    """
    Fraction of vertebral pairs where H1 Dice > H0 Dice.
    Pairs are matched by position (same ordinal index in the pair list).
    A pair where H0 is invalid (empty mask) and H1 is valid counts as H1 winning.

    Returns (fraction, beats, total) so callers can report "N/M pairs" correctly
    rather than back-calculating from fraction * h0.n_valid_pairs (which is wrong
    when H0 has zero valid pairs but H1 has all five).
    """
    beats = 0
    total = 0
    for p0, p1 in zip(h0_result.pairs, h1_result.pairs):
        # Case 1: both valid — compare numerically
        if p0.valid and p1.valid and p0.dice is not None and p1.dice is not None:
            total += 1
            if p1.dice > p0.dice:
                beats += 1
        # Case 2: H0 invalid, H1 valid — H1 unambiguously wins this pair
        elif not p0.valid and p1.valid and p1.dice is not None and p1.dice >= MIN_PAIR_DICE:
            total += 1
            beats += 1
    frac = float(beats / total) if total > 0 else 0.0
    return frac, beats, total


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def analyse_vertebral_alignment(
        study_id: str,
        sag_tss:  np.ndarray,
        sag_vert: np.ndarray,
) -> AlignmentResult:
    """
    Perform full TSS/VERIDAH alignment analysis for one study.

    Parameters
    ----------
    study_id : str
    sag_tss  : 3D int array — TotalSpineSeg sagittal label volume
    sag_vert : 3D int array — VERIDAH (SPINEPS seg-vert_msk) label volume

    Returns
    -------
    AlignmentResult  (fully populated, JSON-serialisable via .to_dict())
    """
    tss_labels_present  = sorted(int(v) for v in np.unique(sag_tss)
                                  if int(v) in TSS_LUMBAR_LABELS)
    vd_labels_present   = sorted(int(v) for v in np.unique(sag_vert)
                                  if int(v) in VD_LUMBAR_LABELS)
    vd_l6_present       = (25 in vd_labels_present)

    # ── Score both hypotheses ──────────────────────────────────────────────────
    h0 = _score_pairs(sag_tss, sag_vert, H0_PAIRS)
    h1 = _score_pairs(sag_tss, sag_vert, H1_PAIRS)
    h0.name = 'H0_aligned'
    h1.name = 'H1_shifted'

    consistency, consistency_beats, consistency_total = (
        _consistency_fraction(h0, h1) if h1.sufficient_data else (0.0, 0, 0)
    )

    score_margin = None
    if h0.mean_dice is not None and h1.mean_dice is not None:
        score_margin = round(float(h1.mean_dice - h0.mean_dice), 4)
    elif h1.mean_dice is not None and not h0.sufficient_data and h1.sufficient_data:
        # Entire sequence is shifted: H0 has NO valid overlapping pairs at all.
        # This is the maximally clear off-by-one: treat effective margin = h1.mean_dice.
        score_margin = round(float(h1.mean_dice), 4)

    # ── Decision logic ─────────────────────────────────────────────────────────
    # H1 (off-by-one) is accepted ONLY if ALL four conditions hold:
    #   C1. VD L6 (label 25) is present — there must be something to shift into
    #   C2. H1 has sufficient data (≥ MIN_PAIRS_REQUIRED valid pairs)
    #   C3. H1 mean Dice exceeds H0 mean Dice by ≥ SCORE_MARGIN, OR H0 is
    #       entirely absent (all sequence pairs have zero overlap under H0)
    #   C4. ≥ CONSISTENCY_FRAC of individual H1 pairs beat their H0 counterpart

    if not h0.sufficient_data and not h1.sufficient_data:
        preferred    = 'insufficient_data'
        confidence   = 'insufficient_data'
        summary_text = (f"Insufficient valid vertebral pairs for alignment analysis "
                        f"(H0={h0.n_valid_pairs} pairs, H1={h1.n_valid_pairs} pairs, "
                        f"need ≥{MIN_PAIRS_REQUIRED})")

    elif (vd_l6_present
          and h1.sufficient_data
          and score_margin is not None
          and score_margin >= SCORE_MARGIN
          and consistency  >= CONSISTENCY_FRAC):
        preferred  = 'H1_shifted'
        margin_abs = abs(score_margin)
        confidence = ('high'     if margin_abs >= 2 * SCORE_MARGIN else
                      'moderate' if margin_abs >= SCORE_MARGIN      else 'low')
        h0_score_str2 = f"{h0.mean_dice:.3f}" if h0.mean_dice is not None else "N/A"
        summary_text = (
            f"OFF-BY-ONE DETECTED [{confidence}]: "
            f"H1 (VERIDAH shifted +1) alignment score {h1.mean_dice:.3f} "
            f"exceeds H0 (aligned) {h0_score_str2} by {score_margin:+.3f}. "
            f"Pair-level consistency = {consistency:.0%} "
            f"({consistency_beats}/{consistency_total} pairs H1>H0). "
            f"VERIDAH L6 (label 25) present. "
            f"Interpretation: VERIDAH has labelled an extra caudal segment "
            f"as 'L6'; TSS subsumes this segment into its sacrum (label 50). "
            f"True anatomy likely LUMBARIZATION — but ground truth required.")

    else:
        preferred  = 'H0_aligned'
        if score_margin is not None:
            margin_abs = abs(score_margin)
            confidence = ('high'     if h0.mean_dice is not None and h0.mean_dice > 0.55 else
                          'moderate' if h0.mean_dice is not None and h0.mean_dice > 0.35 else
                          'low')
        else:
            confidence = 'low'

        reasons = []
        if not vd_l6_present:
            reasons.append("VD L6 absent (no extra caudal segment)")
        if not h1.sufficient_data:
            reasons.append(f"H1 insufficient pairs ({h1.n_valid_pairs})")
        if score_margin is not None and score_margin < SCORE_MARGIN:
            reasons.append(f"margin {score_margin:+.3f} < threshold {SCORE_MARGIN}")
        if consistency < CONSISTENCY_FRAC:
            reasons.append(f"pair consistency {consistency:.0%} < {CONSISTENCY_FRAC:.0%}")

        h0_score_str = f"{h0.mean_dice:.3f}" if h0.mean_dice is not None else "N/A"
        h1_score_str = f"{h1.mean_dice:.3f}" if h1.mean_dice is not None else "N/A"
        summary_text = (
            f"ALIGNED [H0, {confidence}]: "
            f"H0 score {h0_score_str}, H1 score {h1_score_str}. "
            f"H1 not accepted: {'; '.join(reasons) if reasons else 'H0 clearly preferred'}.")

    logger.info(f"  [{study_id}] Alignment: {summary_text}")

    # ── Per-level Dice logging ─────────────────────────────────────────────────
    logger.info(f"  [{study_id}] Vertebral Dice matrix:")
    for p0, p1 in zip(h0.pairs, h1.pairs):
        d0_str = f"{p0.dice:.3f}" if p0.dice is not None else " --- "
        d1_str = f"{p1.dice:.3f}" if p1.dice is not None else " --- "
        winner = "H1>" if (p0.dice and p1.dice and p1.dice > p0.dice) else "   "
        logger.info(
            f"    TSS {p0.tss_label} ({TSS_LUMBAR_LABELS.get(p0.tss_label,'?')}): "
            f"H0→VD{p0.vd_label}={d0_str}  "
            f"H1→VD{p1.vd_label}={d1_str}  {winner}")

    return AlignmentResult(
        study_id             = study_id,
        preferred_hypothesis = preferred,
        confidence           = confidence,
        score_h0             = h0.mean_dice,
        score_h1             = h1.mean_dice,
        score_margin         = score_margin,
        consistency_frac_h1  = round(consistency, 4),
        h0                   = h0,
        h1                   = h1,
        vd_l6_present        = vd_l6_present,
        tss_labels_present   = tss_labels_present,
        vd_labels_present    = vd_labels_present,
        summary              = summary_text,
    )


# ═════════════════════════════════════════════════════════════════════════════
# COHORT STATISTICS (used by main summary reporting)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CohortAlignmentStats:
    """
    Aggregated alignment statistics across a cohort of studies.
    Intended for abstract / paper reporting.
    """
    n_studies:                  int
    n_h0_aligned:               int
    n_h1_shifted:               int
    n_insufficient:             int
    n_vd_l6_present:            int

    # Of studies where H1 was the true off-by-one candidates (VD L6 present)
    n_l6_confirmed_shifted:     int
    n_l6_rejected_shifted:      int

    # Dice statistics across all H0 valid pairs (entire cohort)
    mean_h0_score:              Optional[float]
    std_h0_score:               Optional[float]
    mean_h1_score:              Optional[float]
    std_h1_score:               Optional[float]

    # Agreement: studies where preferred matches castellvi classification
    n_castellvi_differs_h0_h1:  int
    n_phenotype_differs_h0_h1:  int

    # Confidence breakdown
    n_high_confidence:          int
    n_moderate_confidence:      int
    n_low_confidence:           int

    def off_by_one_rate(self) -> float:
        return self.n_h1_shifted / max(self.n_studies, 1)

    def l6_false_positive_rate(self) -> float:
        """Of studies where VERIDAH showed L6, fraction that were NOT true off-by-one."""
        denom = self.n_vd_l6_present
        return self.n_l6_rejected_shifted / max(denom, 1) if denom > 0 else float('nan')

    def to_dict(self) -> dict:
        return {
            'n_studies':                  self.n_studies,
            'n_h0_aligned':               self.n_h0_aligned,
            'n_h1_shifted':               self.n_h1_shifted,
            'n_insufficient':             self.n_insufficient,
            'n_vd_l6_present':            self.n_vd_l6_present,
            'off_by_one_rate_pct':        round(self.off_by_one_rate() * 100, 2),
            'n_l6_confirmed_shifted':     self.n_l6_confirmed_shifted,
            'n_l6_rejected_shifted':      self.n_l6_rejected_shifted,
            'l6_false_positive_rate_pct': round(self.l6_false_positive_rate() * 100, 2)
                                           if self.n_vd_l6_present > 0 else None,
            'mean_h0_score':              self.mean_h0_score,
            'std_h0_score':               self.std_h0_score,
            'mean_h1_score':              self.mean_h1_score,
            'std_h1_score':               self.std_h1_score,
            'n_castellvi_differs_h0_h1':  self.n_castellvi_differs_h0_h1,
            'n_phenotype_differs_h0_h1':  self.n_phenotype_differs_h0_h1,
            'n_high_confidence':          self.n_high_confidence,
            'n_moderate_confidence':      self.n_moderate_confidence,
            'n_low_confidence':           self.n_low_confidence,
        }


def compute_cohort_stats(alignment_results: List[AlignmentResult]) -> CohortAlignmentStats:
    """Aggregate per-study AlignmentResult objects into cohort-level statistics."""
    n          = len(alignment_results)
    n_h0       = sum(1 for r in alignment_results if r.preferred_hypothesis == 'H0_aligned')
    n_h1       = sum(1 for r in alignment_results if r.preferred_hypothesis == 'H1_shifted')
    n_insuf    = sum(1 for r in alignment_results if r.preferred_hypothesis == 'insufficient_data')
    n_l6       = sum(1 for r in alignment_results if r.vd_l6_present)
    n_l6_conf  = sum(1 for r in alignment_results if r.vd_l6_present and r.preferred_hypothesis == 'H1_shifted')
    n_l6_rej   = sum(1 for r in alignment_results if r.vd_l6_present and r.preferred_hypothesis == 'H0_aligned')

    h0_scores  = [r.score_h0 for r in alignment_results if r.score_h0 is not None]
    h1_scores  = [r.score_h1 for r in alignment_results if r.score_h1 is not None]

    n_ct_diff  = sum(1 for r in alignment_results
                     if r.castellvi_h0 is not None
                     and r.castellvi_h1 is not None
                     and r.castellvi_h0 != r.castellvi_h1)
    n_ph_diff  = sum(1 for r in alignment_results
                     if r.phenotype_h0 is not None
                     and r.phenotype_h1 is not None
                     and r.phenotype_h0 != r.phenotype_h1)

    conf_counts = {'high': 0, 'moderate': 0, 'low': 0}
    for r in alignment_results:
        if r.confidence in conf_counts:
            conf_counts[r.confidence] += 1

    return CohortAlignmentStats(
        n_studies                 = n,
        n_h0_aligned              = n_h0,
        n_h1_shifted              = n_h1,
        n_insufficient            = n_insuf,
        n_vd_l6_present           = n_l6,
        n_l6_confirmed_shifted    = n_l6_conf,
        n_l6_rejected_shifted     = n_l6_rej,
        mean_h0_score             = round(float(np.mean(h0_scores)),  4) if h0_scores else None,
        std_h0_score              = round(float(np.std(h0_scores)),   4) if h0_scores else None,
        mean_h1_score             = round(float(np.mean(h1_scores)),  4) if h1_scores else None,
        std_h1_score              = round(float(np.std(h1_scores)),   4) if h1_scores else None,
        n_castellvi_differs_h0_h1 = n_ct_diff,
        n_phenotype_differs_h0_h1 = n_ph_diff,
        n_high_confidence         = conf_counts['high'],
        n_moderate_confidence     = conf_counts['moderate'],
        n_low_confidence          = conf_counts['low'],
    )
