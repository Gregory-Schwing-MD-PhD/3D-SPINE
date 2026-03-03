# AACA 2026 Abstract — Submission Draft

---

## Author Block (formatted per AACA guidelines)

SCHWING, Gregory J. Wayne State University School of Medicine, Detroit, MI, 48201, USA.

---

## Title

Automated Morphometric Classification of Lumbosacral Transitional Vertebrae on MRI: Agreement Between Two Deep-Learning Spinal Segmenters and Implications for Castellvi Typing.

---

## Abstract Body

*(Single structured paragraph — 2000 character limit including spaces. Draft below; trim before submission. Estimated current length: ~1,850 characters — verify with character counter before submitting.)*

INTRODUCTION. Lumbosacral transitional vertebrae (LSTV) affect up to 35% of the population and are a known source of wrong-level spine surgery. Automated enumeration of lumbar segments on routine MRI is complicated by disagreement between segmentation models in the setting of LSTV, particularly regarding whether the most caudal mobile segment should be labeled L5 or L6. The primary objective of this study was to quantify inter-segmenter agreement in lumbar vertebral level assignment between two publicly available deep-learning sagittal MRI spinal segmenters — TotalSpineSeg (TSS) and SPINEPS/VERIDAH — across a large validation cohort. The secondary objective was to evaluate how labeling discordance propagates into a white-box morphometric Castellvi classification pipeline and affects automated LSTV phenotyping. METHODS. A retrospective analysis was performed on n=283 lumbar MRI studies drawn from the Ian Pan LSTV validation subset; one study was excluded due to SPINEPS segmentation failure, yielding a final cohort of n=282. For each study, TSS and SPINEPS vertebral body masks were co-registered and compared using a generalised sequence-level Dice alignment model testing integer level offsets (−2 to +2) across all lumbar pairs simultaneously. The accepted offset was defined as the offset whose mean pair-wise Dice exceeded the aligned (offset=0) score by ≥8 percentage points with ≥60% pair-level consistency. When segmenters disagreed, Castellvi classification was run independently under each hypothesis using the transverse process incident on each segmenter's most caudal lumbar vertebra, and results were compared. Phenotyping (sacralization, lumbarization, normal) and Bayesian LSTV probability were computed from vertebral body morphometrics, disc height indices, and Castellvi type. SUMMARY. [REPORT: total cohort n=282; n (%) studies with offset=0 (aligned), n (%) with offset=+1 (lumbarization candidate), n (%) with offset=−1 (sacralization candidate), n (%) insufficient data. Of studies where SPINEPS identified a VD L6 label: n (%) confirmed offset=+1, n (%) rejected (VD L6 false positive). Mean±SD alignment Dice at accepted offset. N (%) studies where Castellvi classification differed between the two segmenter hypotheses. N (%) LSTV detected overall; breakdown by phenotype (sacralization/lumbarization/normal) and Castellvi subtype. Report castellvi_disagrees rate.] CONCLUSIONS. [REPORT: primary conclusion on inter-segmenter agreement rate; secondary conclusion on how often disagreement changes the Castellvi type; clinical implication — which segmenter's TV assignment should be preferred and under what conditions. Note that ground truth verification via axial nerve morphology remains outstanding.]

---

## Notes for Completion Before Submission

**Results placeholders — fill these from your CSV outputs:**

| Field | Where to find it |
|---|---|
| n aligned / offset=+1 / offset=−1 / insufficient | `lstv_cohort_summary.csv` → `n_aligned`, `n_shifted_plus_1`, `n_shifted_minus_1`, `n_insufficient` |
| VD L6 present / confirmed / rejected | `n_vd_l6_present`, `n_l6_confirmed_m1`, `n_l6_rejected` |
| Mean±SD Dice at accepted offset | `mean_best_score ± std_best_score` |
| Castellvi disagreement rate | count of `castellvi_disagrees == TRUE` in `lstv_per_study.csv` |
| LSTV prevalence + phenotype breakdown | `lstv_per_study.csv` → `lstv_detected`, `lstv_phenotype` |
| Castellvi subtype distribution | `castellvi_type` column |

**Character count:** Paste the abstract body into [charactercounttool.com](https://charactercounttool.com) before submitting — limit is 2,000 characters including spaces and the section headings.

**IRB:** The abstract guidelines require IRB approval or exemption documentation for studies involving living human subjects. Confirm whether the Ian Pan dataset qualifies as retrospective/de-identified and thus exempt, or whether a formal determination letter is needed from Wayne State.

**Submission deadline:** March 3, 2026 at 12:00 PM EST. Late-breaking window closes March 24.

**Fee:** $50 (Associate Member — student/resident rate).

**Presentation format:** Select all three (Platform, TechFair, Poster) at submission time to maximize placement options. The white-box pipeline and Dice matrix visualizations would work well as a TechFair demonstration.

---

## Future Directions Section (for talk/poster — not in abstract body)

Ground truth vertebral level identification on routine lumbar MRI is achievable without full-spine localizers using axial T2-weighted images at the level of the sacral foramina. The L5 nerve root characteristically lacks proximal splitting and is approximately twice the caliber of the L4 peroneal branch at this level. This permits level identification as follows: in a 4-lumbar-segment patient (sacralized L5), a bundle of multiple splitting nerves at the lateral sacrum represents L4; in a standard 5-segment patient, a thin nerve joining a thicker nerve represents the L4 peroneal branch and L5 respectively; in a 6-segment patient (lumbarized S1), two nerves of similar caliber represent L5 and S1. The iliolumbar ligament, which characteristically arises from the L5 transverse process, provides a secondary landmark though its reliability in transitional anatomy has been questioned. A planned next step is prospective radiologist annotation of axial nerve morphology in the discordant-offset subgroup to provide ground truth labels, which will be used to train a convolutional network for automated level verification — closing the loop between segmenter disagreement detection and anatomically confirmed classification.
