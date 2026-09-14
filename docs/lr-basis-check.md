# LR+ basis check: scanned threshold versus exactly 95% specificity

Written by `python -m atlas.lr_basis_check`; the tables behind this page are `results/lr_basis_check_v1.tsv` (cells) and `results/lr_basis_check_genes_v1.tsv` (genes).

**Conclusion.** The positive likelihood ratios in this study are read at the most sensitive observed score threshold that reaches 95% specificity (`atlas.clinical_evidence.lr_at_specificity`), not interpolated to exactly 95% specificity. On the interpolated basis no territory reaches the Strong band either: the largest median is 17.564 (AlphaGenome, splice region), against 17.790 scanned.

Below the Strong threshold the basis is not immaterial. Interpolation changes 1 band assignment and raises 1 cell. It also gives a value to the 10 cells that have none on the scanned basis.

## What the atlas computes

- `lr_at_specificity` scans the observed score values in ascending order and uses the first whose false-positive rate is at most 1 − spec; a false-positive rate of 0 is bounded at 1/(n_normal + 1). The manuscript's Methods describe the same scan.
- It is applied per gene (at least 10 abnormal and 10 normal variants). The reported `lr_at_spec95_median` in `results/clinical_evidence_v1.tsv` is the **median of the per-gene ratios**, over BARD1, PALB2 and RAD51C, the three genes with assay-derived functional calls. The evidence band is assigned to that median.

## How this page is computed

- This module repeats clinical_evidence's label binarisation, merge, territory assignment, gene gating and median. Its scanned medians reproduce `results/clinical_evidence_v1.tsv` in all 69 cells (59 with a value, 10 not evaluable), and it stops if they do not.
- The interpolated value is the companion benchmark's definition, implemented here as `lr_interp_at_specificity`. Sensitivity is read linearly on the empirical ROC between the two observed thresholds that bracket FPR = 0.05, and LR+ = sensitivity / 0.05. When the top score value is already a tie group holding more than 5% of the normal variants, the segment starts at the empty call set.
- Realised specificity is 1 − FPR at the scanned threshold, per gene.

## The four ratios quoted in the text

| predictor | territory | gene | n abnormal | n normal | scanned LR+ | realised specificity | interpolated LR+ |
|---|---|---|---|---|---|---|---|
| AlphaGenome | splice region | BARD1 | 266 | 2471 | 18.730 | 95.02% | 18.647 |
| AlphaGenome | splice region | PALB2 | 319 | 966 | 17.412 | 95.03% | 17.304 |
| AlphaGenome | splice region | RAD51C | 312 | 1418 | 17.790 | 95.06% | 17.564 |
| **AlphaGenome** | **splice region** | **median** | | | **17.790** | | **17.564** |
| Pangolin | splice region | BARD1 | 209 | 2064 | 18.888 | 95.01% | 18.852 |
| Pangolin | splice region | PALB2 | 248 | 837 | 17.287 | 95.10% | 16.935 |
| Pangolin | splice region | RAD51C | 220 | 1020 | 16.273 | 95.00% | 16.273 |
| **Pangolin** | **splice region** | **median** | | | **17.287** | | **16.935** |
| SpliceAI | splice region | BARD1 | 209 | 2064 | 18.792 | 95.01% | 18.756 |
| SpliceAI | splice region | PALB2 | 248 | 837 | 16.052 | 95.10% | 15.726 |
| SpliceAI | splice region | RAD51C | 220 | 1020 | 16.636 | 95.00% | 16.636 |
| **SpliceAI** | **splice region** | **median** | | | **16.636** | | **16.636** |
| CADD | coding/UTR | BARD1 | 932 | 5485 | 13.841 | 95.08% | 13.698 |
| CADD | coding/UTR | PALB2 | 942 | 7638 | 13.168 | 95.09% | 13.076 |
| CADD | coding/UTR | RAD51C | 777 | 2588 | 12.800 | 95.32% | 12.257 |
| **CADD** | **coding/UTR** | **median** | | | **13.168** | | **13.076** |

Realised specificity for these per-gene values runs from 95.00% to 95.32%, and interpolation moves a quoted median by at most 0.351.

## Fig. 6B as a whole

Fig. 6B plots 30 cells with a scanned value. Interpolation changes 0 of their evidence bands. The largest change is 1.849, for phyloP-100way in splice 3–10 bp, and realised specificity per gene runs from 95.00% to 96.37%. The bounds quoted for the four ratios above do not extend to the whole panel.

## Every cell

Medians across genes. The realised specificity is the range across the genes with a scanned value.

| predictor | territory | genes | scanned LR+ | realised specificity | interpolated LR+ | band, scanned → interpolated |
|---|---|---|---|---|---|---|
| AlphaGenome | all | 3 | 4.332 | 95.00%–95.00% | 4.330 | Moderate |
| AlphaGenome | coding/UTR | 3 | 2.128 | 95.00%–95.01% | 2.124 | Supporting |
| AlphaGenome | splice region | 3 | 17.790 | 95.02%–95.06% | 17.564 | Moderate |
| AlphaGenome | splice 3–10 bp | 3 | 15.927 | 95.09%–95.19% | 15.538 | Moderate |
| AlphaGenome | splice 11–50 bp | 3 | 13.957 | 95.02%–95.06% | 13.846 | Moderate |
| SpliceAI | all | 3 | 5.509 | 95.01%–95.01% | 5.496 | Moderate |
| SpliceAI | coding/UTR | 3 | 2.384 | 95.00%–95.02% | 2.382 | Supporting |
| SpliceAI | splice region | 3 | 16.636 | 95.00%–95.10% | 16.636 | Moderate |
| SpliceAI | splice 3–10 bp | 3 | 11.574 | 95.03%–95.19% | 11.136 | Moderate |
| SpliceAI | splice 11–50 bp | 3 | 13.500 | 95.01%–95.14% | 13.333 | Moderate |
| Pangolin | all | 3 | 5.949 | 95.01%–95.01% | 5.942 | Moderate |
| Pangolin | coding/UTR | 3 | 3.823 | 95.00%–95.02% | 3.820 | Supporting |
| Pangolin | splice region | 3 | 17.287 | 95.00%–95.10% | 16.935 | Moderate |
| Pangolin | splice 3–10 bp | 3 | 13.227 | 95.03%–95.19% | 12.727 | Moderate |
| Pangolin | splice 11–50 bp | 3 | 12.375 | 95.01%–95.14% | 12.222 | Moderate |
| CADD | all | 3 | 13.014 | 95.08%–95.21% | 12.839 | Moderate |
| CADD | coding/UTR | 3 | 13.168 | 95.08%–95.32% | 13.076 | Moderate |
| CADD | splice region | 3 | 13.331 | 95.00%–95.22% | 12.772 | Moderate |
| CADD | splice 3–10 bp | 3 | 4.724 | 95.19%–95.58% | 4.701 | Moderate |
| CADD | splice 11–50 bp | 3 | 1.285 | 95.01%–95.14% | 1.250 | below Supporting |
| AlphaMissense | all | 3 | 12.601 | 95.02%–95.05% | 12.476 | Moderate |
| AlphaMissense | coding/UTR | 3 | 12.601 | 95.02%–95.05% | 12.476 | Moderate |
| Evo2 | all | 3 | 9.424 | 95.01%–95.01% | 9.413 | Moderate |
| Evo2 | coding/UTR | 3 | 8.656 | 95.00%–95.02% | 8.648 | Moderate |
| Evo2 | splice region | 3 | 14.735 | 95.00%–95.10% | 14.435 | Moderate |
| Evo2 | splice 3–10 bp | 3 | 9.309 | 95.03%–95.19% | 8.955 | Moderate |
| Evo2 | splice 11–50 bp | 3 | 6.188 | 95.01%–95.14% | 6.111 | Moderate |
| GPN-MSA | all | 3 | 9.160 | 95.01%–95.09% | 9.038 | Moderate |
| GPN-MSA | coding/UTR | 3 | 8.130 | 95.01%–95.04% | 8.115 | Moderate |
| GPN-MSA | splice region | 3 | 12.909 | 95.00%–95.10% | 12.909 | Moderate |
| GPN-MSA | splice 3–10 bp | 3 | 6.516 | 95.03%–95.19% | 6.269 | Moderate |
| GPN-MSA | splice 11–50 bp | 3 | 6.188 | 95.06%–95.14% | 6.111 | Moderate |
| NT-v2-500M | all | 3 | 5.844 | 95.01%–95.01% | 5.837 | Moderate |
| NT-v2-500M | coding/UTR | 3 | 6.057 | 95.00%–95.02% | 6.052 | Moderate |
| NT-v2-500M | splice region | 3 | 5.433 | 95.00%–95.10% | 5.323 | Moderate |
| NT-v2-500M | splice 3–10 bp | 3 | 4.654 | 95.03%–95.19% | 4.478 | Moderate |
| NT-v2-500M | splice 11–50 bp | 3 | 0.910 | 95.01%–95.14% | 0.909 | below Supporting |
| phyloP-100way | all | 3 | 6.123 | 95.01%–95.32% | 6.107 | Moderate |
| phyloP-100way | coding/UTR | 3 | 5.016 | 95.03%–95.42% | 5.001 | Moderate |
| phyloP-100way | splice region | 3 | 12.124 | 95.03%–95.35% | 11.625 | Moderate |
| phyloP-100way | splice 3–10 bp | 3 | 7.456 | 95.37%–96.37% | 5.607 | Moderate |
| phyloP-100way | splice 11–50 bp | 3 | 1.762 | 95.08%–95.63% | 1.538 | below Supporting |
| phastCons-100way | all | 3 | — | — | 2.967 | **not evaluable → Supporting** |
| phastCons-100way | coding/UTR | 3 | — | — | 2.390 | **not evaluable → Supporting** |
| phastCons-100way | splice region | 3 | — | — | 7.982 | **not evaluable → Moderate** |
| phastCons-100way | splice 3–10 bp | 3 | — | — | 3.250 | **not evaluable → Supporting** |
| phastCons-100way | splice 11–50 bp | 3 | 2.982 | 97.26%–98.71% | 0.807 | **Supporting → below Supporting** |
| gnomAD AF (global) | all | 3 | 1.484 | 95.02%–96.31% | 1.019 | below Supporting |
| gnomAD AF (global) | coding/UTR | 3 | 1.098 | 95.45%–95.94% | 1.088 | below Supporting |
| gnomAD AF (global) | splice region | 3 | — | — | 0.828 | **not evaluable → below Supporting** |
| gnomAD AF (global) | splice 3–10 bp | 3 | — | — | 1.500 | **not evaluable → below Supporting** |
| gnomAD AF (popmax) | all | 3 | — | — | 1.088 | **not evaluable → below Supporting** |
| gnomAD AF (popmax) | coding/UTR | 3 | — | — | 1.178 | **not evaluable → below Supporting** |
| gnomAD AF (popmax) | splice region | 3 | — | — | 0.888 | **not evaluable → below Supporting** |
| gnomAD AF (popmax) | splice 3–10 bp | 3 | — | — | 1.125 | **not evaluable → below Supporting** |
| REVEL | all | 3 | 9.808 | 95.02%–95.07% | 9.774 | Moderate |
| REVEL | coding/UTR | 3 | 9.808 | 95.02%–95.07% | 9.774 | Moderate |
| BayesDel addAF | all | 3 | 12.593 | 95.01%–95.02% | 12.543 | Moderate |
| BayesDel addAF | coding/UTR | 3 | 12.498 | 95.01%–95.02% | 12.448 | Moderate |
| ClinPred | all | 3 | 10.077 | 95.01%–95.02% | 10.065 | Moderate |
| ClinPred | coding/UTR | 3 | 10.077 | 95.01%–95.02% | 10.065 | Moderate |
| MetaRNN | all | 3 | 10.100 | 95.00%–95.02% | 10.078 | Moderate |
| MetaRNN | coding/UTR | 3 | 10.115 | 95.00%–95.04% | 10.115 | Moderate |
| PrimateAI | all | 3 | 4.572 | 95.01%–95.11% | 4.474 | Moderate |
| PrimateAI | coding/UTR | 3 | 4.572 | 95.01%–95.11% | 4.474 | Moderate |
| VEST4 | all | 3 | 10.737 | 95.02%–95.04% | 10.760 | Moderate |
| VEST4 | coding/UTR | 3 | 10.774 | 95.02%–95.04% | 10.760 | Moderate |
| ESM-1b | all | 3 | 10.543 | 95.01%–95.04% | 10.521 | Moderate |
| ESM-1b | coding/UTR | 3 | 10.543 | 95.01%–95.04% | 10.521 | Moderate |

## Findings

1. **Strong band.** No median reaches 18.7 on either basis. Per gene, AlphaGenome in BARD1 (splice region) is 18.730 scanned and 18.647 interpolated; SpliceAI in BARD1 (splice region) is 18.792 scanned and 18.756 interpolated; Pangolin in BARD1 (splice region) is 18.888 scanned and 18.852 interpolated. These single-gene values reach the threshold, but the evidence band is assigned to the median across genes.
2. **Realised specificity.** 8 of 174 per-gene values sit above 95.5%, all in coarsely valued scores or small strata:

   | predictor | territory | gene | scanned LR+ | realised specificity | interpolated LR+ |
   |---|---|---|---|---|---|
   | CADD | splice 3–10 bp | PALB2 | 6.082 | 95.58% | 5.376 |
   | phyloP-100way | splice 3–10 bp | BARD1 | 7.456 | 96.37% | 5.607 |
   | phyloP-100way | splice 3–10 bp | PALB2 | 6.369 | 95.67% | 5.512 |
   | phyloP-100way | splice 11–50 bp | RAD51C | 1.762 | 95.63% | 1.538 |
   | phastCons-100way | splice 11–50 bp | PALB2 | 0.000 | 97.26% | 0.807 |
   | phastCons-100way | splice 11–50 bp | RAD51C | 5.964 | 98.71% | 2.959 |
   | gnomAD AF (global) | all | PALB2 | 1.944 | 96.31% | 1.517 |
   | gnomAD AF (global) | coding/UTR | RAD51C | 1.032 | 95.94% | 1.088 |

3. **Interpolation does not always lower the ratio.** It is lower in 55 of 59 cells, unchanged in 3, and higher in VEST4 / all (10.737 → 10.760). It is higher whenever the ROC just past the scanned point is steeper than the scanned ratio.
4. **Band changes.** phastCons-100way in splice 11–50 bp goes from Supporting (2.982) to below Supporting (0.807), with realised specificity 97.26%–98.71%.
5. **Cells without a scanned value.** The 10 cells where no observed threshold reaches 95% specificity all receive an interpolated value, read between calling no variant positive and calling the top tie group positive:

   | predictor | territory | interpolated LR+ | band |
   |---|---|---|---|
   | phastCons-100way | all | 2.967 | Supporting |
   | phastCons-100way | coding/UTR | 2.390 | Supporting |
   | phastCons-100way | splice region | 7.982 | Moderate |
   | phastCons-100way | splice 3–10 bp | 3.250 | Supporting |
   | gnomAD AF (global) | splice region | 0.828 | below Supporting |
   | gnomAD AF (global) | splice 3–10 bp | 1.500 | below Supporting |
   | gnomAD AF (popmax) | all | 1.088 | below Supporting |
   | gnomAD AF (popmax) | coding/UTR | 1.178 | below Supporting |
   | gnomAD AF (popmax) | splice region | 0.888 | below Supporting |
   | gnomAD AF (popmax) | splice 3–10 bp | 1.125 | below Supporting |

   No single score threshold achieves those operating points, so the atlas keeps these cells as not evaluable (Note S15).
6. **Aggregation.** The atlas reports the median of per-gene ratios. The companion reports one ratio per object on the pooled set, with gene-clustered bootstrap intervals. Both use 95% specificity and the Tavtigian point system at a prior of 0.10; this comparison changes neither aggregation.
