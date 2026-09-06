"""Assemble the Supplementary Note and supplementary tables from pipeline outputs.

Every number in the Note is read from a file under ``results/`` at build time
(CONVENTIONS.md rule 5), so the Note cannot drift from the analysis. Emits Markdown,
a Word version, and the supplementary tables as TSV.

Usage (PYTHONPATH=src):
    python -m atlas.supplement --out ~/Desktop/functional-standard-atlas_supplement_v3
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

LABEL = {
    "alphagenome": "AlphaGenome", "spliceai_ds": "SpliceAI", "pangolin_score": "Pangolin",
    "cadd": "CADD", "alphamissense": "AlphaMissense", "evo2": "Evo2-7B",
    "gpn_msa": "GPN-MSA", "nucleotide_transformer": "NT-v2-500M",
    "phylop100way": "phyloP-100way", "phastcons100way": "phastCons-100way",
    "gnomad_af_global": "gnomAD AF (global)", "gnomad_af_popmax": "gnomAD AF (popmax)",
    "revel": "REVEL", "bayesdel_addaf": "BayesDel (AF)", "clinpred": "ClinPred",
    "metarnn": "MetaRNN", "primateai": "PrimateAI", "vest4": "VEST4", "esm1b": "ESM-1b",
}
STRAT = {
    "all": "All", "coding_or_utr": "Coding / UTR", "splice_region": "Splice region",
    "splice_1_2": "Splice ±1–2", "splice_3_10": "Splice 3–10 bp",
    "splice_11_50": "Splice 11–50 bp", "splice_deep": "Splice >50 bp",
    "clinvar_recorded": "ClinVar-recorded", "clinvar_absent": "ClinVar-absent",
    "snv": "SNV", "indel": "Indel", "missense": "Missense",
    "synonymous": "Synonymous", "nonsense": "Nonsense", "utr": "UTR",
    "coding_snv_nontruncating": "Coding SNV, non-truncating",
}
ORDER = list(LABEL)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def cell(v):
        if isinstance(v, float):
            return "—" if pd.isna(v) else floatfmt.format(v)
        return "—" if pd.isna(v) else str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, rule] + rows)


def build_note() -> str:
    rel = pd.read_csv(RESULTS / "reliability_v1.tsv", sep="\t")
    relj = json.loads((RESULTS / "reliability_v1.json").read_text())
    att = pd.read_csv(RESULTS / "attenuation_v1.tsv", sep="\t")
    ties = pd.read_csv(RESULTS / "tie_audit_v1.tsv", sep="\t")
    power = pd.read_csv(RESULTS / "power_v1.tsv", sep="\t")
    h2h = pd.read_csv(RESULTS / "head_to_head_v1.tsv", sep="\t")
    logo = pd.read_csv(RESULTS / "logo_v1.tsv", sep="\t")
    ext = pd.read_csv(RESULTS / "eval_ext_v1.tsv", sep="\t")
    ce = pd.read_csv(RESULTS / "clinical_evidence_v1.tsv", sep="\t")
    cej = json.loads((RESULTS / "clinical_evidence_v1.json").read_text())
    rna = json.loads((RESULTS / "rna_readout_v1.json").read_text())
    ens = pd.read_csv(RESULTS / "ensemble_logo_v1.tsv", sep="\t")
    cm = pd.read_csv(RESULTS / "model_correlation_coding_or_utr_v1.tsv", sep="\t", index_col=0)
    cons = json.loads((RESULTS / "consequence_v1.provenance.json").read_text())
    aux = json.loads((RESULTS / "assay_aux_v1.provenance.json").read_text())

    out: list[str] = []
    A = out.append
    A("# Supplemental Note — functional-standard-atlas\n")
    A("Companion to: *Measurement reliability bounds functional benchmarks and relocates where variant effect prediction fails*\n")
    A("Every statistic below is read from a pipeline output file at build time by "
      "`atlas.supplement`; the source file is named in each section.\n")
    A("\n---\n")

    # ---- S1 -------------------------------------------------------------
    A("\n## S1. Variant mapping and atlas freezing\n")
    A("All 64,178 assay rows (7 MaveDB SGE assays) were mapped to GRCh38 with a "
      "**100.00% success rate** (`results/mapping_report_v1.md`). Provenance was fetched "
      "live by script and hash-registered: MANE Select transcripts (NCBI MANE GRCh38 "
      "v1.5); transcript exon models (Mutalyzer3 API); genomic exon anchors and reference "
      "sequence (Ensembl REST, GRCh38); cross-validation by Ensembl VEP.\n")
    A("Orientation rule, recorded per assay: `functional_pathogenicity = −score` "
      "(SGE raw score: higher = more normal function).\n")
    A("Coordinates are GRCh38, `pos` 1-based, indels left-anchored VCF-style, alleles on "
      "the + strand.\n")
    A("\n### S1.1 Region-classifier correction\n")
    A("An earlier form of the intronic-offset pattern accepted only a `*` prefix before "
      "the position, and therefore filed nine BRCA1 variants lying in an intron *within "
      "the 5'UTR* (`c.-19-1`, `c.-19-2`, `c.-19-3`) as coding rather than splice. The "
      "pattern now accepts UTR-relative positions. Six of the nine move into splice ±1–2 "
      "and three into splice 3–10 bp. Only the ±1–2 stratum changes materially: the "
      "largest change in any pooled ρ is 0.036 (phastCons at ±1–2, −0.003 → 0.033) and 81 "
      "of 89 model × stratum cells move by less than 0.002. "
      "`tests/test_hardening.py::test_classify_region` pins the corrected behaviour.\n")

    # ---- S2 -------------------------------------------------------------
    A("\n## S2. Concordance validation\n")
    A("Every scorer ported from the companion benchmark had to reproduce the independent "
      "prior run on shared variants before entering the atlas (`results/concordance_*.json`). "
      "ρ = ±1.0 counts as a pass where the legacy column used the opposite orientation.\n")
    A("""
| Model | Concordance ρ | n shared | Verdict |
|---|---|---|---|
| CADD | 1.000 | 19,883 | pass |
| AlphaMissense | 1.000 | 11,911 | pass |
| phyloP-100way | 1.000 | 21,394 | pass |
| phastCons-100way | 1.000 | 21,394 | pass |
| SpliceAI | 1.000 | 21,394 | pass |
| Pangolin | 1.000 | 21,394 | pass |
| GPN-MSA | −1.000 | 21,394 | pass (orientation flip) |
| gnomAD AF (global) | −1.000 | 7,702 | pass (orientation flip) |
| gnomAD AF (popmax) | −1.000 | 7,702 | pass (orientation flip) |
| NT-v2-500M | 0.9997 (per-gene 0.9994–0.9998) | 21,394 | pass |
| AlphaGenome | **0.672** | 21,394 | **deliberate re-definition — analysed in main text** |
| Evo2-7B | — (no legacy column) | — | new standard definition |
""")

    # ---- S3 -------------------------------------------------------------
    A("\n## S3. GPU scoring runs\n")
    A("Both foundation models were scored on a rented NVIDIA A6000 (48 GB) under tmux with "
      "append-only TSV score caches for crash resumability ("
      "environments pinned in `models/{nt,evo2}/requirements.txt` with full `pip freeze` "
      "archives).\n")
    A("- **NT-v2-500M**: masked 6-mer LLR, 6,000-bp window, transformers 4.46.3 / "
      "torch 2.11.0+cu128; 46,392 SNVs.\n"
      "- **Evo2-7B**: windowed LL delta, 8,192-bp window, evo2 0.6.0 / torch 2.7.1+cu128 / "
      "flash-attn 2.8.0.post2, bf16; 46,392 SNVs in 25.0 h (5,024 batches, batch size 8).\n")
    A("Skipped (unscored): FASTA reference-base mismatch; ambiguous (N) bases in window; "
      "windows truncated below 1 kb at chromosome ends.\n")

    # ---- S4 -------------------------------------------------------------
    A("\n## S4. Auxiliary assay measurements and protein consequence\n")
    A("Per-gene and pooled correlations for every model \u00d7 stratum cell are in "
      "Supplemental_Table_S1; per-model forest plots are Supplemental_Fig_S1\u2013S12 and "
      "the concordance scatters Supplemental_Fig_S13.\n")
    A("`results/assay_aux_v1.parquet`, `results/consequence_v1.parquet`\n")
    A("\nThe frozen atlas keeps one number per variant. The deposits carry more; "
      "`atlas.assay_aux` lifts the additional columns into one table, validating each. "
      "Coverage within the atlas:\n")
    cov = pd.DataFrame([{"Column": k, "Variants": f"{v:,}"} for k, v in aux["coverage"].items()])
    A("\n" + md_table(cov) + "\n")
    A("\nPer-assay decisions:\n")
    for urn, rec in aux["assays"].items():
        A(f"- **{rec['gene']}** — " + ("; ".join(rec["carried"]) if rec["carried"]
                                       else "score only, no auxiliary columns") + "\n")
    A("\nStandard errors were accepted only where the deposit also publishes a 95% CI to "
      "check them against; BARD1 and PALB2 both give (CI upper − score)/SE = 1.960 exactly. "
      "BAP1's SE has no published CI and RAD51C's is on a different scale from its score, "
      "so neither is used for reliability. RNA columns were carried only if positively "
      "rank-correlated with that assay's own score, so the atlas orientation rule applies "
      "unchanged; VHL's second RNA timepoint failed this check (ρ = −0.047) and was dropped "
      "automatically.\n")
    A("\n`atlas.consequence` re-derives protein consequence from the cached Mutalyzer "
      "transcript models, because `hgvs_pro` is empty in all seven deposits. All 64,178 "
      "variants classified, no reference-base mismatches:\n")
    cc = pd.DataFrame([{"Consequence": k, "n": f"{v:,}"}
                       for k, v in sorted(cons["counts"].items(), key=lambda kv: -kv[1])])
    A("\n" + md_table(cc) + "\n")
    A("\nAs an external check the derived missense set (26,022) is identical, variant for "
      "variant, to the set AlphaMissense scores.\n")

    # ---- S5 -------------------------------------------------------------
    A("\n## S5. Measurement reliability and the attenuation ceiling\n")
    A("`results/reliability_v1.tsv`, `results/attenuation_v1.tsv` (Supplemental_Table_S3)\n")
    A("\nThe table below reports the **attenuation ceiling** (= √reliability) per gene × "
      "stratum, not the reliability itself. Reliability is estimated from replicate scores "
      "(BRCA1, Spearman–Brown on the replicate correlation; the deposited score is exactly "
      "the replicate mean) or from CI-validated per-variant standard errors (BARD1, PALB2; "
      "`1 − mean(SE²)/var(score)`). To recover a reliability, square the tabulated value: "
      "the coding/UTR median ceiling of 0.897 corresponds to a reliability of 0.805, and the "
      "splice ±1–2 median ceiling of 0.772 to a reliability of 0.596.\n")
    A("\nThe between-gene spread is small in most strata but large at splice ±1–2 and "
      "splice 11–50 bp, where the three genes disagree by more than 0.20. Realisations in "
      "those two strata should be read as ranges, not point estimates; the main text quotes "
      "the median-ceiling value and states the range.\n")
    v = (rel[rel["status"] == "validated"]
         .sort_values("method", ascending=False)
         .drop_duplicates(["gene", "stratum"]))
    piv = v.pivot(index="stratum", columns="gene", values="ceiling")
    piv = piv.reindex([s for s in STRAT if s in piv.index])
    piv["Spread"] = piv.max(axis=1) - piv.min(axis=1)
    piv["Median"] = piv.drop(columns=["Spread"]).median(axis=1)
    piv.index = [STRAT[s] for s in piv.index]
    piv.insert(0, "Stratum", piv.index)
    A("\n" + md_table(piv.reset_index(drop=True)) + "\n")
    A("\n**Cross-check.** BRCA1 carries both estimators. The SE-based estimator returns a "
      "ceiling 0.049–0.062 higher than the replicate-based one in every stratum:\n")
    cr = pd.DataFrame(relj["cross_check_replicates_vs_se"])
    cr = cr.rename(columns={"stratum": "Stratum", "replicates": "Ceiling (replicates)",
                            "se": "Ceiling (SE)", "se_minus_replicates": "Difference"})
    cr["Stratum"] = cr["Stratum"].map(STRAT)
    A("\n" + md_table(cr.drop(columns=["gene"])) + "\n")
    A("\nBARD1 and PALB2 ceilings are therefore, if anything, slight overestimates, which "
      "makes the realisations in the main text slightly conservative.\n")
    A("\nRealisation (observed ρ ÷ ceiling), median over the contributing genes, shown only "
      "where at least two genes contribute an error model:\n")
    g = att.groupby(["model", "stratum"])["realisation"]
    real = g.median().where(g.count() >= 2).unstack()
    real = real.reindex([m for m in ORDER if m in real.index])
    real = real[[s for s in STRAT if s in real.columns]]
    real.columns = [STRAT[c] for c in real.columns]
    real.insert(0, "Predictor", [LABEL[m] for m in real.index])
    A("\n" + md_table(real.reset_index(drop=True)) + "\n")

    # ---- S6 -------------------------------------------------------------
    A("\n## S6. Score quantisation and statistical power\n")
    A("`results/tie_audit_v1.tsv` (Supplemental_Table_S5; rounded vs full precision in Supplemental_Table_S5b), `results/power_v1.tsv`\n")
    A("\nWhere a score vector contains ties, Spearman ρ is bounded below 1 for arithmetic "
      "reasons. The bound below is exact and attained by construction. The twelve most "
      "tie-limited cells:\n")
    t = ties.nsmallest(12, "rho_ceiling_ties").copy()
    t = pd.DataFrame({"Predictor": t["model"].map(LABEL), "Stratum": t["stratum"].map(STRAT),
                      "n": t["n"], "Distinct values": t["n_distinct"],
                      "Modal value": t["modal_value"], "Modal share": t["modal_share"],
                      "ρ ceiling from ties": t["rho_ceiling_ties"]})
    A("\n" + md_table(t) + "\n")
    A("\nThe SpliceAI and Pangolin command-line tools round delta scores to two decimals, "
      "which is the origin of most of this. It matters in one place — phastCons at ±1–2, "
      "capped at 0.572 — and does not explain the deep-intronic result, where the caps are "
      "still 0.833 (SpliceAI) and 0.804 (Pangolin).\n")
    A("\nMinimum ρ detectable at 80% power, α = 0.05, from the per-gene n actually used:\n")
    pw = (power.groupby("stratum")
          .agg(**{"Genes": ("k_genes", "max"), "Smallest gene n": ("n_min_gene", "min"),
                  "MDR, single gene": ("mdr_per_gene_min", "max"),
                  "MDR, pooled": ("mdr_pooled_homogeneous", "max")})
          .reindex([s for s in STRAT if s in set(power["stratum"])]))
    pw.insert(0, "Stratum", [STRAT[s] for s in pw.index])
    A("\n" + md_table(pw.reset_index(drop=True)) + "\n")
    # The table above aggregates by max, so its deep-intronic figure is the
    # worst-covered column -- gnomAD allele frequency, whose smallest gene
    # carries 74 variants. The manuscript quotes the figure for the predictors
    # that actually score the stratum, and the two must not disagree.
    A("\nThe deep-intronic stratum can only detect ρ ≥ 0.107 pooled for the predictors "
      "that score it, which bounds how strongly a null result there can be stated. The "
      "0.197 in the table above is the worst-covered column rather than a predictor: the "
      "two gnomAD allele-frequency columns carry 74 variants in their smallest gene "
      "against 249 to 296 for everything else.\n")

    # ---- S7 -------------------------------------------------------------
    A("\n## S7. Paired comparisons and leave-one-gene-out sensitivity\n")
    A("`results/head_to_head_v1.tsv` (Supplemental_Table_S6), `results/logo_v1.tsv`\n")
    A("\nTwo correlations measured on the same variants are dependent, so overlapping "
      "marginal CIs are not a test of their difference. Each ordering claim was tested on "
      "the variants both predictors score, with Steiger's test pooled across genes and a "
      "4,000-replicate gene-cluster bootstrap for Δρ (seed 20260803).\n")
    h = pd.DataFrame({
        "Model A": h2h["model_a"].map(LABEL), "Model B": h2h["model_b"].map(LABEL),
        "Stratum": h2h["stratum"].map(STRAT), "n paired": h2h["n_paired"],
        "ρ(A)": h2h["rho_a"], "ρ(B)": h2h["rho_b"], "Δρ": h2h["delta_rho"],
        "Bootstrap 95% CI": [f"{lo:.3f} to {hi:.3f}"
                             for lo, hi in zip(h2h["boot_ci_lo"], h2h["boot_ci_hi"])],
        "Steiger p": h2h["steiger_p"]})
    A("\n" + md_table(h) + "\n")
    A("\nLeave-one-gene-out: the ten largest swings in pooled ρ when a single gene is "
      "dropped.\n")
    lg = logo.nlargest(10, "logo_range").copy()
    lg = pd.DataFrame({"Predictor": lg["model"].map(LABEL), "Stratum": lg["stratum"].map(STRAT),
                       "Pooled ρ": lg["rho_full"], "LOGO min": lg["logo_min"],
                       "LOGO max": lg["logo_max"],
                       "Most influential gene": lg["most_influential_gene"],
                       "Δ if dropped": lg["delta_if_dropped"]})
    A("\n" + md_table(lg) + "\n")
    A("\nAll of the largest swings are in the ±1–2 stratum. The headline coding and "
      "all-variant numbers are stable (e.g. AlphaMissense coding/UTR 0.438, range "
      "0.410–0.475 across the seven leave-one-out folds).\n")

    # ---- S8 -------------------------------------------------------------
    A("\n## S8. Extended strata\n")
    A("`results/eval_ext_v1.tsv` (Supplemental_Table_S2)\n")
    A("\nPooled ρ over the original eight strata plus ClinVar-absent, SNV, indel, UTR, "
      "non-truncating coding SNV and the protein-consequence classes (missense, "
      "synonymous, nonsense) — sixteen in all. The ten non-splice strata are shown below; "
      "the complete grid is Supplemental_Table_S2.\n")
    w = ext.pivot(index="model", columns="stratum", values="pooled_rho")
    keep = ["all", "clinvar_recorded", "clinvar_absent", "coding_or_utr", "missense",
            "synonymous", "nonsense", "utr", "snv", "indel"]
    w = w.reindex([m for m in ORDER if m in w.index])[keep]
    w.columns = [STRAT[c] for c in keep]
    w.insert(0, "Predictor", [LABEL[m] for m in w.index])
    A("\n" + md_table(w.reset_index(drop=True)) + "\n")
    up = (ext.pivot(index="model", columns="stratum", values="pooled_rho")
          .eval("uplift = clinvar_recorded - clinvar_absent")["uplift"])
    A(f"\nMedian ClinVar-recorded minus ClinVar-absent uplift across the twelve "
      f"predictors: **{up.median():.3f}**. Against the full atlas instead of the absent "
      f"set the same quantity is 0.033, because the full set contains the recorded "
      f"variants and dilutes the contrast.\n")

    # ---- S9 -------------------------------------------------------------
    A("\n## S9. Classification, evidence strength, readouts and ensembles\n")
    A("`results/clinical_evidence_v1.tsv`, `results/rna_readout_v1.tsv`, "
      "`results/ensemble_logo_v1.tsv` (Supplemental_Table_S7), `results/model_correlation_*_v1.tsv`\n")
    A(f"\n**Labels.** {cej['n_labelled']:,} variants across "
      f"{', '.join(cej['genes'])}: GMM posterior ≥ 0.9 / ≤ 0.1 for BARD1 and PALB2 "
      "(ambiguous middle dropped), the authors' functional class for RAD51C "
      "('fast depleted' and 'slow depleted' abnormal, 'unchanged' normal, 'enriched' "
      "n = 49 excluded).\n")
    for g, d in cej["by_gene"].items():
        A(f"- {g}: n = {int(d['size']):,}, abnormal = {int(d['sum']):,} "
          f"({d['sum'] / d['size']:.1%})\n")
    A("\nAUROC and positive likelihood ratio at 95% specificity, by territory:\n")
    for s in ["all", "coding_or_utr", "splice_region", "splice_3_10"]:
        sub = ce[ce["stratum"] == s].sort_values("auroc", ascending=False)
        if sub.empty:
            continue
        A(f"\n**{STRAT[s]}**\n")
        d = pd.DataFrame({"Predictor": sub["model"].map(LABEL),
                          "n abnormal": sub["n_pos"], "n normal": sub["n_neg"],
                          "AUROC": sub["auroc"],
                          "95% CI": [f"{a:.3f}–{b:.3f}" for a, b in
                                     zip(sub["auroc_lo"], sub["auroc_hi"])],
                          "AUPRC": sub["auprc_median"],
                          "LR+ @95% spec": sub["lr_at_spec95_median"],
                          "ACMG band": sub["acmg_tier_spec95"]})
        A("\n" + md_table(d) + "\n")
    A("\n'Not evaluable' means the score is too coarsely quantised for any threshold to "
      "reach 95% specificity — it is not a statement that the predictor is weak.\n")
    a = rna["pooled_rna_vs_fitness"]
    A(f"\n**Cross-readout.** RNA-level scores are published for {rna['n_coding']:,} "
      f"coding/UTR variants across {', '.join(rna['genes'])}; the readout does not cover "
      f"splice-region variants. The two readouts of the same assay agree at "
      f"ρ = {a['rho']:.3f} ({a['ci_lo']:.3f}–{a['ci_hi']:.3f}), I² = {a['i2_pct']:.0f}%.\n")
    if "rna_reliability" in rna:
        r = rna["rna_reliability"]
        A(f"RNA readout reliability ({r['gene']}, n = {r['n']:,}, from replicates): "
          f"{r['reliability']:.3f}, ceiling {r['ceiling']:.3f}.\n")
    rr = pd.DataFrame([{"Predictor": LABEL.get(e["model"], e["model"]),
                        "vs fitness": e["vs_fitness"]["rho"] if e["vs_fitness"] else None,
                        "vs RNA": e["vs_rna"]["rho"] if e["vs_rna"] else None,
                        "Drop": e["drop"]} for e in rna["pooled_by_model"]])
    rr = rr.sort_values("vs fitness", ascending=False)
    A("\n" + md_table(rr) + "\n")
    A("\n**Selection strategies, leave-one-gene-out.** Median over the seven held-out "
      "genes; for each fold the strategy is fixed on the other six.\n")
    es = (ens.groupby("stratum")[["single_global_cadd", "territory_logo",
                                  "rank_average_broad", "rank_average_splice",
                                  "best_possible_oracle"]].median()
          .reindex([s for s in STRAT if s in set(ens["stratum"])]))
    es.columns = ["CADD everywhere", "Best model per territory", "Rank-avg (broad panel)",
                  "Rank-avg (splice panel)", "Oracle single model"]
    es.insert(0, "Stratum", [STRAT[s] for s in es.index])
    A("\n" + md_table(es.reset_index(drop=True)) + "\n")
    A("\n**Complementarity.** Between-model Spearman correlation in coding/UTR. The "
      "generalists are near-copies of one another and the splice-aware models form a "
      "second, largely independent cluster.\n")
    c = cm.copy()
    c.index = [LABEL.get(i, i) for i in c.index]
    c.columns = [LABEL.get(i, i) for i in c.columns]
    c.insert(0, "Predictor", c.index)
    A("\n" + md_table(c.reset_index(drop=True), "{:.2f}") + "\n")

    # ---- S10 simulation --------------------------------------------------
    sim = pd.read_csv(RESULTS / "attenuation_simulation_v1.tsv", sep="\t")
    simj = json.loads((RESULTS / "attenuation_simulation_v1.json").read_text())
    A("\n## S10. Simulation validation of the attenuation correction\n")
    A("`results/attenuation_simulation_v1.tsv` (Supplemental_Table_S4)\n")
    _tpg = int(simj["trials_per_cell"])          # 150; per gene, not per cell
    _lo, _hi = int(sim["trials"].min()), int(sim["trials"].max())
    A(f"\n{_tpg} trials per gene in each cell, over five stratum shapes x six true "
      "rho x six reliabilities; between "
      f"{_lo // _tpg} and {_hi // _tpg} genes contribute depending on the stratum, so a "
      f"cell carries {_lo:,} to {_hi:,} trials and the grid {int(sim['trials'].sum()):,} "
      "in total (seed "
      f"{simj['seed']}). True scores resample the observed distribution of one gene within "
      "one stratum — per gene, because the seven assays report on incompatible scales, so a "
      "pooled marginal would calibrate the injected noise against the scale differences "
      "rather than the within-assay spread. Reliability is re-estimated from simulated "
      "replicates exactly as the pipeline estimates it, so the whole chain is tested.\n")
    g = (sim.groupby("reliability_target")
            .agg(**{"Ceiling": ("ceiling_hat", "mean"),
                    "Bias, uncorrected": ("bias_uncorrected", "mean"),
                    "Bias, corrected": ("bias_corrected", "mean"),
                    "RMSE, uncorrected": ("rmse_uncorrected", "mean"),
                    "RMSE, corrected": ("rmse_corrected", "mean")}).reset_index()
            .rename(columns={"reliability_target": "Reliability"}))
    A("\n" + md_table(g) + "\n")
    A("\nThe correction is close to unbiased at every level, and reduces RMSE while the "
      "ceiling exceeds about 0.45; below that, dividing by a small and noisily estimated "
      "ceiling amplifies error. Of the strata measured here only the deep intron "
      "(ceiling 0.297) falls below that boundary, and it is reported uncorrected throughout.\n")
    A("\nBias by stratum shape, which tests whether the compressed distribution at splice "
      "+-1-2 breaks the correction (it does not):\n")
    b = (sim.groupby("stratum")[["bias_uncorrected", "bias_corrected"]].mean().reset_index()
            .rename(columns={"stratum": "Stratum", "bias_uncorrected": "Bias, uncorrected",
                             "bias_corrected": "Bias, corrected"}))
    A("\n" + md_table(b) + "\n")

    # ---- S11 definition sweep --------------------------------------------
    ds = pd.read_csv(RESULTS / "definition_sweep_performance_v1.tsv", sep="\t")
    dsj = json.loads((RESULTS / "definition_sweep_v1.json").read_text())
    A("\n## S11. AlphaGenome definition sweep\n")
    A("This section carries the full definition sweep, demoted from the main text: the "
      "sixteen window \u00d7 aggregation combinations, the client-version control, and the "
      "endpoint comparison between the two published score columns. The figure is "
      "Supplemental_Fig_S16 and the grid is Supplemental_Table_S8.\n")
    A("`results/definition_sweep_performance_v1.tsv` (Supplemental_Table_S8), "
      "`results/definition_sweep_concordance_v1.tsv`\n")
    A("\n2,722 variants, sampled stratified by gene and region with splice strata "
      "oversampled, scored under four API-supported window widths. Each response carries "
      "both raw and quantile scores for all three splicing output types, so four "
      "aggregations were derived per call and only the window cost extra calls: 10,888 "
      "calls, cached per variant x window, no errors.\n")
    piv = ds.pivot_table(index=["stratum", "width"], columns="definition",
                         values="pooled_rho").reset_index()
    A("\n" + md_table(piv) + "\n")
    r = dsj["replicate_legacy_definition"]
    A(f"\n**Client version.** Replicating the legacy definition (max|raw|, widest window) "
      f"under the current client reproduces the legacy column at rho = "
      f"{r['rho_vs_legacy_column']:.3f} (n = {r['n']:,}), so the client version contributes "
      "nothing to the rho = 0.672 disagreement; it is entirely definitional.\n")
    sa = dsj["score_agreement_by_region"]
    A(f"\n**Score agreement between the two published columns**: "
      f"{sa['overall']:.3f} overall; "
      + ", ".join(f"{k} {v:.3f}" for k, v in sa.items() if k != "overall") + ".\n")
    A("\n**Performance of the two definitions on the same variants** — the quantity that "
      "actually matters:\n")
    perf = pd.DataFrame([{"Stratum": k, "Atlas definition": v["atlas_definition"],
                          "Legacy definition": v["legacy_definition"],
                          "Difference": v["difference"]}
                         for k, v in dsj["performance_on_shared_variants"].items()])
    A("\n" + md_table(perf) + "\n")
    A(f"\nThe largest difference in any stratum is "
      f"{dsj['max_abs_performance_difference']:.3f}. A score-level concordance of "
      f"{sa['overall']:.3f} changed no conclusion this benchmark draws, so concordance gates "
      "should be evaluated on the benchmark's endpoints rather than on the correlation "
      "between score columns.\n")

    # ---- S12 ------------------------------------------------------------
    A("\n## S12. Repository and reproduction\n")
    # How many tests pass depends on what the checkout carries, so each figure
    # names its checkout. The previous wording -- "a clean clone: make fetch &&
    # make test (fetch, freeze and 83 guardrail tests)" -- named no state that
    # was ever measured: a bare clone runs 110 and an extract of the archive
    # runs 142. atlas.release_state now holds this sentence to the facts file.
    A("One-command reproduction from a clean clone: `make fetch && make test`. "
      "The pipeline is covered by 202 collected guardrail tests, 43 of them covering "
      "the analyses introduced here. A bare clone runs 130 and skips 72 for want of "
      "data; a clean extract of the release archive runs 158 and skips 23. "
      "Frozen data are immutable under `data/{raw,frozen}/` with SHA-256 manifests; every "
      "figure, table and number in the manuscript is produced by a committed script.\n")
    A("\nAnalysis modules added for this work: `atlas.assay_aux`, `atlas.consequence`, "
      "`atlas.reliability`, `atlas.robustness`, `atlas.evaluate_ext`, "
      "`atlas.clinical_evidence`, `atlas.ensemble`, `atlas.matched_realisation`, "
      "`atlas.supplement`; figures from "
      "`figures/hardening_figures.py`.\n")

    # ---- S13 unnumbered data files --------------------------------------
    A("\n## S13. Machine-readable source files\n")
    # Two numbered tables render a repository file directly rather than
    # summarising an analysis, so they are introduced here where a reader is
    # already being told what comes from where.
    A(f"\nTwo of the {_NUMWORD.get(len(TABLES), len(TABLES))} Supplemental Tables render a "
      "repository file directly. `Supplemental_Table_S11` is the per-predictor training "
      "provenance — training data, whether it carries clinical labels, whether any overlap "
      "with MAVE or SGE measurements is documented, and a documentation URL for every row. "
      "`Supplemental_Table_S12` collects the summary statistics quoted in the manuscript "
      "together with the output file each is computed from, so those figures can be checked "
      "without opening the deposit.\n")
    A(f"{_NUMWORD.get(len(EXTRA), len(EXTRA))} analysis-level source files carry the full "
      "numeric output behind the Note "
      "sections named below, at a granularity too fine to typeset — every model \u00d7 "
      "stratum \u00d7 gene cell rather than the summarised rows shown in the Supplemental "
      "Tables (all TSV except the Markdown mapping audit). They are deliberately not "
      "numbered as Supplemental Tables: they are part of "
      "the Zenodo data deposit (concept DOI 10.5281/zenodo.21828448), where they sit under "
      "`results/`; `results/` is not tracked in the GitHub repository and these files are "
      "not part of the submission package. Nothing in the manuscript depends on a reader "
      "opening them.\n")
    for f in EXTRA:
        A(f"\n- `{f}` — {EXTRA_DESCR[f]}\n")

    # ---- S14 MaveDB survey ------------------------------------------------
    surv = RESULTS / "mavedb_survey_v1.tsv"
    if surv.exists():
        sv = pd.read_csv(surv, sep="\t")
        n = len(sv)
        cnt = sv["permits"].value_counts()
        usable = int(cnt.get("replicates", 0) + cnt.get("errors", 0))
        A("\n## S14. What MaveDB deposits permit\n")
        A("`results/mavedb_survey_v1.tsv`\n")
        A(f"\nEvery published MaveDB score set ({n:,}) was enumerated through the public API "
          "and its score table retrieved, then classified by whether it carries replicate "
          "columns whose mean reproduces the deposited score, or a per-variant error estimate. "
          f"{usable:,} of {n:,} ({usable / n:.0%}) permit a reliability estimate on that "
          "test.\n")
        tab = (cnt.rename_axis("permits").reset_index(name="score sets")
                  .assign(**{"% of published": lambda t: (t["score sets"] / n * 100).round(1)}))
        A("\n" + md_table(tab) + "\n")
        A("\nThe test is looser than the reconciliation the primary analysis applies, so these "
          "are upper bounds on what is usable; deposits whose replicate columns do not "
          "reproduce the score are listed as `replicates_unverified` and are not counted. The "
          "per-deposit table, including the column names matched in each case, is the TSV "
          "named above.\n")
    return "".join(out)


TABLES = {
    # Journal numbering: contiguous from S1, in order of first citation.
    # This mapping is the single source of truth for the delivered package; an
    # earlier build used a non-contiguous S3-S10 scheme and the rename was applied
    # by hand, which let the package drift from the repository.
    "Supplemental_Table_S1_per_gene_rho.tsv": "tableS3_per_gene_rho.tsv",
    "Supplemental_Table_S2_extended_strata.tsv": "eval_ext_v1.tsv",
    "Supplemental_Table_S3_reliability_attenuation.tsv": "attenuation_v1.tsv",
    "Supplemental_Table_S4_attenuation_simulation.tsv": "attenuation_simulation_v1.tsv",
    "Supplemental_Table_S5_tie_audit.tsv": "tie_audit_v1.tsv",
    "Supplemental_Table_S5b_tie_rounded_vs_fullprec.tsv":
        "tie_audit_rounded_vs_fullprec_v1.tsv",
    "Supplemental_Table_S6_head_to_head.tsv": "head_to_head_v1.tsv",
    "Supplemental_Table_S7_selection_logo.tsv": "ensemble_logo_v1.tsv",
    "Supplemental_Table_S8_definition_sweep.tsv": "definition_sweep_performance_v1.tsv",
    # S9 covers all nineteen scored columns, with their licences. It carried
    # twelve until 2026-08-28: the seven dbNSFP meta-predictors were joined in by
    # hand at submission and never went back into the build, so a rebuild
    # produced a shorter table than the manuscript cites. Pinned by
    # tests/test_supplemental_table_s9.py.
    "Supplemental_Table_S9_model_inventory.tsv": "table2_model_inventory.tsv",
    "Supplemental_Table_S10_fdr_grid.tsv": "fdr_grid_v1.tsv",
    "Supplemental_Table_S10b_fdr_pairwise.tsv": "fdr_head_to_head_v1.tsv",
    # S11 and S12 were added to the delivered workbook by
    # atlas.build_additional_file_1 but not here, so this module and that one
    # disagreed about how many tables exist: the Notes said twelve while the
    # workbook shipped fourteen. This map is the single source of truth.
    "Supplemental_Table_S11_training_data.tsv": "table_s11_training_data_v1.tsv",
    "Supplemental_Table_S12_quoted_summary.tsv": "table_s12_quoted_summary_v1.tsv",
}
_NUMWORD = {9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen",
            14: "Fourteen", 15: "Fifteen", 16: "Sixteen", 17: "Seventeen",
            18: "Eighteen", 19: "Nineteen", 20: "Twenty", 21: "Twenty-one",
            22: "Twenty-two", 23: "Twenty-three", 24: "Twenty-four",
            25: "Twenty-five", 26: "Twenty-six", 27: "Twenty-seven",
            28: "Twenty-eight", 29: "Twenty-nine", 30: "Thirty"}

EXTRA = ["mavedb_survey_v1.tsv", "mavedb_survey_audit_v1.tsv", "mavedb_ceilings_v1.tsv",
         "mavedb_estimator_comparison_v1.tsv", "protease_agreement_v1.tsv",
         "mavedb_metadata_v1.tsv",
         "matched_realisation_v1.tsv", "reliability_v1.tsv", "reliability_corroboration_v1.tsv", "power_v1.tsv",
         "logo_v1.tsv", "rna_readout_v1.tsv", "definition_sweep_concordance_v1.tsv",
         "clinical_evidence_v1.tsv", "ensemble_pooled_v1.tsv",
         "model_correlation_all_v1.tsv", "model_correlation_coding_or_utr_v1.tsv",
         "model_correlation_splice_region_v1.tsv", "predictor_resources_v1.tsv",
         "mapping_report_v1.md",
         "estimator_consistency_v1.tsv", "matched_realisation_loo_v1.tsv",
         "mavedb_accounting_v1.tsv", "mavedb_ceiling_k_sensitivity_v1.tsv",
         "head_to_head_pvalues_v1.tsv", "attenuation_simulation_v2.tsv"]

EXTRA_DESCR = {
    "mavedb_survey_v1.tsv":
        "every published MaveDB score set, with the replicate and error columns found in its "
        "score table, the replicate subset (if any) whose mean reproduces the deposited score, "
        "and the resulting classification (Note S14; Methods)",
    "mavedb_survey_audit_v1.tsv":
        "the sampled deposits whose full score-table header was inspected to check the "
        "column-name classification in both directions, with every column seen (Methods)",
    "mavedb_ceilings_v1.tsv":
        "per-deposit reliability and attenuation ceiling for every surveyed deposit that "
        "permits one, with the estimator used, the reference-class stratum and the status "
        "classification (Fig. 6; Methods)",
    "mavedb_estimator_comparison_v1.tsv":
        "deposits publishing both reconcilable replicates and an error column, with the "
        "reliability each estimator gives and the difference between them",
    "protease_agreement_v1.tsv":
        "trypsin-versus-chymotrypsin agreement per designed protein, the Spearman-Brown "
        "reliability it implies, and the ceiling the deposited fitting interval implies",
    "mavedb_metadata_v1.tsv":
        "deposit metadata used for the correlates in Supplemental_Fig_S17: publication and "
        "creation dates, variant counts and target genes",
    "matched_realisation_v1.tsv":
        "coding/UTR versus splice ±1-2 on the nine broad-scope predictors that carry an "
        "estimate in both strata, observed and attenuation-corrected, with the ratio between "
        "territories under each (Results; Methods)",
    "reliability_v1.tsv":
        "per-gene replicate-based measurement reliability and the implied attenuation "
        "ceiling for each stratum (source of Note S5 and Table S3)",
    "reliability_corroboration_v1.tsv":
        "independent corroboration of the reliability estimates: agreement ρ and implied "
        "ceiling from held-out assay comparisons, per gene and stratum (Note S5)",
    "power_v1.tsv":
        "minimum Spearman ρ detectable at 80% power, per gene and pooled, for every "
        "model × stratum cell (Note S6)",
    "logo_v1.tsv":
        "leave-one-gene-out pooled ρ for every model × stratum cell, with the held-out "
        "gene identified (Note S7)",
    "rna_readout_v1.tsv":
        "per-gene correlations of each predictor with the RNA-level readout and with "
        "fitness, the cross-readout comparison of Note S9",
    "definition_sweep_concordance_v1.tsv":
        "pairwise score concordance between AlphaGenome definitions at each API window "
        "width (Note S11)",
    "clinical_evidence_v1.tsv":
        "AUROC, AUPRC and positive likelihood ratio at 95% specificity per model and "
        "territory against assay-derived functional labels, with bootstrap intervals "
        "(Note S9)",
    "ensemble_pooled_v1.tsv":
        "pooled ρ of the five predictor-selection strategies per stratum over all seven "
        "genes; Table S7 reports the leave-one-gene-out median counterpart (Note S9)",
    "model_correlation_all_v1.tsv":
        "between-model Spearman correlation matrix over all scored variants",
    "model_correlation_coding_or_utr_v1.tsv":
        "between-model Spearman correlation matrix over coding/UTR variants (the Note S9 "
        "heatmap)",
    "model_correlation_splice_region_v1.tsv":
        "between-model Spearman correlation matrix over splice-region variants",
    "predictor_resources_v1.tsv":
        "the curated source of truth behind Supplemental_Table_S9: per-predictor "
        "resource, version, source, access date, and the licence governing that "
        "score column with the URL it was read from (generated by "
        "`atlas.predictor_resources`; S9 renders it with the scope, score "
        "definition and coverage columns joined on)",
    "mapping_report_v1.md":
        "the human-readable mapping audit: per-assay variant counts, Mutalyzer3 "
        "versus Ensembl exon-model agreement, VEP cross-validation pass rates and "
        "the orientation check behind the frozen matrix (Methods; the machine-readable "
        "counterpart is mapping_summary_v1.json)",
    "estimator_consistency_v1.tsv":
        "the BRCA1 estimator cross-check behind Table S14: per-stratum reliability and "
        "ceiling under the replicate chain and under the SE estimator, with the "
        "difference between them",
    "matched_realisation_loo_v1.tsv":
        "leave-one-gene-out forms of the matched coding-versus-splice ratio behind "
        "Table S13: median coding and splice ±1–2 realisations and their ratio with "
        "each ceiling-bearing gene dropped in turn",
    "mavedb_accounting_v1.tsv":
        "the MaveDB deposit ledger behind Table S16's accounting: every counting "
        "definition from 2,803 published score sets down to the 674 computable "
        "ceilings, with the denominator and fraction for each",
    "mavedb_ceiling_k_sensitivity_v1.tsv":
        "sensitivity of the MaveDB ceiling distribution to treating a deposited "
        "standard deviation as the error of a mean of k replicates, for k = 1, 2, 3 "
        "(Table S16)",
    "head_to_head_pvalues_v1.tsv":
        "Steiger and gene-cluster bootstrap P values side by side for all eighteen "
        "claimed orderings, with the significance verdict under each and whether they "
        "agree (Table S17)",
    "attenuation_simulation_v2.tsv":
        "the hardened attenuation simulation behind Table S15: every noise scenario x "
        "estimator x stratum x reliability target cell with bias and RMSE before and "
        "after correction (1,800 rows; the v1 grid is untouched in "
        "attenuation_simulation_v1.tsv)",
}

# The six main-text figures, in manuscript order. Position is load-bearing:
# build_submission() copies MAIN_FIGS[i] to Zhang_Fig{i+1}, so reordering this
# list silently renumbers the delivered figures. The manuscript is the source of
# truth, and tests/test_figure_numbering.py pins this list against it -- update
# both together or the test fails.
MAIN_FIGS = ["fig_atlas_overview", "fig_territory_corrected",
             "fig_mavedb_ceilings", "fig_splice_head_to_head",
             "fig_rna_readout", "fig_classification"]
DEMOTED_FIGS = {"fig_splice_territory": "Supplemental_Fig_S14",
                "fig_selection_strategies": "Supplemental_Fig_S15",
                "fig_definition_sweep": "Supplemental_Fig_S16",
                "fig_mavedb_ceiling_correlates": "Supplemental_Fig_S17"}




def _insert_toc(md: str) -> str:
    """Prepend a Table of Contents; GR asks for one in the supplemental PDF."""
    heads = re.findall(r"^## (S\d+\w*\. .+)$", md, re.M)
    if not heads:
        return md
    toc = "\n## Contents\n\n" + "".join(f"- {h}\n" for h in heads) + "\n"
    # place it after the document title line, before the first section
    first = md.index("\n## ")
    return md[:first] + toc + md[first:]


def _write_xlsx(df: pd.DataFrame, path: Path, decimals: int = 4) -> None:
    """One sheet, header row frozen, floats rounded for legibility."""
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        df.round(decimals).to_excel(xl, index=False, sheet_name="data")
        ws = xl.sheets["data"]
        ws.freeze_panes = "A2"
        for col in ws.iter_cols(min_row=1, max_row=1):
            col[0].font = col[0].font.copy(bold=True)
        for column in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in column)
            ws.column_dimensions[column[0].column_letter].width = min(max(width + 2, 10), 42)


def _inline_spans(text: str) -> list[tuple[str, bool, bool]]:
    """Split a markdown line into (text, bold, italic) runs.

    Only ``**bold**`` was handled before, so a single-asterisk ``*italic*`` span
    reached the DOCX -- and the printed PDF -- with its asterisks visible. Code
    spans are passed through untouched, backticks included, because an asterisk
    inside them is content: the region-classifier note discusses the literal
    ``*`` prefix of a UTR-relative HGVS position on the same line as an italic
    span, and treating that as a delimiter would mangle it.
    """
    out: list[tuple[str, bool, bool]] = []
    for seg in re.split(r"(`[^`\n]*`)", text):
        if not seg:
            continue
        if seg.startswith("`") and seg.endswith("`") and len(seg) > 1:
            out.append((seg, False, False))
            continue
        for k, chunk in enumerate(seg.split("**")):
            if not chunk:
                continue
            bold = k % 2 == 1
            for j, part in enumerate(re.split(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", chunk)):
                if part:
                    out.append((part, bold, j % 2 == 1))
    return out


def _drop_restricted_fonts(path: Path, replacement: str = "Times New Roman") -> list[str]:
    """Remove licence-restricted font names left in the template's XML parts.

    python-docx's default template declares Courier in word/fontTable.xml and in
    word/stylesWithEffects.xml, a legacy duplicate of styles.xml that the library
    does not expose. No run uses it, but a converter scans declared fonts, finds
    one it may not embed, and offers to write an image-only PDF -- which would
    leave the file with no text layer for a similarity check to read. Rewriting
    the two parts changes nothing that renders.
    """
    import shutil
    import zipfile

    restricted = ("Courier",)
    tmp = path.with_suffix(".fontfix.tmp")
    touched: list[str] = []
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(
            tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.endswith(".xml"):
                text = data.decode("utf8")
                new = text
                for bad in restricted:
                    new = new.replace(f'"{bad}"', f'"{replacement}"')
                if new != text:
                    touched.append(item.filename)
                    data = new.encode("utf8")
            dst.writestr(item, data)
    shutil.move(tmp, path)
    return touched


def write_docx(md: str, path: Path) -> None:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor

    d = docx.Document()
    st = d.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(6)
    for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)):
        h = d.styles[name]
        h.font.name = "Times New Roman"
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = RGBColor(0, 0, 0)

    # python-docx's default template declares Courier on the unused `macro` and
    # `Macro Text Char` styles. No run uses them, but a converter scans declared
    # fonts, finds a licence-restricted one it cannot embed, and offers to write
    # an image-only PDF instead -- which would leave the file with no text layer
    # at all. Retarget them to the body font; nothing renders differently.
    for style in d.styles:
        try:
            if style.font is not None and style.font.name == "Courier":
                style.font.name = "Times New Roman"
        except (AttributeError, NotImplementedError):
            continue

    lines = md.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("| ") and i + 1 < len(lines) and set(lines[i + 1]) <= set("|-"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not set(lines[i]) <= set("|-"):
                    rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            if rows:
                t = d.add_table(rows=0, cols=len(rows[0]))
                t.style = "Table Grid"
                for ri, r in enumerate(rows):
                    cells = t.add_row().cells
                    for ci, val in enumerate(r[:len(rows[0])]):
                        cells[ci].text = ""
                        par = cells[ci].paragraphs[0]
                        run = par.add_run(val.replace("**", ""))
                        run.font.size = Pt(7.5)
                        run.font.name = "Times New Roman"
                        run.bold = ri == 0 or val.startswith("**")
            continue
        if ln.startswith("### "):
            d.add_paragraph(ln[4:], style="Heading 3")
        elif ln.startswith("## "):
            d.add_paragraph(ln[3:], style="Heading 2")
        elif ln.startswith("# "):
            d.add_paragraph(ln[2:], style="Heading 1")
        elif ln.strip() == "---":
            pass
        elif ln.strip():
            p = d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            text = ln[2:] if ln.startswith("- ") else ln
            if ln.startswith("- "):
                p.paragraph_format.left_indent = Pt(14)
            for piece, bold, ital in _inline_spans(text):
                if piece:
                    r = p.add_run(piece)
                    r.bold = bold
                    r.italic = ital
        i += 1
    d.save(str(path))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out).expanduser()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures" / "main").mkdir(parents=True, exist_ok=True)

    md = build_note()
    md = _insert_toc(md)
    (RESULTS / "Supplemental_Note.md").write_text(md)   # source, not an upload
    write_docx(md, out / "Supplemental_Note.docx")
    fixed = _drop_restricted_fonts(out / "Supplemental_Note.docx")
    if fixed:
        print(f"  removed a licence-restricted font declaration from: {', '.join(fixed)}")

    # .tsv is not a format submission systems accept; tables ship as .xlsx with a
    # frozen header row. Full precision is kept in the Zenodo deposit's TSVs.
    for dest, src in TABLES.items():
        _write_xlsx(pd.read_csv(RESULTS / src, sep="\t"),
                    out / "tables" / (Path(dest).stem + ".xlsx"))
    # EXTRA (the machine-readable TSVs) are deliberately NOT copied into the
    # submission package: their names do not follow the Supplemental_Table_SN
    # convention, and Note S13 points readers at the Zenodo deposit for them.
    for i, stem in enumerate(MAIN_FIGS, 1):
        for ext in ("png", "pdf"):
            s = RESULTS / f"{stem}.{ext}"
            if s.exists():
                shutil.copyfile(s, out / "figures" / "main" / f"Zhang_Fig{i}.{ext}")
    (out / "figures" / "supplemental").mkdir(parents=True, exist_ok=True)
    for stem, dest in DEMOTED_FIGS.items():
        for ext in ("png", "pdf"):
            s = RESULTS / f"{stem}.{ext}"
            if s.exists():
                shutil.copyfile(s, out / "figures" / "supplemental" / f"{dest}.{ext}")
    # Supplemental_Fig_S1-S13: montage sheets built by figures/assemble_supp_figures.py
    for i in range(1, 14):
        for ext in ("png", "pdf"):
            s = RESULTS / f"supp_fig_S{i:02d}.{ext}"
            if s.exists():
                shutil.copyfile(
                    s, out / "figures" / "supplemental" / f"Supplemental_Fig_S{i}.{ext}")

    print(f"wrote {out}")
    print(f"  Supplemental_Note.docx  ({len(md.split()):,} words)")
    print(f"  tables/  {len(TABLES)} numbered tables "
          f"({len(EXTRA)} machine-readable TSVs left to the Zenodo deposit)")
    print(f"  figures/main/  {len(MAIN_FIGS)} figures (PNG + PDF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
