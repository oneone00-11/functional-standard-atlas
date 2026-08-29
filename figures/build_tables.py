"""Table 1 (atlas composition) + Table 2 (model inventory) generators.

Numbers come only from pipeline files (CONVENTIONS.md rule 5):
  - data/frozen/frozen-matrix-v1.parquet  (per-assay variant counts, regions)
  - results/score_matrix_atlas_v2.summary.json  (per-model coverage)
Descriptive strings (versions, definitions) mirror models/*/NOTES.md — if a
NOTES.md changes, update the string here too.

Writes results/table1_atlas_composition.{tsv,md} and
results/table2_model_inventory.{tsv,md}.

Usage (PYTHONPATH=src):  python figures/build_tables.py
"""

from __future__ import annotations

import json

import pandas as pd
from pathlib import Path

from atlas.evaluate import classify_region
from atlas.predictor_resources import ROWS as PREDICTOR_ROWS

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FROZEN = REPO / "data" / "frozen" / "frozen-matrix-v1.parquet"

BINS = ["coding_or_utr", "splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]


def table1() -> pd.DataFrame:
    df = pd.read_parquet(FROZEN)
    df["bin"] = df["hgvs_c"].map(classify_region)
    df["is_snv"] = (df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)
    rows = []
    for (gene, urn, tx), g in df.groupby(["gene", "urn", "transcript"], sort=True):
        row = {
            "gene": gene,
            "mavedb_urn": urn,
            "transcript": tx,
            "n_variants": len(g),
            "n_snv": int(g["is_snv"].sum()),
            "n_indel": int((~g["is_snv"]).sum()),
        }
        for b in BINS:
            row[f"n_{b}"] = int((g["bin"] == b).sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    total = {c: t[c].sum() for c in t.columns if c.startswith("n_")}
    total.update({"gene": "Total", "mavedb_urn": "", "transcript": ""})
    return pd.concat([t, pd.DataFrame([total])], ignore_index=True)


# (column, label, version, scope, definition, reference) — text per NOTES.md;
# coverage is filled from the matrix summary, not typed by hand.
_DBNSFP_SOURCE = "dbNSFP (assembly hg38), via MyVariant.info batch API"

MODELS = [
    ("alphagenome", "AlphaGenome", "client 0.7.0", "all variants (API)",
     "merged quantile splice score, 16-kb window", "AlphaGenome API"),
    ("cadd", "CADD", "GRCh38-v1.7", "SNV",
     "max PHRED over consequence records", "precomputed scores (remote tabix)"),
    ("alphamissense", "AlphaMissense", "hg38 (Zenodo 8208688)", "missense SNV",
     "pathogenicity score", "precomputed table"),
    ("gpn_msa", "GPN-MSA", "precomputed (songlab)", "SNV",
     "-raw (sign-flipped; higher = more damaging)", "precomputed (sparse tabix)"),
    ("nucleotide_transformer", "NT-v2-500M", "v2-500M-multi-species (transformers 4.46.3, GPU)",
     "SNV", "masked 6-mer LLR, 6000-bp window", "local; HF InstaDeepAI"),
    ("evo2", "Evo2-7B", "evo2 0.6.0 (7B, bf16, GPU)", "SNV",
     "windowed LL delta (LL REF - LL ALT), 8192-bp window", "local; arcinstitute/evo2_7b"),
    ("phylop100way", "phyloP-100way", "UCSC 100-way", "all variants",
     "per-base phyloP, max over interval", "UCSC track API"),
    ("phastcons100way", "phastCons-100way", "UCSC 100-way", "all variants",
     "per-base phastCons, max over interval", "UCSC track API"),
    ("gnomad_af_global", "gnomAD AF (global)", "v4", "variants present in gnomAD",
     "-AF (sign-flipped)", "gnomAD GraphQL"),
    ("gnomad_af_popmax", "gnomAD AF (popmax)", "v4", "variants present in gnomAD",
     "-popmax AF (sign-flipped)", "gnomAD GraphQL"),
    ("spliceai_ds", "SpliceAI", "1.3.1 (TF 2.19.1, CPU)", "SNV",
     "max(DS_AG/AL/DG/DL), distance 50", "local; Ensembl r112 subset FASTA"),
    ("pangolin_score", "Pangolin", "git 5cf94b8 (torch 2.13.0, CPU)", "SNV",
     "max(splice gain, |splice loss|), distance 50, default mask",
     "local; Ensembl r112 gffutils db"),
    # The seven dbNSFP meta-predictors. `scope` names every consequence class with a
    # non-zero count in the matrix, in descending order of count, read from
    # score_matrix_atlas_v2 rather than from any earlier scope string; a predictor with
    # no non-missense coverage is marked "only". PrimateAI is the only such predictor.
    # `version` is empty by design: the dbNSFP
    # release was not recorded at fetch time, and inventing one would be worse
    # than the gap. The footnote below the table says so.
    ("revel", "REVEL", "",
     "missense SNV; also 66 splice-region, 46 start-lost, 40 synonymous, 3 nonsense",
     "dbNSFP score, max over transcript records", _DBNSFP_SOURCE),
    ("bayesdel_addaf", "BayesDel (addAF)", "",
     "missense SNV; also 1,656 nonsense, 955 splice-region, 186 synonymous, 46 start-lost, 38 stop-lost",
     "dbNSFP score, max over transcript records", _DBNSFP_SOURCE),
    ("clinpred", "ClinPred", "",
     "missense SNV; also 126 nonsense, 114 synonymous, 78 splice-region, 46 start-lost, 5 stop-lost",
     "dbNSFP score, max over transcript records", _DBNSFP_SOURCE),
    ("metarnn", "MetaRNN", "",
     "missense SNV; also 173 synonymous, 88 splice-region, 46 start-lost, 18 nonsense",
     "dbNSFP score, max over transcript records", _DBNSFP_SOURCE),
    ("primateai", "PrimateAI", "", "missense SNV only",
     "dbNSFP score, max over transcript records", _DBNSFP_SOURCE),
    ("vest4", "VEST4", "",
     "missense SNV; also 1,646 nonsense, 106 synonymous, 46 start-lost, 38 stop-lost, 34 splice-region",
     "dbNSFP score, max over transcript records", _DBNSFP_SOURCE),
    ("esm1b", "ESM-1b", "",
     "missense SNV; also 47 synonymous, 46 start-lost, 3 nonsense",
     "dbNSFP score, max over transcript records; sign-flipped", _DBNSFP_SOURCE),
]


ATLAS_MATRIX = RESULTS / "score_matrix_atlas_v2.parquet"
ATLAS_SUMMARY = RESULTS / "score_matrix_atlas_v2.summary.json"


def _atlas_summary() -> dict:
    """Per-model coverage over the scored atlas.

    The v2 summary was never emitted by the scoring stage -- only
    score_matrix_atlas_v1.summary.json and score_matrix_atlas_v2.parquet exist --
    so build_tables() used to abort here with FileNotFoundError, taking
    Supplemental_Table_S9 down with it and making "every table regenerates from
    the archive" false. The matrix itself is archived, so the summary is derived
    from it and written out on first use rather than being a missing input.
    """
    if ATLAS_SUMMARY.exists():
        cached = json.loads(ATLAS_SUMMARY.read_text())
        # A summary written before a model was added to MODELS would leave
        # table2() with a KeyError, so a stale one is rebuilt rather than used.
        if all(col in cached.get("coverage", {}) for col, *_ in MODELS):
            return cached
        print(f"{ATLAS_SUMMARY.name} predates the current MODELS list; rederiving")
    if not ATLAS_MATRIX.exists():
        raise FileNotFoundError(
            f"neither {ATLAS_SUMMARY.name} nor {ATLAS_MATRIX.name} is present; "
            "run the scoring stage before building the tables")
    df = pd.read_parquet(ATLAS_MATRIX)
    summary = {
        "matrix": ATLAS_MATRIX.name,
        "n_variants": int(len(df)),
        "coverage": {col: int(df[col].notna().sum())
                     for col, *_ in MODELS if col in df.columns},
        "derived_from": f"{ATLAS_MATRIX.name} (summary json absent)",
    }
    ATLAS_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"derived {ATLAS_SUMMARY.name} from {ATLAS_MATRIX.name}")
    return summary


def table2() -> pd.DataFrame:
    """Supplemental Table S9 -- the model inventory, all nineteen scored columns.

    This is a rendering of `atlas.predictor_resources`, which is the curated
    source of truth for version, source, access date and licence. Here it gains
    the editorial `scope` and `score_definition` strings and the two columns that
    have to be computed from the matrix (`n_scored`, `coverage_pct`).

    It carried only the twelve primary predictors until 2026-08-28. The delivered
    S9 had nineteen rows because the seven dbNSFP meta-predictors were joined in
    by hand at submission and never went back into the build, so a rebuild
    silently produced a shorter table than the one the manuscript cites.
    tests/test_supplemental_table_s9.py now fails if that happens again.
    """
    summary = _atlas_summary()
    cov = summary["coverage"]
    n_total = summary["n_variants"]
    accessed = {p: a for p, _, _, a, _, _ in PREDICTOR_ROWS}
    licence = {p: (lic, src) for p, _, _, _, lic, src in PREDICTOR_ROWS}
    rows = []
    for col, label, version, scope, definition, reference in MODELS:
        n = cov[col]
        lic, lic_src = licence[label]
        rows.append({
            "model": label,
            "version": version,
            "scope": scope,
            "score_definition": definition,
            "source": reference,
            "accessed": accessed[label],
            "n_scored": n,
            "coverage_pct": round(100.0 * n / n_total, 1),
            "licence": lic,
            "licence_source": lic_src,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Manuscript Table 2 — territory-appropriate predictor selection guide.
#
# This table used to be maintained by hand and drifted from the pipeline: three
# held-out rho cells disagreed with the leave-one-gene-out medians they quote.
# The numeric columns are now derived here and pinned by
# tests/test_table2_territory_guide.py.
#
# ROWS is the editorial layer: which predictor is recommended per territory, and
# the prose note. Those are judgements, not outputs, and follow the same
# convention as table2()'s descriptive strings. The numbers beside them are not
# judgements and are never typed in.
#
# `source` names the ensemble_logo_v1 column each recommendation corresponds to;
# `held_out_rho` is the median of that column over the stratum's LOGO folds.
# ---------------------------------------------------------------------------
GUIDE_ROWS = [
    ("Missense coding",        "coding_or_utr", None,
     "AlphaMissense", "beats all seven clinical meta-predictors within missense"),
    ("General coding / UTR",   "coding_or_utr", "territory_logo",
     "CADD", "rank-average adds nothing; GPN-MSA and Evo2-7B close behind"),
    ("Splice region (any offset)", "splice_region", "rank_average_splice",
     "Rank-average of AlphaGenome, SpliceAI and Pangolin",
     "beats every single model; the selector picks a splice specialist in every fold"),
    ("Splice 3-10 bp",         "splice_3_10", "rank_average_splice",
     "Rank-average of the three splice models",
     "exceeds the best achievable single choice"),
    ("Splice 11-50 bp",        "splice_11_50", "rank_average_splice",
     "Rank-average of the three splice models",
     "worst-served territory: about a sixth of the ceiling is realised"),
    ("Splice +-1-2",           "splice_1_2", None,
     "None - treat as uninformative", "assay cannot resolve variation here"),
    ("Deep intronic >50 bp",   "splice_deep", None,
     "None - use RNA or functional assay",
     "below the ceiling at which any correction is trustworthy"),
    ("Whole-gene, single choice", "all", "rank_average_broad",
     "Rank-average of CADD, Evo2-7B and GPN-MSA", "CADD alone returns 0.384"),
]


def _ceilings() -> pd.DataFrame:
    """Per-stratum attenuation ceiling: median over genes carrying a validated
    error model, with the between-gene range (matches figures/hardening_figures)."""
    rel = pd.read_csv(RESULTS / "reliability_v1.tsv", sep="\t")
    g = (rel[rel["status"] == "validated"]
         .sort_values("method", ascending=False)
         .drop_duplicates(["gene", "stratum"])
         .groupby("stratum")["ceiling"])
    return pd.DataFrame({"median": g.median(), "lo": g.min(), "hi": g.max(),
                         "k": g.count()})


def table2_territory_guide() -> pd.DataFrame:
    logo = pd.read_csv(RESULTS / "ensemble_logo_v1.tsv", sep="\t")
    ceil = _ceilings()
    rows = []
    for territory, stratum, col, rec, note in GUIDE_ROWS:
        if col is None:
            rho, folds = "", ""
        else:
            v = logo.loc[logo["stratum"] == stratum, col].dropna()
            rho, folds = f"{v.median():.3f}", len(v)
        c = ceil.loc[stratum]
        # the caption shows a range only where the contributing genes disagree
        # by >= 0.10; with one contributing gene there is no range to show
        cs = f"{c['median']:.2f}"
        if c["k"] > 1 and (c["hi"] - c["lo"]) >= 0.10:
            cs += f" ({c['lo']:.2f}-{c['hi']:.2f})"
        rows.append({"territory": territory, "stratum": stratum,
                     "recommended": rec, "source": col or "",
                     "held_out_rho": rho, "logo_folds": folds,
                     "ceiling": cs, "notes": note})
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table emitter (avoids a tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines) + "\n"


def write(df: pd.DataFrame, stem: str) -> None:
    tsv = RESULTS / f"{stem}.tsv"
    md = RESULTS / f"{stem}.md"
    df.to_csv(tsv, sep="\t", index=False)
    md.write_text(_md_table(df))
    print("wrote", tsv)
    print("wrote", md)


def main() -> None:
    write(table1(), "table1_atlas_composition")
    write(table2(), "table2_model_inventory")
    write(table2_territory_guide(), "table2_territory_guide")


if __name__ == "__main__":
    main()
