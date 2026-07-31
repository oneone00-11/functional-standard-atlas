"""Table 1 (atlas composition) + Table 2 (model inventory) generators.

Numbers come only from pipeline files (AGENTS.md rule 5):
  - data/frozen/frozen-matrix-v1.parquet  (per-assay variant counts, regions)
  - results/score_matrix_atlas_v1.summary.json  (per-model coverage)
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
MODELS = [
    ("alphagenome", "AlphaGenome", "client 0.7.0", "all variants (API)",
     "merged quantile splice score, 16-kb window", "AlphaGenome API"),
    ("cadd", "CADD", "GRCh38-v1.7", "SNV",
     "max PHRED over consequence records", "precomputed scores (remote tabix)"),
    ("alphamissense", "AlphaMissense", "hg38 (Zenodo 8208688)", "missense SNV",
     "pathogenicity score", "precomputed table"),
    ("gpn_msa", "GPN-MSA", "precomputed (songlab)", "SNV",
     "-raw (sign-flipped; higher = more damaging)", "precomputed (sparse tabix)"),
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
]


def table2() -> pd.DataFrame:
    summary = json.loads((RESULTS / "score_matrix_atlas_v1.summary.json").read_text())
    cov = summary["coverage"]
    n_total = summary["n_variants"]
    rows = []
    for col, label, version, scope, definition, reference in MODELS:
        n = cov[col]
        rows.append({
            "model": label,
            "version": version,
            "scope": scope,
            "score_definition": definition,
            "source": reference,
            "n_scored": n,
            "coverage_pct": round(100.0 * n / n_total, 1),
        })
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


if __name__ == "__main__":
    main()
