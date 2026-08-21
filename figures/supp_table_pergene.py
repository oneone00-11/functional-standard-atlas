"""Supplementary Table S3 — full per-gene rho dump from eval JSONs.

Long-format TSV: model, stratum, gene, n, rho, plus pooled meta-analysis
rows (gene = "POOLED"). Numbers read only from results/eval_*.json
(CONVENTIONS.md rule 5); this table is also the drafting source for prose.

Writes results/tableS3_per_gene_rho.{tsv,md}.

Usage (PYTHONPATH=src):  python figures/supp_table_pergene.py
"""

from __future__ import annotations

import json

import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

MODELS = [  # (eval key, label) — heatmap order
    ("alphagenome", "AlphaGenome"),
    ("spliceai_ds", "SpliceAI"),
    ("pangolin_score", "Pangolin"),
    ("cadd", "CADD"),
    ("alphamissense", "AlphaMissense"),
    ("evo2", "Evo2-7B"),
    ("gpn_msa", "GPN-MSA"),
    ("nucleotide_transformer", "NT-v2-500M"),
    ("phylop100way", "phyloP-100way"),
    ("phastcons100way", "phastCons-100way"),
    ("gnomad_af_global", "gnomAD AF (global)"),
    ("gnomad_af_popmax", "gnomAD AF (popmax)"),
]

STRATA = [
    ("all", "All"),
    ("coding_or_utr", "Coding / UTR"),
    ("splice_region", "Splice region (all)"),
    ("splice_1_2", "Splice ±1–2"),
    ("splice_3_10", "Splice 3–10 bp"),
    ("splice_11_50", "Splice 11–50 bp"),
    ("splice_deep", "Splice >50 bp"),
    ("clinvar_recorded", "ClinVar-recorded"),
]


def main() -> None:
    rows = []
    for key, label in MODELS:
        path = RESULTS / f"eval_{key}.json"
        if not path.exists():
            print(f"note: {path.name} missing — {label} skipped")
            continue
        data = json.loads(path.read_text())
        for skey, slabel in STRATA:
            st = data["strata"].get(skey)
            if not st:
                continue
            for g in st.get("per_gene", []):
                if g.get("rho") is None:
                    continue
                rows.append({
                    "model": label, "stratum": slabel, "gene": g["gene"],
                    "n": g["n"], "rho": round(g["rho"], 4),
                    "ci_lo": "", "ci_hi": "", "i2_pct": "",
                })
            p = st.get("pooled")
            if p:
                rows.append({
                    "model": label, "stratum": slabel, "gene": "POOLED",
                    "n": st.get("n_variants", ""),
                    "rho": round(p["pooled_rho"], 4),
                    "ci_lo": round(p["rho_ci_lo"], 4),
                    "ci_hi": round(p["rho_ci_hi"], 4),
                    "i2_pct": round(p["i2_pct"], 1),
                })
    df = pd.DataFrame(rows)
    tsv = RESULTS / "tableS3_per_gene_rho.tsv"
    df.to_csv(tsv, sep="\t", index=False)
    print("wrote", tsv, f"({len(df)} rows)")

    # Compact pooled-only markdown for quick drafting reference (hand-rolled
    # emitter; tabulate is not a repo dependency).
    pooled = df[df["gene"] == "POOLED"].copy()
    wide = pooled.pivot(index="model", columns="stratum", values="rho")
    wide = wide.reindex([lb for _k, lb in MODELS if lb in wide.index])
    cols = ["model"] + list(wide.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for model, row in wide.iterrows():
        cells = [f"{v:.3f}" if pd.notna(v) else "—" for v in row]
        lines.append("| " + model + " | " + " | ".join(cells) + " |")
    md = RESULTS / "tableS3_pooled_summary.md"
    md.write_text("\n".join(lines) + "\n")
    print("wrote", md)


if __name__ == "__main__":
    main()
