"""What AlphaGenome's change of definition does to the territory grid.

From score matrix v3 on, ``alphagenome`` is the official definition (the merged splicing
score of the official notebook, A3.1). Here, on the SNVs (``atlas.population``) and in
the seven territories, each gene's Spearman rho and the pooled rho
(atlas.evaluate_ext's estimator: at least 30 variants per gene, DerSimonian-Laird on
Fisher z with at least two genes) by the older definition (matrix v2's ``alphagenome``,
merged quantile scores over 16 kb, client 0.7.0) beside the official one, and the
difference (official minus older).

Output: results/alphagenome_definition_change_v1.tsv, one row per stratum x gene, and a
row per stratum with gene "pooled".

Usage (PYTHONPATH=src):  python -m atlas.alphagenome_definition_change
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import paths, population
from atlas.evaluate import dl_pool
from atlas.evaluate_ext import TERRITORIES, per_gene, territory_masks

REPO = paths.REPO
RESULTS = paths.RESULTS
OLDER = RESULTS / "score_matrix_atlas_v2.parquet"   # the older definition's column
COLUMNS = ["population", "stratum", "gene", "n_older", "rho_older", "n_official",
           "rho_official", "rho_difference", "k_genes", "ci_lo_older", "ci_hi_older",
           "ci_lo_official", "ci_hi_official"]


def _pool(pg: list[dict]) -> tuple[float, float, float]:
    if len(pg) < 2:
        return np.nan, np.nan, np.nan
    p = dl_pool([r["z"] for r in pg], [r["var"] for r in pg])
    return math.tanh(p["pooled"]), math.tanh(p["ci_lo"]), math.tanh(p["ci_hi"])


def table(matrix: Path | None = None, older: Path = OLDER) -> pd.DataFrame:
    d = population.load(population.SNV, matrix)
    old = pd.read_parquet(older, columns=["variant_id", "alphagenome"])
    d = d.merge(old.rename(columns={"alphagenome": "alphagenome_older"}), on="variant_id",
                how="left", validate="one_to_one")
    rows = []
    for stratum, mask in territory_masks(d).items():
        if stratum not in TERRITORIES:
            continue
        s = d[mask]
        a = per_gene(s.dropna(subset=["alphagenome_older", "functional_pathogenicity"]),
                     "alphagenome_older")
        b = per_gene(s.dropna(subset=["alphagenome", "functional_pathogenicity"]), "alphagenome")
        ga, gb = {r["gene"]: r for r in a}, {r["gene"]: r for r in b}
        for gene in sorted(set(ga) | set(gb)):
            x, y = ga.get(gene), gb.get(gene)
            rows.append({"population": population.SNV, "stratum": stratum, "gene": gene,
                         "n_older": x["n"] if x else np.nan, "rho_older": x["rho"] if x else np.nan,
                         "n_official": y["n"] if y else np.nan,
                         "rho_official": y["rho"] if y else np.nan,
                         "rho_difference": (y["rho"] - x["rho"]) if x and y else np.nan})
        po, lo_o, hi_o = _pool(a)
        pn, lo_n, hi_n = _pool(b)
        rows.append({"population": population.SNV, "stratum": stratum, "gene": "pooled",
                     "n_older": sum(r["n"] for r in a), "rho_older": po,
                     "n_official": sum(r["n"] for r in b), "rho_official": pn,
                     "rho_difference": pn - po, "k_genes": len(b),
                     "ci_lo_older": lo_o, "ci_hi_older": hi_o,
                     "ci_lo_official": lo_n, "ci_hi_official": hi_n})
    t = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("n_older", "n_official", "k_genes"):
        t[c] = t[c].astype("Int64")
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)
    t = table()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / paths.name("alphagenome_definition_change"), sep="\t", index=False)
    pooled = t[t["gene"] == "pooled"]
    print(pooled[["stratum", "k_genes", "rho_older", "rho_official", "rho_difference"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
