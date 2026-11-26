"""The deposits' own classes (A4b.5): which variants each deposit calls abnormal, the rho a
predictor could reach that knew the class and nothing within it, and rho within each class.

Rules ("abnormal"; every other variant of the deposit is normal):
  BARD1, PALB2   atlas.assay_aux's posterior label_abnormal above 0.5;
  CTCF, SFPQ, RAD51D, XRCC2, SBDS   the deposit's gmm_density_abnormal above 0.5;
  BAP1           the deposit's processed_BH_FDR_continuous below 0.01 and
                 processed_LFC_continuous below 0 (the score falls);
  RAD51C         atlas.assay_aux's label_class fast depleted or slow depleted.
BRCA1, BRCA2 and VHL deposit no classes and are not listed.

results/deposit_classes_v1.tsv: gene x stratum (atlas.reliability.STRATA; SNVs with a
functional score: the core genes' from the score matrix, the external genes' from their
deposits, coding variants included, y = -score): n, the abnormal variants' number and
share, and the class bound: the Spearman rho of the measurement with an ordering that
ranks the abnormal variants by their measurement and ties every other variant below them
(empty when no variant is abnormal). It is the rho a predictor could reach that knew the
deposit's class and nothing within it. With effects below the deposit's threshold inside a
class the true ceiling is higher, so it is not a strict bound.

results/within_class_rho_v1.tsv: gene x stratum x predictor (the core genes' 23, the
external genes' 13 columns, SNVs the predictor scored that carry a functional score): n and
rho over all of them, over the normal class and over the abnormal class; empty where n is
below 30. The external genes' predictors score intronic SNVs only, so their all rows,
which would repeat splice_region, are not written (as G18 does for the matched ceilings).

Usage (PYTHONPATH=src):  python -m atlas.deposit_classes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import paths, population, predictors
from atlas.reliability import STRATA, _stratum_mask

MIN_N = 30
LABEL_ABNORMAL = 0.5
GMM_ABNORMAL = 0.5
BAP1_FDR = 0.01
RAD51C_ABNORMAL = ("fast depleted", "slow depleted")
CORE_CLASSIFIED = ("BARD1", "PALB2", "BAP1", "RAD51C")
EXTERNAL = ("CTCF", "SFPQ", "RAD51D", "XRCC2", "SBDS")
GENES = CORE_CLASSIFIED + EXTERNAL
RULES = {"BARD1": "assay_aux label_abnormal > 0.5", "PALB2": "assay_aux label_abnormal > 0.5",
         "BAP1": "deposit processed_BH_FDR_continuous < 0.01 and processed_LFC_continuous < 0",
         "RAD51C": "assay_aux label_class fast depleted or slow depleted",
         **{g: "deposit gmm_density_abnormal > 0.5" for g in EXTERNAL}}
CLASS_COLUMNS = ["gene", "group", "stratum", "n", "n_abnormal", "share_abnormal",
                 "class_bound", "rule"]
RHO_COLUMNS = ["gene", "group", "stratum", "model", "n_all", "rho_all", "n_normal",
               "rho_normal", "n_abnormal", "rho_abnormal"]


def spearman(a, b) -> float:
    from atlas.rank_ceiling import spearman as sp
    return sp(a, b)


def class_bound(y, abnormal) -> float:
    """rho of y with the ordering: the abnormal variants by y, the rest tied below."""
    y, ab = np.asarray(y, float), np.asarray(abnormal, bool)
    if not ab.any():
        return np.nan
    x = np.where(ab, y, y.min() - 1.0)
    return spearman(x, y)


# ==========================================================================
# the variants and their classes
# ==========================================================================
def core_variants() -> pd.DataFrame:
    """The four classified core genes' SNVs with a functional score, their class and the
    23 predictors' scores."""
    m = population.load(population.SNV)
    m = m[m["gene"].isin(CORE_CLASSIFIED) & m["functional_pathogenicity"].notna()]
    aux = pd.read_parquet(paths.path("assay_aux"),
                          columns=["variant_id", "gene", "label_abnormal", "label_class"])
    m = m.merge(aux, on=["variant_id", "gene"], how="left")
    bap1 = pd.read_csv(paths.REPO / "data" / "raw" / "mavedb" / "urn_mavedb_00000662-0-1"
                       / "scores.csv", low_memory=False)
    fdr = pd.to_numeric(bap1["processed_BH_FDR_continuous"], errors="coerce")
    lfc = pd.to_numeric(bap1["processed_LFC_continuous"], errors="coerce")
    bap1_ab = dict(zip(bap1["accession"], (fdr < BAP1_FDR) & (lfc < 0)))
    ab = np.select(
        [m["gene"].isin(["BARD1", "PALB2"]), m["gene"].eq("BAP1"), m["gene"].eq("RAD51C")],
        [m["label_abnormal"] > LABEL_ABNORMAL, m["variant_id"].map(bap1_ab).fillna(False),
         m["label_class"].isin(RAD51C_ABNORMAL)], default=False)
    m["abnormal"] = ab.astype(bool)
    m["group"] = "core"
    m["y"] = m["functional_pathogenicity"]
    return m


def external_variants() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the five deposits' SNVs with a score and their class; external_scores' intronic
    SNVs with their class and 13 columns)."""
    from atlas.external_reliability import EXTERNAL as SPECS, deposit_population, load_deposit
    rows, cls = [], {}
    for urn, spec in SPECS.items():
        if spec["gene"] not in EXTERNAL:
            continue
        d = load_deposit(urn)
        d = d[deposit_population(d["hgvs_c"]) == population.SNV]
        y = -pd.to_numeric(d["score"], errors="coerce")
        ab = pd.to_numeric(d["gmm_density_abnormal"], errors="coerce") > GMM_ABNORMAL
        x = pd.DataFrame({"gene": spec["gene"], "group": "external", "id": d["accession"],
                          "region": d["region"], "y": y, "abnormal": ab}).dropna(subset=["y"])
        rows.append(x)
        cls |= dict(zip(x["id"], x["abnormal"]))
    dep = pd.concat(rows, ignore_index=True)
    sc = pd.read_parquet(paths.path("external_scores"))
    sc = sc[sc["gene"].isin(EXTERNAL) & sc["functional_pathogenicity"].notna()].copy()
    sc["abnormal"] = sc["variant_urn"].map(cls).fillna(False).astype(bool)
    sc["region"] = sc["stratum"]
    sc["group"] = "external"
    sc["y"] = sc["functional_pathogenicity"]
    return dep, sc


# ==========================================================================
# the tables
# ==========================================================================
def class_table(core: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene in GENES:
        g = core[core["gene"] == gene] if gene in CORE_CLASSIFIED else dep[dep["gene"] == gene]
        for stratum in STRATA:
            c = g[_stratum_mask(g["region"], stratum)]
            if not len(c):
                continue
            ab = c["abnormal"].to_numpy(bool)
            rows.append({"gene": gene, "group": g["group"].iloc[0], "stratum": stratum,
                         "n": int(len(c)), "n_abnormal": int(ab.sum()),
                         "share_abnormal": float(ab.mean()),
                         "class_bound": class_bound(c["y"], ab) if len(c) >= 2 else np.nan,
                         "rule": RULES[gene]})
    return pd.DataFrame(rows, columns=CLASS_COLUMNS)


def _rho(c: pd.DataFrame, model: str) -> tuple[int, float]:
    if len(c) < MIN_N:
        return int(len(c)), np.nan
    return int(len(c)), float(c[model].corr(c["y"], method="spearman"))


def within_class_table(core: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene in GENES:
        ext = gene in EXTERNAL
        g = sc[sc["gene"] == gene] if ext else core[core["gene"] == gene]
        models = predictors.EXTERNAL if ext else predictors.PREDICTORS
        for stratum in STRATA:
            if ext and stratum == "all":
                continue
            s = g[_stratum_mask(g["region"], stratum)]
            for model in models:
                if model not in s.columns:
                    continue
                c = s[s[model].notna()]
                if len(c) < MIN_N:
                    continue
                n_all, r_all = _rho(c, model)
                n_n, r_n = _rho(c[~c["abnormal"]], model)
                n_a, r_a = _rho(c[c["abnormal"]], model)
                rows.append({"gene": gene, "group": "external" if ext else "core",
                             "stratum": stratum, "model": model, "n_all": n_all,
                             "rho_all": r_all, "n_normal": n_n, "rho_normal": r_n,
                             "n_abnormal": n_a, "rho_abnormal": r_a})
    return pd.DataFrame(rows, columns=RHO_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(paths.RESULTS))
    args = ap.parse_args(argv)
    out = Path(args.out)
    core = core_variants()
    dep, sc = external_variants()
    classes = class_table(core, dep)
    classes.to_csv(out / paths.name("deposit_classes"), sep="\t", index=False)
    rho = within_class_table(core, sc)
    rho.to_csv(out / paths.name("within_class_rho"), sep="\t", index=False)
    show = classes[classes["stratum"].isin(["splice_3_10", "splice_11_50"])]
    print(show.drop(columns=["rule"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"{len(rho)} within-class rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
