"""The deposits' own classes (A4b.5): which variants each deposit calls abnormal, the rho a
predictor could reach that knew the class and nothing within it, and rho within each class;
how well each predictor separates the two classes, as an AUROC (A4c, K19).

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

results/deposit_call_auroc_v1.tsv (A4c, K19): the rows of within_class_rho_v1: n, the
abnormal and normal variants' numbers, the AUROC of the predictor's score for the deposit's
call (abnormal the positive class; the score as rho takes it, higher more damaging; the
Mann-Whitney identity with mid-ranks for ties, atlas.clinical_evidence.roc_auc), empty
unless each class holds at least 10 variants; its 95% interval on the logit scale with the
Hanley-McNeil variance (the delta method, as atlas.clinical_evidence pools AUROC; empty when
the AUROC is 0 or 1); rho, which is within_class_rho's rho_all; selection_overlap
(atlas.predictors). results/deposit_call_auroc_v1.json: per stratum x predictor and group
(core, external, all), the AUROC's and rho's median, minimum and maximum across genes and
the number of genes, a cell with selection_overlap entering none; rho also over the genes
with an AUROC. Checks: the AUROC table's n, n_abnormal, n_normal and rho are
within_class_rho's bit for bit, and the two tables above, computed again, have the
delivered files' bytes. A table already at the output with the same bytes is not
written again; if a check fails, everything goes to <out>/_failed/ and the exit status is 1.

Usage (PYTHONPATH=src):  python -m atlas.deposit_classes [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import clinical_evidence as ce
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
MIN_CLASS = 10              # each class, as atlas.clinical_evidence's MIN_POS and MIN_NEG
CI_Z = 1.96                 # as atlas.evaluate.dl_pool, which clinical_evidence's AUROC uses
AUROC_COLUMNS = ["gene", "group", "stratum", "model", "n", "n_abnormal", "n_normal", "auroc",
                 "auroc_lo", "auroc_hi", "rho", "selection_overlap"]


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


def cells(core: pd.DataFrame, sc: pd.DataFrame):
    """The within-class table's rows: (gene, group, stratum, predictor, the SNVs it scored
    that carry a functional score), n >= MIN_N, the external genes without their all rows."""
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
                yield gene, "external" if ext else "core", stratum, model, c


def within_class_table(core: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene, group, stratum, model, c in cells(core, sc):
        n_all, r_all = _rho(c, model)
        n_n, r_n = _rho(c[~c["abnormal"]], model)
        n_a, r_a = _rho(c[c["abnormal"]], model)
        rows.append({"gene": gene, "group": group, "stratum": stratum, "model": model,
                     "n_all": n_all, "rho_all": r_all, "n_normal": n_n, "rho_normal": r_n,
                     "n_abnormal": n_a, "rho_abnormal": r_a})
    return pd.DataFrame(rows, columns=RHO_COLUMNS)


# ==========================================================================
# the predictors against the deposits' calls (A4c, K19)
# ==========================================================================
def auroc(abnormal, score) -> float:
    """AUROC with the abnormal variants the positive class (atlas.clinical_evidence.roc_auc:
    the Mann-Whitney identity, mid-ranks for ties); empty unless each class holds at least
    MIN_CLASS variants."""
    ab = np.asarray(abnormal, bool)
    if ab.sum() < MIN_CLASS or (~ab).sum() < MIN_CLASS:
        return float("nan")
    return ce.roc_auc(ab.astype(int), np.asarray(score, float))


def auroc_interval(auc: float, n_pos: int, n_neg: int) -> tuple[float, float]:
    """The 95% interval on the logit scale with the Hanley-McNeil variance, by the delta
    method, as atlas.clinical_evidence pools AUROC; empty when the AUROC is empty, 0 or 1."""
    if not (0 < auc < 1):
        return float("nan"), float("nan")
    z = ce._logit(auc)
    se = math.sqrt(ce.hanley_mcneil_var(auc, n_pos, n_neg)) / (auc * (1 - auc))
    inv = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
    return inv(z - CI_Z * se), inv(z + CI_Z * se)


def call_auroc_table(core: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene, group, stratum, model, c in cells(core, sc):
        n, rho = _rho(c, model)
        ab = c["abnormal"].to_numpy(bool)
        n_ab, n_no = int(ab.sum()), int((~ab).sum())
        auc = auroc(ab, c[model])
        lo, hi = auroc_interval(auc, n_ab, n_no)
        rows.append({"gene": gene, "group": group, "stratum": stratum, "model": model,
                     "n": n, "n_abnormal": n_ab, "n_normal": n_no, "auroc": auc,
                     "auroc_lo": lo, "auroc_hi": hi, "rho": rho,
                     "selection_overlap": predictors.is_overlap(model, gene)})
    return pd.DataFrame(rows, columns=AUROC_COLUMNS)


def _span(v: pd.Series, genes: pd.Series) -> dict:
    keep = v.notna()
    v, genes = v[keep], genes[keep]
    if not len(v):
        return {"n_genes": 0, "median": None, "min": None, "max": None, "genes": []}
    return {"n_genes": int(len(v)), "median": float(v.median()), "min": float(v.min()),
            "max": float(v.max()), "genes": genes.tolist()}


def auroc_summary(t: pd.DataFrame) -> list[dict]:
    """Per stratum x predictor and group (core, external, all): the AUROC's and rho's median,
    minimum and maximum across genes, a cell with selection_overlap entering none; rho also
    over the genes with an AUROC."""
    flag = t["selection_overlap"].astype(bool)
    out = []
    for stratum in STRATA:
        for model in predictors.PREDICTORS:
            here = t[(t["stratum"] == stratum) & (t["model"] == model)]
            for group in ("core", "external", "all"):
                g = here if group == "all" else here[here["group"] == group]
                if not len(g):
                    continue
                left_out = g[flag[g.index]]
                g = g[~flag[g.index]]
                a = g[g["auroc"].notna()]
                out.append({"stratum": stratum, "model": model, "group": group,
                            "auroc": _span(g["auroc"], g["gene"]),
                            "rho": _span(g["rho"], g["gene"]),
                            "rho_genes_with_auroc": _span(a["rho"], a["gene"]),
                            "left_out_selection_overlap": left_out["gene"].tolist()})
    return out


def _bits_equal(a: pd.Series, b: pd.Series) -> bool:
    x, y = a.to_numpy(float), b.to_numpy(float)
    return bool(((x.view(np.uint64) == y.view(np.uint64)) | (np.isnan(x) & np.isnan(y))).all())


def auroc_checks(t: pd.DataFrame, rho: pd.DataFrame, texts: dict[str, str]) -> dict:
    """(1) n, n_abnormal, n_normal and rho are within_class_rho's bit for bit: the table just
    computed, and the delivered file where it is there; (2) the two tables computed again have
    the delivered files' bytes."""
    keys = ["gene", "group", "stratum", "model"]
    pairs = {"n": "n_all", "n_abnormal": "n_abnormal", "n_normal": "n_normal", "rho": "rho_all"}
    one = {}
    sources = {"computed_here": rho}
    delivered = paths.path("within_class_rho")
    if delivered.exists():
        sources[delivered.name] = pd.read_csv(delivered, sep="\t", float_precision="round_trip")
    for label, w in sources.items():
        same_rows = (len(w) == len(t) and (w[keys].astype(str).to_numpy()
                                           == t[keys].astype(str).to_numpy()).all())
        one[label] = {"rows": int(len(w)), "rows_here": int(len(t)), "same_rows": bool(same_rows),
                      "columns_differing": [] if not same_rows else
                      [c for c, wc in pairs.items() if not _bits_equal(t[c], w[wc])]}
        one[label]["holds"] = bool(same_rows and not one[label]["columns_differing"])
    two = {}
    for key, text in texts.items():
        p = paths.path(key)
        two[p.name] = (None if not p.exists()
                       else p.read_bytes() == text.encode("utf-8"))
    return {"same_as_within_class_rho": one | {"holds": all(v["holds"] for v in one.values())},
            "tables_computed_again_same_bytes": two | {
                "holds": all(v is not False for v in two.values())}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(paths.RESULTS))
    args = ap.parse_args(argv)
    core = core_variants()
    dep, sc = external_variants()
    classes = class_table(core, dep)
    rho = within_class_table(core, sc)
    calls = call_auroc_table(core, sc)
    texts = {"deposit_classes": classes.to_csv(sep="\t", index=False),
             "within_class_rho": rho.to_csv(sep="\t", index=False)}
    checks = auroc_checks(calls, rho, texts)
    holds = all(c["holds"] for c in checks.values())
    out = Path(args.out) if holds else Path(args.out) / "_failed"
    out.mkdir(parents=True, exist_ok=True)
    for key, text in texts.items():          # a table already there with these bytes is
        p = out / paths.name(key)            # not written again
        if not (p.exists() and p.read_bytes() == text.encode("utf-8")):
            p.write_text(text, encoding="utf-8")
    calls.to_csv(out / paths.name("deposit_call_auroc"), sep="\t", index=False)
    rec = {"definition": {
               "rows": paths.name("within_class_rho") + "'s: gene x stratum x predictor, SNVs "
                       f"the predictor scored that carry a functional score, n >= {MIN_N}; the "
                       "external genes without their all rows",
               "calls": RULES,
               "auroc": "abnormal the positive class; the score as rho takes it (higher more "
                        "damaging); the Mann-Whitney identity, mid-ranks for ties "
                        f"(atlas.clinical_evidence.roc_auc); empty unless each class holds at "
                        f"least {MIN_CLASS} variants",
               "interval": f"logit(AUROC) +/- {CI_Z} x sqrt(Hanley-McNeil variance) / "
                           "(AUROC (1 - AUROC)), back-transformed (the delta method, as "
                           "atlas.clinical_evidence pools AUROC); empty when the AUROC is 0 or 1",
               "rho": paths.name("within_class_rho") + "'s rho_all",
               "across_genes": "a cell with selection_overlap (atlas.predictors."
                               "SELECTION_OVERLAP) enters no quantity across genes",
               "selection_overlap": predictors.SELECTION_OVERLAP},
           "checks": checks,
           "by_stratum_and_predictor": auroc_summary(calls)}
    (out / paths.name("deposit_call_auroc_summary")).write_text(
        json.dumps(_jsonable(rec), indent=1) + "\n")
    show = classes[classes["stratum"].isin(["splice_3_10", "splice_11_50"])]
    print(show.drop(columns=["rule"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"{len(rho)} within-class rows; {len(calls)} AUROC rows, "
          f"{int(calls['auroc'].notna().sum())} with an AUROC")
    print(json.dumps(_jsonable(checks), indent=1))
    return 0 if holds else 1


def _jsonable(obj):
    from atlas.rank_ceiling import jsonable
    return jsonable(obj)


if __name__ == "__main__":
    raise SystemExit(main())
