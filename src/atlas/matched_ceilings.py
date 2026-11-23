"""Attenuation ceilings matched to the variants each correlation uses, with bootstrap
intervals; and the realisation of the external genes.

v5 estimates one ceiling per gene x stratum from every variant with error information
(results/reliability_v1.tsv), and each predictor's rho from the variants that predictor
scored (results/attenuation_v1.tsv). Where the two sets differ, realisation's numerator
and denominator come from different variants: BARD1's +-1-2 bp stratum holds 147 variants,
28 of them 3 bp deletions across the splice site, while SpliceAI, Pangolin and CADD score
only the 119 SNVs. Here the ceiling is estimated again on exactly the variants that enter
each rho. Nothing v5 wrote is rewritten; its functions are imported, not changed.

Genes and estimators (v5's):
  core      BRCA1 by replicates (Spearman between the two replicate fitness scores, raised
            to their mean by Spearman-Brown, atlas.reliability.spearman_brown), BARD1 and
            PALB2 by validated SE (1 - mean(SE^2) / var(functional_pathogenicity), clipped to
            [0, 1]); results/score_matrix_atlas_v2.parquet with results/assay_aux_v1.parquet;
            the nineteen predictors of atlas.reliability.MODELS; the seven strata of
            atlas.reliability.STRATA
  external  CTCF, SFPQ, RAD51D, XRCC2 and SBDS by SE (checked against each deposit's 95%
            interval in atlas.external_reliability); results/external_scores_v1.parquet;
            its nine score columns; the four intronic strata and splice_region
Scopes:
  scored    the variants the predictor scored that carry a functional score and the error
            information: v5's rho set, less any variant without error information
  snv       those of them whose ref and alt are single bases; the external genes hold SNVs
            only, so for them this is the one scope
Cells: gene x stratum x predictor x scope with at least MIN_N (30) variants. Each gives n,
rho (Spearman of the predictor against functional_pathogenicity, atlas.evaluate's
direction), the reliability and ceiling on the cell's own variants (for SE also the
reliability before clipping), realisation = rho / ceiling (empty where the ceiling is 0),
and for comparison v5's stratum ceiling (core: reliability_v1.tsv's primary route;
external: external_reliability_v1.tsv's main rows) with the realisation it gives.

Bootstrap: B = 2,000 resamples of the cell's variants with replacement (BRCA1's replicate
pairs are resampled as variants), from a seed fixed per cell (from SEED and the cell's key;
written out). Each resample recomputes rho, reliability, ceiling and realisation; their
2.5% and 97.5% quantiles are reported, with the share of resamples whose realisation
exceeds 1 among those where it is defined.

Outputs: results/matched_ceilings_v1.tsv, one row per cell, and
results/matched_ceilings_v1.json: (1) by stratum, for the core three, the external five and
all eight, the matched ceiling's median and range and each predictor's cross-gene median
rho and realisation (snv scope); (2) v5's coding-against-+-1-2 comparison redone with the
matched ceilings, both scopes, with each core gene left out in turn; (3) the cells whose
matched ceiling differs from the stratum ceiling by more than 0.05, and those whose
realisation exceeds 1 or whose interval crosses 1; with the checkpoints and the functional
score's distribution per gene x stratum.

Usage (PYTHONPATH=src):  python -m atlas.matched_ceilings [--workers N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from atlas.evaluate import classify_region
from atlas.matched_realisation import CODING, MISSENSE_ONLY, SPLICE
from atlas.reliability import MIN_N, MODELS, STRATA, _stratum_mask, spearman_brown

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
RAW = REPO / "data" / "raw"

SEED = 20261122
B = 2000
CORE = {"BRCA1": "replicates", "BARD1": "se", "PALB2": "se"}
EXTERNAL_GENES = ("CTCF", "SFPQ", "RAD51D", "XRCC2", "SBDS")
EXTERNAL_MODELS = ["spliceai_ds", "pangolin_score", "cadd", "gpn_msa", "phylop100way",
                   "phastcons100way", "gnomad_af_global", "gnomad_af_popmax",
                   "alphagenome_official"]
INTRONIC = ["splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]
EXTERNAL_STRATA = ["splice_region"] + INTRONIC
SCOPES = ("scored", "snv")
DIFF_LIST = 0.05
# the instruction's checkpoints (a trial outside this repository, not a result of it):
# (all variants with error information, the SNVs SpliceAI, Pangolin and CADD scored)
CHECKPOINTS = {("BARD1", "splice_1_2"): (0.586, 0.454), ("PALB2", "splice_1_2"): (0.794, 0.752)}
CHECK_TOL = 0.005
CHECK_MODELS = ["spliceai_ds", "pangolin_score", "cadd"]


# ==========================================================================
# the variants
# ==========================================================================
def load_core(matrix: Path = RESULTS / "score_matrix_atlas_v2.parquet",
              aux: Path = RESULTS / "assay_aux_v1.parquet") -> pd.DataFrame:
    m = pd.read_parquet(matrix)
    m = m[m["gene"].isin(CORE)]
    a = pd.read_parquet(aux, columns=["variant_id", "gene", "fit_rep1", "fit_rep2", "se",
                                      "se_status"])
    df = m.merge(a, on=["variant_id", "gene"], how="left")
    df["region"] = df["hgvs_c"].map(classify_region)
    df["method"] = df["gene"].map(CORE)
    rep = df["fit_rep1"].notna() & df["fit_rep2"].notna()
    se = df["se"].notna() & (df["se_status"] == "validated")
    df["has_error"] = np.where(df["method"] == "replicates", rep, se)
    df["group"] = "core"
    return _common(df)


def load_external(path: Path = RESULTS / "external_scores_v1.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns={"standard_error": "se"})
    df["region"] = df["hgvs_c"].map(classify_region)
    if not (df["region"] == df["stratum"]).all():
        raise SystemExit("external_scores_v1's stratum differs from classify_region")
    df["fit_rep1"] = df["fit_rep2"] = np.nan
    df["method"] = "se"
    df["has_error"] = df["se"].notna()
    df["group"] = "external"
    return _common(df)


def _common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["snv"] = (df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)
    return df.reset_index(drop=True)


# ==========================================================================
# the estimators, for one set of variants (point) and for many resamples (rows)
# ==========================================================================
def point(x, y, method, se=None, r1=None, r2=None) -> dict:
    """rho, reliability and ceiling as v5 computes them (the same expressions, so the
    same floats where the variants are the same)."""
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    if method == "replicates":
        rep = float(pd.Series(r1).corr(pd.Series(r2), method="spearman"))
        rel = spearman_brown(rep)
        raw = np.nan
    else:
        var = float(pd.Series(y).var())
        err = float((pd.Series(se) ** 2).mean())
        raw = 1 - err / var if var > 0 else np.nan
        rel = float(np.clip(raw, 0.0, 1.0)) if var > 0 else np.nan
        rep = np.nan
    ceiling = float(np.sqrt(max(rel, 0.0))) if np.isfinite(rel) else np.nan
    real = rho / ceiling if np.isfinite(ceiling) and ceiling > 0 else np.nan
    return {"rho": rho, "rep_rho": rep, "reliability": rel, "reliability_unclipped": raw,
            "ceiling": ceiling, "realisation": real}


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman rho of each row of a with the same row of b (average ranks for ties)."""
    ra = rankdata(a, axis=1)
    rb = rankdata(b, axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(axis=1) / den, np.nan)


def resample_stats(idx: np.ndarray, x, y, method, se=None, r1=None, r2=None) -> dict:
    """The estimators on each resample, one per row of idx."""
    rho = spearman_rows(x[idx], y[idx])
    if method == "replicates":
        r = spearman_rows(r1[idx], r2[idx])
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(1 + r != 0, 2 * r / (1 + r), np.nan)
    else:
        yy = y[idx]
        var = yy.var(axis=1, ddof=1)
        err = (se[idx] ** 2).mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.where(var > 0, 1 - err / var, np.nan)
        rel = np.clip(raw, 0.0, 1.0)
    ceiling = np.sqrt(np.clip(rel, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        real = np.where(ceiling > 0, rho / ceiling, np.nan)
    return {"rho": rho, "reliability": rel, "ceiling": ceiling, "realisation": real}


def cell_seed(key: tuple) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}|{'|'.join(map(str, key))}".encode())
                          .digest()[:8], "big")


def bootstrap(arrays: dict, method: str, seed: int, b: int = B, budget: int = 4_000_000) -> dict:
    """Quantiles of the estimators over b resamples, drawn in chunks of rows."""
    n = len(arrays["x"])
    rng = np.random.default_rng(seed)
    chunk = max(1, min(b, budget // max(n, 1)))
    out = {k: [] for k in ("rho", "reliability", "ceiling", "realisation")}
    done = 0
    while done < b:
        k = min(chunk, b - done)
        idx = rng.integers(0, n, size=(k, n))
        res = resample_stats(idx, arrays["x"], arrays["y"], method, arrays.get("se"),
                             arrays.get("r1"), arrays.get("r2"))
        for name in out:
            out[name].append(res[name])
        done += k
    stats = {}
    for name, parts in out.items():
        v = np.concatenate(parts)
        ok = v[np.isfinite(v)]
        stats[f"{name}_lo"] = float(np.quantile(ok, 0.025)) if len(ok) else np.nan
        stats[f"{name}_hi"] = float(np.quantile(ok, 0.975)) if len(ok) else np.nan
        if name == "realisation":
            stats["share_realisation_gt_1"] = float((ok > 1).mean()) if len(ok) else np.nan
            stats["bootstrap_realisation_undefined"] = int(len(v) - len(ok))
    return stats


# ==========================================================================
# the cells
# ==========================================================================
def cell_tasks(df: pd.DataFrame, models: list[str], strata: list[str],
               scopes=SCOPES) -> list[tuple]:
    tasks = []
    for gene, g in df.groupby("gene"):
        method = g["method"].iloc[0]
        for stratum in strata:
            s = g[_stratum_mask(g["region"], stratum) & g["has_error"]
                  & g["functional_pathogenicity"].notna()]
            for model in models:
                if model not in s.columns:
                    continue
                base = s[s[model].notna()]
                for scope in scopes:
                    c = base[base["snv"]] if scope == "snv" else base
                    if len(c) < MIN_N:
                        continue
                    arrays = {"x": c[model].to_numpy(float),
                              "y": c["functional_pathogenicity"].to_numpy(float)}
                    if method == "replicates":
                        arrays |= {"r1": c["fit_rep1"].to_numpy(float),
                                   "r2": c["fit_rep2"].to_numpy(float)}
                    else:
                        arrays["se"] = c["se"].to_numpy(float)
                    key = (g["group"].iloc[0], gene, stratum, model, scope)
                    tasks.append((key, method, arrays))
    return tasks


def run_cell(task: tuple) -> dict:
    key, method, arrays = task
    group, gene, stratum, model, scope = key
    p = point(arrays["x"], arrays["y"], method, arrays.get("se"), arrays.get("r1"),
              arrays.get("r2"))
    seed = cell_seed(key)
    return {"group": group, "gene": gene, "stratum": stratum, "model": model, "scope": scope,
            "n": len(arrays["x"]), "method": method, **p, "seed": seed, "bootstrap_B": B,
            **bootstrap(arrays, method, seed)}


def compute(tasks: list[tuple], workers: int = 1) -> pd.DataFrame:
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run_cell, tasks, chunksize=4))
    else:
        rows = [run_cell(t) for t in tasks]
    return pd.DataFrame(rows)


def stratum_ceilings() -> pd.DataFrame:
    """v5's stratum ceiling per gene: the core genes' primary route, the external genes'
    main rows."""
    from atlas.external_reliability import v5_primary
    core = v5_primary(pd.read_csv(RESULTS / "reliability_v1.tsv", sep="\t"))
    ext = pd.read_csv(RESULTS / "external_reliability_v1.tsv", sep="\t")
    ext = ext[ext["role"] == "main"]
    both = pd.concat([core[["gene", "stratum", "method", "ceiling"]],
                      ext[["gene", "stratum", "method", "ceiling"]]], ignore_index=True)
    return both.rename(columns={"method": "v5_method", "ceiling": "v5_ceiling"})


def add_v5(table: pd.DataFrame, ceil: pd.DataFrame) -> pd.DataFrame:
    t = table.merge(ceil, on=["gene", "stratum"], how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        t["realisation_v5_ceiling"] = np.where(t["v5_ceiling"] > 0,
                                               t["rho"] / t["v5_ceiling"], np.nan)
    t["ceiling_minus_v5"] = t["ceiling"] - t["v5_ceiling"]
    return t


# ==========================================================================
# checkpoints
# ==========================================================================
def check_attenuation(table: pd.DataFrame, df: pd.DataFrame,
                      att: pd.DataFrame | None = None) -> dict:
    """The scored scope's n and rho against attenuation_v1.tsv, cell by cell. A cell
    with fewer variants only because some lack error information is listed and kept;
    any other difference is a failure."""
    att = att if att is not None else pd.read_csv(RESULTS / "attenuation_v1.tsv", sep="\t",
                                                  float_precision="round_trip")
    ours = table[(table["group"] == "core") & (table["scope"] == "scored")] \
        .set_index(["gene", "stratum", "model"])
    same, fewer, other = 0, [], []
    for r in att.itertuples(index=False):
        key = (r.gene, r.stratum, r.model)
        if key not in ours.index:
            other.append({"cell": key, "why": "no matched cell"})
            continue
        o = ours.loc[key]
        if int(o["n"]) == int(r.n) and o["rho"] == r.rho:
            same += 1
            continue
        g = df[(df["gene"] == r.gene) & _stratum_mask(df["region"], r.stratum)
               & df[r.model].notna() & df["functional_pathogenicity"].notna()]
        lacking = int((~g["has_error"]).sum())
        if int(o["n"]) == int(r.n) - lacking and lacking > 0:
            fewer.append({"cell": key, "v5_n": int(r.n), "n": int(o["n"]),
                          "without_error_information": lacking})
        else:
            other.append({"cell": key, "v5_n": int(r.n), "n": int(o["n"]),
                          "v5_rho": float(r.rho), "rho": float(o["rho"])})
    return {"cells_in_attenuation": int(len(att)), "identical": same,
            "fewer_only_for_missing_error_information": fewer, "other_differences": other}


def check_trial(table: pd.DataFrame, df: pd.DataFrame) -> dict:
    """The instruction's checkpoint: BARD1 and PALB2 at +-1-2 bp, the ceiling on every
    variant with error information, and on the SNVs SpliceAI, Pangolin and CADD scored."""
    out = {}
    for (gene, stratum), (want_all, want_snv) in CHECKPOINTS.items():
        g = df[(df["gene"] == gene) & (df["region"] == stratum) & df["has_error"]]
        p_all = point(g["functional_pathogenicity"], g["functional_pathogenicity"], "se",
                      g["se"])
        snv = g[g["snv"] & g[CHECK_MODELS].notna().all(axis=1)]
        p_snv = point(snv["functional_pathogenicity"], snv["functional_pathogenicity"], "se",
                      snv["se"])
        cells = table[(table["gene"] == gene) & (table["stratum"] == stratum)
                      & (table["scope"] == "snv") & table["model"].isin(CHECK_MODELS)]
        pg = cells[cells["model"] == "pangolin_score"].iloc[0]
        out[f"{gene} {stratum}"] = {
            "n_all": int(len(g)), "ceiling_all": p_all["ceiling"], "instruction_all": want_all,
            "n_snv": int(len(snv)), "ceiling_snv": p_snv["ceiling"], "instruction_snv": want_snv,
            "snv_cells_ceiling": cells.set_index("model")["ceiling"].to_dict(),
            "pangolin_rho_snv": float(pg["rho"]), "pangolin_realisation_snv": float(pg["realisation"]),
            "holds": bool(abs(p_all["ceiling"] - want_all) <= CHECK_TOL
                          and abs(p_snv["ceiling"] - want_snv) <= CHECK_TOL
                          and all(abs(c - want_snv) <= CHECK_TOL
                                  for c in cells["ceiling"]))}
    return out


# ==========================================================================
# the summary
# ==========================================================================
def _range(s: pd.Series) -> dict:
    s = s.dropna()
    return ({"n_genes": int(len(s)), "median": float(s.median()), "min": float(s.min()),
             "max": float(s.max())} if len(s) else {"n_genes": 0})


def snv_ceilings(df: pd.DataFrame, strata: list[str]) -> pd.DataFrame:
    """Per gene x stratum, the ceiling on every SNV with a functional score and error
    information (no predictor's coverage applied)."""
    rows = []
    for gene, g in df.groupby("gene"):
        method = g["method"].iloc[0]
        for stratum in strata:
            s = g[_stratum_mask(g["region"], stratum) & g["has_error"] & g["snv"]
                  & g["functional_pathogenicity"].notna()]
            if len(s) < MIN_N:
                continue
            p = point(s["functional_pathogenicity"], s["functional_pathogenicity"], method,
                      s["se"], s["fit_rep1"], s["fit_rep2"])
            rows.append({"group": g["group"].iloc[0], "gene": gene, "stratum": stratum,
                         "n": int(len(s)), "ceiling": p["ceiling"]})
    return pd.DataFrame(rows)


GROUPS = {"core_3": ["core"], "external_5": ["external"], "all_8": ["core", "external"]}


def by_stratum(table: pd.DataFrame, snvc: pd.DataFrame) -> dict:
    """Part 1, on the snv scope: per stratum and group, the SNV ceiling's median and
    range across genes, and per predictor the cross-gene medians."""
    out = {}
    snv = table[table["scope"] == "snv"]
    for stratum in STRATA:
        rec = {}
        for name, groups in GROUPS.items():
            c = snvc[(snvc["stratum"] == stratum) & snvc["group"].isin(groups)]
            t = snv[(snv["stratum"] == stratum) & snv["group"].isin(groups)]
            if not len(c) and not len(t):
                continue
            per_model = {}
            for model, d in t.groupby("model"):
                per_model[model] = {"n_genes": int(d["gene"].nunique()),
                                    "matched_ceiling": _range(d["ceiling"]),
                                    "rho_median": float(d["rho"].median()),
                                    "realisation_median": float(d["realisation"].median())
                                    if d["realisation"].notna().any() else None}
            rec[name] = {"snv_ceiling": _range(c["ceiling"]),
                         "snv_ceiling_by_gene": dict(zip(c["gene"], c["ceiling"])),
                         "per_predictor": per_model}
        if rec:
            out[stratum] = rec
    return out


def coding_vs_splice(table: pd.DataFrame) -> dict:
    """Part 2: v5's matched coding-against-+-1-2 comparison (atlas.matched_realisation),
    with the matched ceilings, for each scope; the full set and each core gene left out."""
    from atlas.matched_realisation import matched_metrics
    core = table[(table["group"] == "core") & ~table["model"].isin(MISSENSE_ONLY)]
    v5 = pd.read_csv(RESULTS / "matched_realisation_loo_v1.tsv", sep="\t")
    out = {"v5": v5.to_dict("records")}
    for scope in SCOPES:
        s = core[core["scope"] == scope]
        rows = []
        for tag, sub in [("full (all 3 genes)", s)] + [(f"drop {g}", s[s["gene"] != g])
                                                       for g in sorted(CORE)]:
            tab = (sub.groupby(["stratum", "model"])[["rho", "realisation"]].median()
                   .reset_index())
            wide = tab.pivot(index="model", columns="stratum", values=["rho", "realisation"])
            if ("realisation", CODING) not in wide.columns or \
                    ("realisation", SPLICE) not in wide.columns:
                continue
            have = wide.dropna(subset=[("realisation", CODING), ("realisation", SPLICE)])
            for rec in matched_metrics(have):
                rows.append({"scope": tag, **rec, "models": sorted(have.index)})
        out[scope] = rows
    return out


def lists(table: pd.DataFrame) -> dict:
    """Part 3: the cells whose matched ceiling differs from the stratum ceiling by more
    than DIFF_LIST, and those whose realisation exceeds 1 or whose interval crosses 1."""
    keep = ["group", "gene", "stratum", "model", "scope", "n"]
    d = table[table["ceiling_minus_v5"].abs() > DIFF_LIST]
    over = table[(table["realisation"] > 1) | (table["realisation_hi"] > 1)]
    return {
        "ceiling_differs_from_stratum_by_more_than_0.05": d[keep + [
            "ceiling", "v5_ceiling", "ceiling_minus_v5"]].to_dict("records"),
        "realisation_above_1_or_interval_crossing_1": over[keep + [
            "rho", "ceiling", "realisation", "realisation_lo", "realisation_hi",
            "share_realisation_gt_1"]].to_dict("records")}


def deposit_class_map() -> dict[str, str]:
    """accession -> the deposit's own class (abnormal / normal / ...), for the eight genes."""
    from atlas.external_reliability import EXTERNAL
    from atlas.external_scores import deposit_classes
    out = {}
    paths = {"urn:mavedb:00000097-0-2": RAW / "mavedb" / "urn_mavedb_00000097-0-2",
             "urn:mavedb:00001250-a-2": RAW / "mavedb" / "urn_mavedb_00001250-a-2",
             "urn:mavedb:00001259-a-2": RAW / "mavedb" / "urn_mavedb_00001259-a-2"}
    for urn, spec in EXTERNAL.items():
        if spec["gene"] in EXTERNAL_GENES:
            paths[urn] = RAW / "mavedb_external" / ("urn_mavedb_" + urn.split(":")[-1])
    for urn, d in paths.items():
        dep = pd.read_csv(d / "scores.csv", low_memory=False)
        meta = json.loads((d / "metadata.json").read_text())
        _, cls = deposit_classes(meta, dep)
        out.update(dict(zip(dep["accession"], cls)))
    return out


def distributions(df: pd.DataFrame, classes: dict[str, str]) -> list[dict]:
    """Per gene x stratum, functional_pathogenicity's 5/25/50/75/95th percentiles and the
    deposit's class counts, on every variant with error information and on the SNVs among
    them (no predictor's coverage applied)."""
    rows = []
    key = np.where(df["group"] == "core", df["variant_id"], df.get("variant_urn"))
    df = df.assign(cls=pd.Series(key).map(classes).fillna("unclassified").to_numpy())
    for gene, g in df.groupby("gene"):
        strata = STRATA if g["group"].iloc[0] == "core" else EXTERNAL_STRATA
        for stratum in strata:
            base = g[_stratum_mask(g["region"], stratum) & g["has_error"]
                     & g["functional_pathogenicity"].notna()]
            for scope, s in (("with_error", base), ("snv_with_error", base[base["snv"]])):
                if not len(s):
                    continue
                q = s["functional_pathogenicity"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                rows.append({"gene": gene, "stratum": stratum, "scope": scope, "n": int(len(s)),
                             "functional_pathogenicity_quantiles":
                                 dict(zip(["q05", "q25", "q50", "q75", "q95"], map(float, q))),
                             "deposit_classes": {k: int(v) for k, v in
                                                 s["cls"].value_counts().items()}})
    return rows


def jsonable(obj):
    """NaN as null, numpy scalars as Python ones, recursively."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ==========================================================================
# command line
# ==========================================================================
COLUMNS = ["group", "gene", "stratum", "model", "scope", "n", "method", "rho", "rep_rho",
           "reliability", "reliability_unclipped", "ceiling", "realisation", "v5_method",
           "v5_ceiling", "ceiling_minus_v5", "realisation_v5_ceiling", "seed", "bootstrap_B",
           "rho_lo", "rho_hi", "reliability_lo", "reliability_hi", "ceiling_lo", "ceiling_hi",
           "realisation_lo", "realisation_hi", "share_realisation_gt_1",
           "bootstrap_realisation_undefined"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)
    core, ext = load_core(), load_external()
    tasks = (cell_tasks(core, MODELS, STRATA)
             + cell_tasks(ext, EXTERNAL_MODELS, EXTERNAL_STRATA, scopes=("snv",)))
    print(f"[matched] {len(tasks)} cells, {B} resamples each, {args.workers} workers",
          flush=True)
    table = add_v5(compute(tasks, args.workers), stratum_ceilings())[COLUMNS]
    df = pd.concat([core, ext], ignore_index=True)
    att = check_attenuation(table, core)
    trial = check_trial(table, core)
    failures = []
    if att["other_differences"]:
        failures.append(f"{len(att['other_differences'])} scored cells differ from "
                        "attenuation_v1.tsv beyond missing error information")
    failures += [f"checkpoint {k}" for k, v in trial.items() if not v["holds"]]
    summary = {"definition": {"seed": SEED, "bootstrap_B": B, "min_n": MIN_N,
                              "scopes": list(SCOPES), "core": CORE,
                              "external_genes": list(EXTERNAL_GENES),
                              "cross_gene_scope": "snv"},
               "checkpoints": {"attenuation_v1": att, "instruction_trial": trial},
               "1_by_stratum": by_stratum(table, snv_ceilings(df, STRATA)),
               "2_coding_vs_splice_1_2": coding_vs_splice(table),
               "3_lists": lists(table),
               "distributions": distributions(df, deposit_class_map()),
               "failures": failures}
    out = Path(args.out)
    if failures:
        (out / "_scratch").mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "_scratch" / "matched_ceilings_failed.tsv", sep="\t", index=False)
        print("checkpoints fail:", failures, file=sys.stderr)
        return 1
    table.to_csv(out / "matched_ceilings_v1.tsv", sep="\t", index=False)
    (out / "matched_ceilings_v1.json").write_text(json.dumps(jsonable(summary), indent=1) + "\n")
    print(f"[matched] wrote {len(table)} cells; checkpoints hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
