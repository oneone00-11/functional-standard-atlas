"""The rank-scale ceilings of the tables' cells (A4b.4): one set of variants under the two
error models of atlas.error_models, the neutral model first.

A cell's set is its variants with a functional score and the error information: in a layer
ceiling every such variant of the stratum, in a matched ceiling those the predictor scored.
Under each model atlas.rank_ceiling.estimate_cell gives the point, the bootstrap interval
(B = 200), the floor (the 95th percentile of 100 data sets without effect) and estimable
(the interval's lower bound above the floor); with a matched cell's predictors, also the
realisation's interval on the same resamples. The seed comes from the cell's key (rank,
group, gene, stratum, population, model, the set's sha256), so one set gets the same
numbers in every table, whichever process computes it.

The models' inputs for a set (atlas.error_models):
  neutral  the SE genes: s = sigma(SE) and the gene's neutral residuals z = r / sigma(SE),
           fixed; BRCA1: s = 1 and its neutral residuals;
  own      the SE genes: the SE pooled in the set, sqrt(mean SE^2) (estimate_cell(pooled=
           True): the resampled variants' in each resample), Gaussian; BRCA1: s = 1 and the
           set's own half-differences d = (rep1 - rep2) / 2, rebuilt in each resample.

The neutral model's range (atlas.control_classes): lof_share is the share of the set's
variants at the loss-of-function level (a score above (t_N + t_L) / 2 of
results/control_classes_v1.tsv); neutral_in_range is lof_share <= 0.5. Outside the range
the neutral model's numbers are computed and written but enter no summary, as a cell that
is not estimable; the own model is not affected.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from atlas import error_models as em
from atlas import rank_ceiling as rc

MODELS = ("neutral", "own")
LOF_SHARE_MAX = 0.5
GENES = em.GENES                     # BRCA1 and the eight SE genes


def model_columns(model: str) -> list[str]:
    return [f"rank_reliability_{model}", f"ceiling_{model}", f"ceiling_{model}_lo",
            f"ceiling_{model}_hi", f"floor_{model}", f"estimable_{model}", f"seed_{model}",
            f"null_seed_{model}"]


def realisation_columns(model: str) -> list[str]:
    return [f"realisation_{model}", f"realisation_{model}_lo", f"realisation_{model}_hi",
            f"share_realisation_gt_1_{model}", f"bootstrap_realisation_undefined_{model}"]


COLUMNS = (["set_sha", "lof_share", "neutral_in_range"]
           + [c for m in MODELS for c in model_columns(m)] + ["pooled_scale_own"])


# ==========================================================================
# the inputs
# ==========================================================================
def model_inputs(gene: str, model: str, n: int, se=None, d=None) -> dict:
    """The keyword arguments of atlas.rank_ceiling.estimate_cell (s included) for a set of
    n variants of the gene, given their SEs (an SE gene) or half-differences (BRCA1)."""
    if gene not in GENES:
        raise ValueError(f"{gene} has no error model")
    if model == "neutral":
        s = np.ones(n) if gene == "BRCA1" else em.error_scale(gene)(np.asarray(se, float))
        return {"s": s, "em": em.neutral_model(gene)}
    if model == "own":
        if gene == "BRCA1":
            return {"s": np.ones(n), "residuals": np.asarray(d, float)}
        return {"s": np.asarray(se, float), "em": rc.ErrorModel(), "pooled": True}
    raise ValueError(f"unknown model {model!r}")


def lof_share(gene: str, y) -> float:
    """The share of variants above the gene's loss-of-function threshold."""
    from atlas.control_classes import levels
    return float((np.asarray(y, float) > levels().loc[gene, "lof_threshold"]).mean())


def task(group: str, gene: str, stratum: str, pop: str, ids, y, se=None, d=None,
         predictors: dict | None = None) -> tuple:
    """One set's work: both models' inputs built here, in the calling process."""
    ids = [str(i) for i in ids]
    y = np.asarray(y, float)
    inputs = {m: model_inputs(gene, m, len(ids), se, d) for m in MODELS}
    return ((group, gene, stratum, pop), ids, y, inputs, predictors or {}, lof_share(gene, y))


def run(t: tuple) -> dict:
    (group, gene, stratum, pop), ids, y, inputs, preds, lof = t
    sha = rc.set_sha(ids)
    out = {"group": group, "gene": gene, "stratum": stratum, "population": pop,
           "set_sha": sha[:16], "n": len(ids), "lof_share": lof}
    for model in MODELS:
        kw = dict(inputs[model])
        s = kw.pop("s")
        r = rc.estimate_cell(ids, y, s, rc.cell_key(group, gene, stratum, pop, model, sha),
                             predictors=preds or None, **kw)
        r.pop("boot")
        out[model] = r
    return out


def compute(tasks: list[tuple], workers: int = 1) -> list[dict]:
    """The sets' results in the order of tasks (the largest sets are started first)."""
    order = sorted(range(len(tasks)), key=lambda i: -len(tasks[i][1]))
    todo = [tasks[i] for i in order]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(run, todo))
    else:
        done = [run(t) for t in todo]
    out: list[dict] = [{}] * len(tasks)
    for i, r in zip(order, done):
        out[i] = r
    return out


def columns(r: dict) -> dict:
    """A set's flat columns (COLUMNS)."""
    row = {"set_sha": r["set_sha"], "lof_share": r["lof_share"],
           "neutral_in_range": bool(r["lof_share"] <= LOF_SHARE_MAX)}
    for m in MODELS:
        x = r[m]
        row |= dict(zip(model_columns(m), (x["rank_reliability"], x["ceiling"],
                                           x["ceiling_lo"], x["ceiling_hi"], x["floor"],
                                           x["estimable"], x["seed"], x["null_seed"])))
    row["pooled_scale_own"] = r["own"].get("pooled_scale", np.nan)
    return row


def realisation(r: dict, predictor: str, rho: float) -> dict:
    """A matched cell's realisation under each model: rho / ceiling, and the interval on
    the set's resamples."""
    row = {}
    for m in MODELS:
        x = r[m]
        c = x["ceiling"]
        q = x["realisation"][predictor]
        row |= dict(zip(realisation_columns(m),
                        (rho / c if c > 0 else np.nan, q["realisation_lo"], q["realisation_hi"],
                         q["share_realisation_gt_1"], q["bootstrap_realisation_undefined"])))
    return row


# ==========================================================================
# summaries
# ==========================================================================
def usable(t: pd.DataFrame, model: str) -> pd.Series:
    """The rows whose ceiling under the model enters a summary: estimable, and for the
    neutral model in its range."""
    ok = t[f"estimable_{model}"].astype("boolean").fillna(False).astype(bool)
    if model == "neutral":
        ok &= t["neutral_in_range"].astype("boolean").fillna(False).astype(bool)
    return ok


def _range(v: pd.Series) -> dict:
    v = pd.to_numeric(v, errors="coerce").dropna()
    return {"n": int(len(v)), "median": float(v.median()) if len(v) else None,
            "min": float(v.min()) if len(v) else None, "max": float(v.max()) if len(v) else None}


def stratum_summary(t: pd.DataFrame, model: str, strata) -> list[dict]:
    """Per stratum, over the rows with a rank-scale ceiling: the genes, those whose ceiling
    enters (estimable, in range), those not estimable and those out of range, the
    ceiling's median and range over the genes that enter, and the cells in and out of the
    neutral model's range."""
    out = []
    t = t[t[f"ceiling_{model}"].notna()]
    for stratum in strata:
        s = t[t["stratum"] == stratum]
        if s.empty:
            continue
        ok = usable(s, model)
        est = s[f"estimable_{model}"].astype(bool)
        rng = s["neutral_in_range"].astype(bool)
        out.append({"stratum": stratum, "genes": sorted(s["gene"]),
                    "genes_entering": sorted(s.loc[ok, "gene"]),
                    "n_genes_entering": int(ok.sum()),
                    "genes_not_estimable": sorted(s.loc[~est, "gene"]),
                    "genes_out_of_neutral_range": sorted(s.loc[~rng, "gene"]),
                    "cells_in_neutral_range": int(rng.sum()),
                    "cells_out_of_neutral_range": int((~rng).sum()),
                    "ceiling": _range(s.loc[ok, f"ceiling_{model}"])})
    return out


def range_counts(t: pd.DataFrame, strata) -> list[dict]:
    """Per stratum: the cells in and out of the neutral model's range."""
    t = t[t["lof_share"].notna()]
    return [{"stratum": st, "in_range": int(t.loc[t["stratum"] == st, "neutral_in_range"].astype(bool).sum()),
             "out_of_range": int((~t.loc[t["stratum"] == st, "neutral_in_range"].astype(bool)).sum())}
            for st in strata if (t["stratum"] == st).any()]


# the first supplement's trial (SNVs, point estimates, five bins): own / neutral
TRIAL = {("BRCA1", "splice_3_10"): (0.803, 0.764), ("BRCA1", "splice_11_50"): (0.539, 0.347),
         ("BARD1", "splice_3_10"): (0.591, 0.674), ("BARD1", "splice_11_50"): (0.245, 0.308),
         ("PALB2", "splice_3_10"): (0.574, 0.762), ("PALB2", "splice_11_50"): (0.320, 0.486),
         ("BAP1", "splice_3_10"): (0.835, 0.693), ("BAP1", "splice_11_50"): (0.713, 0.465),
         ("CTCF", "splice_3_10"): (0.577, 0.467), ("CTCF", "splice_11_50"): (0.330, 0.102),
         ("SFPQ", "splice_3_10"): (0.535, 0.657), ("SFPQ", "splice_11_50"): (0.459, 0.524),
         ("RAD51D", "splice_3_10"): (0.708, 0.680), ("RAD51D", "splice_11_50"): (0.589, 0.298),
         ("XRCC2", "splice_3_10"): (0.457, 0.670), ("XRCC2", "splice_11_50"): (0.245, 0.375),
         ("SBDS", "splice_3_10"): (0.462, 0.429), ("SBDS", "splice_11_50"): (0.277, 0.146)}
TRIAL_NEAR_FLOOR = {("CTCF", "splice_11_50", "neutral"), ("XRCC2", "splice_11_50", "own"),
                    ("SBDS", "splice_11_50", "neutral")}
TRIAL_TOL = 0.05


def trial_comparison(layer: pd.DataFrame, model: str) -> list[dict]:
    """The layer ceilings (SNVs) against the trial, under one model."""
    snv = layer[layer["population"] == "snv"].set_index(["gene", "stratum"])
    out = []
    for (gene, stratum), (own, neu) in TRIAL.items():
        if (gene, stratum) not in snv.index:
            continue
        r = snv.loc[(gene, stratum)]
        want = neu if model == "neutral" else own
        got = float(r[f"ceiling_{model}"])
        near = (gene, stratum, model) in TRIAL_NEAR_FLOOR
        out.append({"gene": gene, "stratum": stratum, "n": int(r["n"]), "trial": want,
                    "here": got, "difference": got - want, "near_the_floor": near,
                    "within_tolerance": None if near else bool(abs(got - want) <= TRIAL_TOL)})
    return out
