"""The rank-scale ceilings of the tables' cells (A4b.4): one set of variants under the two
error models of atlas.error_models, the neutral model first.

A cell's set is its variants with a functional score and the error information: in a layer
ceiling every such variant of the stratum, in a matched ceiling those the predictor scored.
Under each model atlas.rank_ceiling.estimate_cell gives the point, the bootstrap interval
(B = 1,000; 200 in the third generation), the floor (the 95th percentile of 500 data sets
without effect; 100 in the third generation) and estimable (the interval's lower bound
above the floor); with a matched cell's predictors, also the realisation's interval on the
same resamples. The seed comes from the cell's key (rank, group, gene, stratum, population,
model, the set's sha256), so one set gets the same numbers in every table, whichever
process computes it. Each set also carries the third generation's numbers, computed from
its first 200 resamples and first 100 null data sets (first_columns, first_realisation).

A store (compute(store=DIR)) keeps each set's result as it is finished: DIR/<name>.pkl, and
DIR/<name>.npz with each model's resampled ceilings (<model>_boot), null estimates
(<model>_null) and, per predictor, the resamples' rho (<model>_rho_<predictor>). A set
already in the store is read, not computed: its result depends on its seed alone, so a
run resumed gives the bits of a run in one go. The pickle records what it was computed
from (fingerprint); a set whose inputs differ is an error.

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

import hashlib
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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


def run(t: tuple, store: Path | None = None) -> dict:
    """One set under both models; with a store, its result and arrays are written there."""
    (group, gene, stratum, pop), ids, y, inputs, preds, lof = t
    sha = rc.set_sha(ids)
    out = {"group": group, "gene": gene, "stratum": stratum, "population": pop,
           "set_sha": sha[:16], "n": len(ids), "lof_share": lof}
    arrays = {}
    for model in MODELS:
        kw = dict(inputs[model])
        s = kw.pop("s")
        r = rc.estimate_cell(ids, y, s, rc.cell_key(group, gene, stratum, pop, model, sha),
                             predictors=preds or None, **kw)
        boot, null, rho = r.pop("boot"), r.pop("null"), r.pop("rho_boot", {})
        r["first"] = rc.first_resamples(boot, null, rho)
        out[model] = r
        arrays |= {f"{model}_boot": boot, f"{model}_null": null}
        arrays |= {f"{model}_rho_{k}": v for k, v in rho.items()}
    if store is not None:
        save(store, t, out, arrays)
    return out


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------
def store_name(t: tuple) -> str:
    (group, gene, stratum, pop), ids = t[0], t[1]
    return f"{group}_{gene}_{stratum}_{pop}_{rc.set_sha(ids)[:16]}"


def fingerprint(t: tuple) -> str:
    """sha256 of what a set's result is computed from: the estimator's settings, the
    identifiers, y, each model's inputs and the predictors' scores."""
    (group, gene, stratum, pop), ids, y, inputs, preds, lof = t
    h = hashlib.sha256()
    settings = (rc.B, rc.NULLS, rc.ITER_POINT, rc.ITER_RESAMPLE, rc.REPEAT_POINT,
                rc.REPEAT_RESAMPLE, rc.GRID_QUANTILES, rc.GRID_EVEN, rc.GRID_PAD, rc.KDE_POINTS,
                group, gene, stratum, pop, MODELS)
    h.update(repr(settings).encode())
    h.update("\n".join(map(str, ids)).encode())
    h.update(np.ascontiguousarray(y, float).tobytes())
    for m in MODELS:
        kw = inputs[m]
        h.update(m.encode())
        h.update(np.ascontiguousarray(kw["s"], float).tobytes())
        if kw.get("em") is not None and kw["em"].z is not None:
            h.update(np.ascontiguousarray(kw["em"].z, float).tobytes())
        if kw.get("residuals") is not None:
            h.update(np.ascontiguousarray(kw["residuals"], float).tobytes())
        h.update(repr((kw.get("em") is None, bool(kw.get("pooled")))).encode())
    for k in sorted(preds):
        h.update(k.encode())
        h.update(np.ascontiguousarray(preds[k], float).tobytes())
    return h.hexdigest()


def _replace(path: Path, write) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        write(fh)
    os.replace(tmp, path)


def save(store: Path, t: tuple, out: dict, arrays: dict) -> None:
    store = Path(store)
    name = store_name(t)
    _replace(store / f"{name}.npz", lambda fh: np.savez(fh, **arrays))
    _replace(store / f"{name}.pkl",
             lambda fh: pickle.dump({"fingerprint": fingerprint(t), "result": out}, fh))


def load(store: Path, t: tuple) -> dict | None:
    """A set's stored result, or None; an error if it was computed from other inputs."""
    p = Path(store) / f"{store_name(t)}.pkl"
    if not p.exists() or not (p.with_suffix(".npz")).exists():
        return None
    rec = pickle.loads(p.read_bytes())
    if rec["fingerprint"] != fingerprint(t):
        raise SystemExit(f"{p}: stored from other inputs; remove it to compute the set again")
    return rec["result"]


def compute(tasks: list[tuple], workers: int = 1, store: Path | None = None,
            stats: dict | None = None) -> list[dict]:
    """The sets' results in the order of tasks (the largest sets are started first). With
    a store, the sets already there are read and each set computed is written as it is
    finished; stats, when given, gets the number of sets and of those read from the store."""
    order = sorted(range(len(tasks)), key=lambda i: -len(tasks[i][1]))
    done: dict[int, dict] = {}
    if store is not None:
        Path(store).mkdir(parents=True, exist_ok=True)
        for i in order:
            r = load(store, tasks[i])
            if r is not None:
                done[i] = r
    todo = [i for i in order if i not in done]
    if stats is not None:
        stats |= {"sets": len(tasks), "sets_read_from_the_store": len(done)}
    if store is not None:
        print(f"[rank] {len(tasks)} sets: {len(done)} in the store, {len(todo)} to compute "
              f"({workers} workers)", flush=True)
    t0 = time.time()

    def note(i: int, k: int) -> None:
        if store is not None:
            r = done[i]
            print(f"[rank] {k}/{len(todo)} {r['group']} {r['gene']} {r['stratum']} "
                  f"{r['population']} n={r['n']} {time.time() - t0:.0f} s", flush=True)

    if workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, tasks[i], store): i for i in todo}
            for k, f in enumerate(as_completed(futures), 1):
                i = futures[f]
                done[i] = f.result()
                note(i, k)
    else:
        for k, i in enumerate(todo, 1):
            done[i] = run(tasks[i], store)
            note(i, k)
    if store is not None:
        print(f"[rank] done in {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    return [done[i] for i in range(len(tasks))]


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


def first_columns(r: dict) -> dict:
    """A set's interval, floor and estimable from its first 200 resamples and 100 null
    data sets, under the names of the table's columns: the third generation's."""
    row = {}
    for m in MODELS:
        x = r[m]["first"]
        row |= {f"ceiling_{m}_lo": x["ceiling_lo"], f"ceiling_{m}_hi": x["ceiling_hi"],
                f"floor_{m}": x["floor"], f"estimable_{m}": x["estimable"]}
    return row


def first_realisation(r: dict, predictor: str) -> dict:
    """A matched cell's realisation interval from the first 200 resamples."""
    row = {}
    for m in MODELS:
        q = r[m]["first"]["realisation"][predictor]
        row |= dict(zip(realisation_columns(m)[1:],
                        (q["realisation_lo"], q["realisation_hi"], q["share_realisation_gt_1"],
                         q["bootstrap_realisation_undefined"])))
    return row


FIRST = [c for m in MODELS for c in (f"ceiling_{m}_lo", f"ceiling_{m}_hi", f"floor_{m}",
                                     f"estimable_{m}")]
FIRST_REALISATION = [c for m in MODELS for c in realisation_columns(m)[1:]]
# the columns that may differ from the third generation's (B and the nulls); every other
# column of a table is the third generation's, bit for bit
MAY_CHANGE = FIRST + FIRST_REALISATION


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


# --------------------------------------------------------------------------
# the fourth generation against the third (A4c.1)
# --------------------------------------------------------------------------
def check_v3(here: pd.DataFrame, path: Path, keys: list[str],
             first: pd.DataFrame | None = None, realisation: bool = False) -> dict:
    """Against the third generation's file: (1) every column of the file but MAY_CHANGE
    bit for bit (the point estimates and the rest); (2) with first (the same rows as here),
    the interval, floor and estimable from the first 200 resamples and 100 null data sets
    bit for bit the file's (and, with realisation, the realisation's interval)."""
    from atlas.reliability import _same, same_rows
    f = pd.read_csv(path, sep="\t", float_precision="round_trip")
    a = here.sort_values(keys, kind="stable").reset_index(drop=True)
    b = f.sort_values(keys, kind="stable").reset_index(drop=True)
    rows_equal = len(a) == len(b) and _same(a[keys].astype(str).agg("|".join, axis=1),
                                             b[keys].astype(str).agg("|".join, axis=1))
    fixed = [c for c in b.columns if c not in MAY_CHANGE]
    point = {"columns": len(fixed), "rows": int(len(b)), "rows_here": int(len(a)),
             "columns_missing_here": [c for c in b.columns if c not in a.columns],
             "columns_differing": [c for c in fixed if c in a.columns and not _same(a[c], b[c])],
             "may_change": MAY_CHANGE if realisation else FIRST}
    point["holds"] = bool(rows_equal and not point["columns_missing_here"]
                          and not point["columns_differing"])
    out = {"file": Path(path).name, "point_estimates_and_the_rest": point}
    if first is not None:
        fa = first.sort_values(keys, kind="stable").reset_index(drop=True)
        cols = FIRST + (FIRST_REALISATION if realisation else [])
        differ = [c for c in cols if not _same(fa[c], b[c])]
        ranked = b["ceiling_neutral"].notna()
        out["first_200_resamples_and_100_nulls"] = {
            "columns": cols, "rows_with_a_rank_ceiling": int(ranked.sum()),
            "columns_differing": differ,
            "rows_differing": int(sum((~same_rows(fa[c], b[c])).sum() for c in differ)),
            "holds": bool(len(fa) == len(b) and not differ)}
    return out


def _bool(v: pd.Series) -> pd.Series:
    return v.map(lambda x: None if pd.isna(x) else str(x) == "True")


def estimable_changes(t3: pd.DataFrame, t4: pd.DataFrame, keys: list[str], model: str,
                      extra: tuple = ("n", "lof_share", "neutral_in_range")) -> list[dict]:
    """The rows whose estimable under the model differs between the generations, with
    both generations' lower bound, upper bound and floor."""
    m = t3.merge(t4, on=keys, suffixes=("_v3", "_v4"), validate="one_to_one")
    a, b = _bool(m[f"estimable_{model}_v3"]), _bool(m[f"estimable_{model}_v4"])
    hit = a.notna() & b.notna() & (a != b)
    out = []
    for i in m.index[hit]:
        r = m.loc[i]
        rec = {k: r[k] for k in keys} | {k: r[f"{k}_v4"] for k in extra if f"{k}_v4" in m}
        for g in ("v3", "v4"):
            rec |= {f"ceiling_lo_{g}": r[f"ceiling_{model}_lo_{g}"],
                    f"ceiling_hi_{g}": r[f"ceiling_{model}_hi_{g}"],
                    f"floor_{g}": r[f"floor_{model}_{g}"], f"estimable_{g}": bool(r[f"estimable_{model}_{g}"])}
        rec["ceiling"] = r[f"ceiling_{model}_v4"]
        out.append(rec)
    return out


def bound_changes(t3: pd.DataFrame, t4: pd.DataFrame, keys: list[str], model: str) -> dict:
    """v4 - v3 of the interval's bounds and the floor over the rows with a rank-scale
    ceiling: n, the median change, the median and the largest absolute change (and where)."""
    m = t3.merge(t4, on=keys, suffixes=("_v3", "_v4"), validate="one_to_one")
    m = m[m[f"ceiling_{model}_v4"].notna()]
    out = {}
    for name, c in (("ceiling_lo", f"ceiling_{model}_lo"), ("ceiling_hi", f"ceiling_{model}_hi"),
                    ("floor", f"floor_{model}")):
        d = (m[f"{c}_v4"] - m[f"{c}_v3"]).astype(float)
        if not len(d):
            continue
        j = d.abs().idxmax()
        out[name] = {"n": int(len(d)), "median": float(d.median()),
                     "median_abs": float(d.abs().median()), "max_abs": float(d.abs().max()),
                     "largest": {k: m.loc[j, k] for k in keys} | {"v3": float(m.loc[j, f"{c}_v3"]),
                                                                  "v4": float(m.loc[j, f"{c}_v4"])}}
    return out


def entering_side_by_side(s3: list[dict], s4: list[dict]) -> list[dict]:
    """Two generations' stratum summaries (stratum_summary): per stratum, the genes that
    enter, v3 beside v4."""
    a = {r["stratum"]: r for r in s3}
    out = []
    for r in s4:
        o = a.get(r["stratum"], {})
        out.append({"stratum": r["stratum"], "n_genes_entering_v3": o.get("n_genes_entering"),
                    "n_genes_entering_v4": r["n_genes_entering"],
                    "genes_entering_v3": o.get("genes_entering"),
                    "genes_entering_v4": r["genes_entering"]})
    return out


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
