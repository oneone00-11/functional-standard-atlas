"""Does the remeasurement simulation predict the agreement of independent replicates?
(A4b.3)

atlas.rank_ceiling estimates the rank-scale reliability of a score from the score and its
error alone. Where a deposit also publishes the replicates the score is made of, the
estimate can be put to the test: from the combined score and its error, predict the mean
Spearman rho between two single replicates, and compare it with what the replicates
show.

Deposits: the nine of results/mavedb_estimator_comparison_v2.tsv that publish both
replicates and an error column (CYP2C19 VAMP-seq, non-synonymous and synonymous; TSC2,
RapGAP and Tuberin domains; G6PD VAMP-seq; T. brucei RBP6, four scores), their score
tables fetched with atlas.mavedb_estimator_comparison.fetch's request (the reply kept in
data/raw/rank_ceiling_validation/, untracked, with the date); and BRCA1's SGE deposit
from the score matrix and atlas.assay_aux, by stratum (its 3,893 variants are all SNVs).

Per deposit (or BRCA1 cell):
  1 checks: the score against the replicates' mean (largest difference), k, the error
    column (the estimator comparison's);
  2 the error column's scale: R = mean(the replicates' sample variance / k) /
    mean(error^2). Near 1 on the log scale the column is the SE of the mean; near 1/k,
    the SD of one replicate (the mean's SE is the column / sqrt(k)). Off from both by
    more than a factor 2: ``scale_unclear``, taken as the mean's SE and flagged. s_i is
    the mean's SE of variant i, so set;
  3 observed: the mean pairwise Spearman rho of the replicates (r_obs), and Pearson's;
  4 predicted: the mean pairwise Spearman rho of k simulated single replicates (error
    scale s sqrt(k)), from the true-value distribution estimated on the combined score
    and s (atlas.rank_ceiling, point settings). Four predictions:
       replicate    the acceptance. One pooled scale s = sqrt(mean(s_i^2)); the shape the
                    replicates' deviations from their mean, z_ij = (r_ij - mean_i) /
                    (s sqrt(k - 1)), all variants together. BRCA1: within the cell
                    d = (rep1 - rep2) / 2 with s = 1, the own model of the ceilings;
       per_variant  a diagnostic: as replicate with s_i variant by variant (the scale
                    no ceiling table uses). BRCA1 has no per-variant error: none;
       own          Gaussian with the pooled s. BRCA1 deposits no error column: none;
       neutral      where synonymous variants are recognised (hgvs_pro p.=, p.(=) or a
                    name ending in =; 00001199-a-1 takes those of its experiment's
                    synonymous deposit 00001199-a-2): the error scale sigma(s) calibrated
                    on them as atlas.error_models calibrates a gene's
                    (atlas.rank_ceiling.ErrorScale on their s_i and r = score - their
                    median), s'_i = sigma(s_i), the shape z = r / sigma(s). BRCA1: its
                    neutral SNVs, z = r, s = 1. None recognised: none. Well below r_obs,
                    it says the synonymous variants carry real effects, or the
                    replicates share variation;
  5 the combined score's rank-scale ceiling under each of the four, beside the
    replicates' bounds [sqrt(r_obs), sqrt(SB_k(r_obs))] and the variance formula
    sqrt(1 - mean(s_i^2) / var(score)) (BRCA1: mean(d^2), d^2 being the square of the SE
    atlas.assay_aux derives from its replicates).

Acceptance: |replicate prediction - r_obs| <= 0.10 for every deposit with n >= 300 whose
error scale is clear, and every BRCA1 cell with n >= 100. The other three are reported.

Outputs: results/rank_ceiling_validation_v1.tsv and .json.

Usage (PYTHONPATH=src):  python -m atlas.rank_ceiling_validation [fetch] [--workers N]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from atlas import paths, population
from atlas import rank_ceiling as rc
from atlas.reliability import STRATA, _stratum_mask, spearman_brown

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "raw" / "rank_ceiling_validation"
FETCH_LOG = CACHE / "fetch_log.json"
BRCA1 = "urn:mavedb:00000097-0-2"
DEPOSITS = ["urn:mavedb:00001199-a-1", "urn:mavedb:00001199-a-2", "urn:mavedb:00001267-b-1",
            "urn:mavedb:00001267-a-1", "urn:mavedb:00001266-b-1", "urn:mavedb:00001212-a-1",
            "urn:mavedb:00001212-a-2", "urn:mavedb:00001212-a-3", "urn:mavedb:00001212-a-4"]
EXPERIMENT = {"urn:mavedb:00001199-a-1": "CYP2C19 VAMP-seq, non-synonymous",
              "urn:mavedb:00001199-a-2": "CYP2C19 VAMP-seq, synonymous",
              "urn:mavedb:00001267-b-1": "TSC2 RapGAP domain",
              "urn:mavedb:00001267-a-1": "TSC2 Tuberin domain",
              "urn:mavedb:00001266-b-1": "G6PD VAMP-seq",
              "urn:mavedb:00001212-a-1": "T. brucei RBP6 DMS, score 1",
              "urn:mavedb:00001212-a-2": "T. brucei RBP6 DMS, score 2",
              "urn:mavedb:00001212-a-3": "T. brucei RBP6 DMS, score 3",
              "urn:mavedb:00001212-a-4": "T. brucei RBP6 DMS, score 4",
              BRCA1: "BRCA1 SGE"}
SYNONYMOUS_FROM = {"urn:mavedb:00001199-a-1": "urn:mavedb:00001199-a-2"}
MODELS = ("replicate", "per_variant", "own", "neutral")
SCALE_FACTOR = 2.0
TOL = 0.10
MIN_N_DEPOSIT = 300
MIN_N_BRCA1 = 100
MIN_N = 30
RETRY_WAIT = (10, 30, 60, 120)

COLUMNS = (["urn", "experiment", "population", "stratum", "n", "k",
            "score_vs_mean_max_abs_diff", "error_column", "R", "error_scale", "scale_unclear",
            "pooled_s", "r_obs", "pearson_obs"]
           + [f"pred_{m}" for m in MODELS] + [f"pred_minus_obs_{m}" for m in MODELS]
           + ["n_neutral", "neutral_source", "neutral_largest_block"]
           + [f"ceiling_{m}_model" for m in MODELS]
           + ["bound_sqrt_r", "bound_sqrt_sb_k", "variance_formula", "acceptance_applies",
              "passes", "note"])


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ==========================================================================
# the score tables
# ==========================================================================
def cache_path(urn: str) -> Path:
    return CACHE / ("urn_mavedb_" + urn.split(":")[-1]) / "scores.csv"


def fetch(urns=DEPOSITS) -> dict:
    """atlas.mavedb_estimator_comparison.fetch's request, with the reply kept."""
    from atlas.mavedb_survey import API, TIMEOUT
    log = json.loads(FETCH_LOG.read_text()) if FETCH_LOG.exists() else {"score_tables": {}}
    FETCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    for urn in urns:
        p = cache_path(urn)
        if p.exists() and urn in log["score_tables"]:
            continue
        url = f"{API}/score-sets/{urn}/scores"
        attempts = []
        for wait in RETRY_WAIT + (None,):
            try:
                r = requests.get(url, timeout=TIMEOUT)
                r.raise_for_status()
                break
            except requests.RequestException as exc:
                attempts.append(f"{utc()} {type(exc).__name__}: {str(exc)[:120]}")
                if wait is None:
                    raise
                time.sleep(wait)
        pd.read_csv(io.StringIO(r.text), low_memory=False)          # a table, as fetch reads it
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
        log["score_tables"][urn] = {"url": url, "fetched_utc": utc(), "bytes": len(r.content),
                                    "sha256": hashlib.sha256(r.content).hexdigest(),
                                    "failed_attempts": attempts}
        FETCH_LOG.write_text(json.dumps(log, indent=1) + "\n")
        print(f"[fetch] {urn}: {len(r.content)} bytes", flush=True)
    return log


def load(urn: str) -> pd.DataFrame:
    """The cached table, read as atlas.mavedb_estimator_comparison.fetch reads the reply."""
    p = cache_path(urn)
    if not p.exists():
        raise FileNotFoundError(f"{urn}: run `python -m atlas.rank_ceiling_validation fetch`")
    return pd.read_csv(io.StringIO(p.read_text()), low_memory=False)


def specification() -> dict[str, dict]:
    """Replicates from the survey (reconciling_replicates), the error column from the
    estimator comparison."""
    s = pd.read_csv(paths.path("mavedb_survey"), sep="\t").fillna("").set_index("urn")
    e = pd.read_csv(paths.path("mavedb_estimator_comparison"), sep="\t").set_index("urn")
    return {u: {"replicates": s.loc[u, "reconciling_replicates"].split(";"),
                "error_column": e.loc[u, "error_column"]} for u in DEPOSITS}


def is_synonymous(hgvs_pro: pd.Series) -> pd.Series:
    h = hgvs_pro.astype(str).str.strip()
    return h.str.endswith("=") | h.str.endswith("=)")


def prepared(urn: str, spec: dict) -> dict:
    """The deposit's usable rows, the error scale and its s."""
    df = load(urn)
    reps, col = spec["replicates"], spec["error_column"]
    k = len(reps)
    x = pd.DataFrame({"score": pd.to_numeric(df["score"], errors="coerce"),
                      "err": pd.to_numeric(df[col], errors="coerce"),
                      **{f"r{j}": pd.to_numeric(df[c], errors="coerce") for j, c in enumerate(reps)},
                      "hgvs_pro": df.get("hgvs_pro", pd.Series(np.nan, index=df.index)),
                      "id": df["accession"]})
    rc_ = [f"r{j}" for j in range(k)]
    x = x.dropna(subset=["score", "err"] + rc_)
    x = x[x["err"] > 0].reset_index(drop=True)
    R = float((x[rc_].var(axis=1, ddof=1) / k).mean() / (x["err"] ** 2).mean())
    d_se, d_sd = abs(np.log(R)), abs(np.log(R * k))
    scale = "se_of_mean" if d_se <= d_sd else "sd_of_single"
    unclear = bool(min(d_se, d_sd) > np.log(SCALE_FACTOR))
    s = x["err"].to_numpy(float) / (np.sqrt(k) if (scale == "sd_of_single" and not unclear) else 1.0)
    return {"x": x, "reps": rc_, "k": k, "R": R, "scale": scale, "unclear": unclear, "s": s,
            "error_column": col,
            "max_diff": float((x["score"] - x[rc_].mean(axis=1)).abs().max())}


# ==========================================================================
# simulation of k single replicates
# ==========================================================================
def predicted_replicates(y, s, em: rc.ErrorModel, k: int, rng) -> tuple[float, float]:
    """(the mean pairwise Spearman rho of k simulated single replicates, error scale
    s sqrt(k); the combined score's ceiling, error scale s), one fit."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    g, L, w = rc.fit(y, s, em, rc.ITER_POINT)
    ceiling = float(np.sqrt(max(rc.remeasure(L, w, g, s, em, rng, rc.REPEAT_POINT), 0.0)))
    P = L * w
    P /= P.sum(1, keepdims=True)
    cdf = np.cumsum(P, 1)
    n, last, sk = len(y), len(g) - 1, s * np.sqrt(k)
    means = []
    for _ in range(rc.REPEAT_POINT):
        t = g[np.minimum((rng.random(n)[:, None] > cdf).sum(1), last)]
        reps = [t + sk * em.draw(n, rng) for _ in range(k)]
        means.append(np.mean([rc.spearman(reps[a], reps[b])
                              for a, b in combinations(range(k), 2)]))
    return float(np.mean(means)), ceiling


def observed(R: np.ndarray) -> tuple[float, float]:
    """Mean pairwise Spearman and Pearson of the columns of R."""
    k = R.shape[1]
    sp = [rc.spearman(R[:, a], R[:, b]) for a, b in combinations(range(k), 2)]
    pe = [float(np.corrcoef(R[:, a], R[:, b])[0, 1]) for a, b in combinations(range(k), 2)]
    return float(np.mean(sp)), float(np.mean(pe))


def replicate_shape(R: np.ndarray, s) -> np.ndarray:
    """z_ij = (r_ij - mean_i) / (s_i sqrt(k - 1)), raveled; s one value or one per row."""
    k = R.shape[1]
    s = np.broadcast_to(np.asarray(s, float), (R.shape[0],))
    return ((R - R.mean(axis=1)[:, None]) / (s[:, None] * np.sqrt(k - 1))).ravel()


def _evaluate(task: tuple) -> dict:
    """One deposit or BRCA1 cell: task = (row header, ids, y, the error's square per
    variant for the variance formula, replicate matrix, k, {model: spec}), a spec being
    a reason it is not computable (text) or {"s": scale per variant, "z": the shape's
    residuals (None: Gaussian), "per_variant": whether z is one residual per variant,
    aligned with ids}."""
    head, ids, y, err2, R, k, specs = task
    from atlas.matched_ceilings import cell_seed
    order = np.argsort(np.asarray(ids, str), kind="stable")
    y, R = np.asarray(y, float)[order], np.asarray(R, float)[order]
    err2 = np.asarray(err2, float)[order]
    r_obs, p_obs = observed(R)
    row = dict(head) | {"n": int(len(y)), "k": int(k), "r_obs": r_obs, "pearson_obs": p_obs,
                        "bound_sqrt_r": float(np.sqrt(max(r_obs, 0.0))),
                        "bound_sqrt_sb_k": float(np.sqrt(max(spearman_brown(r_obs, k), 0.0))),
                        "variance_formula": float(np.sqrt(max(1 - err2.mean() / y.var(ddof=1),
                                                              0.0)))}
    notes = [row.get("note", "")] if row.get("note") else []
    for model in MODELS:
        spec = specs[model]
        if isinstance(spec, str):
            row[f"pred_{model}"] = row[f"ceiling_{model}_model"] = np.nan
            row[f"pred_minus_obs_{model}"] = np.nan
            notes.append(f"{model}: {spec}")
            continue
        s = np.asarray(spec["s"], float)[order]
        z = spec["z"]
        if z is not None and spec.get("per_variant"):
            z = np.asarray(z, float)[order]                 # per variant (BRCA1's d)
        em = rc.ErrorModel() if z is None else rc.ErrorModel(np.asarray(z, float))
        rng = np.random.default_rng(cell_seed(("rank_validation", row["urn"], row["population"],
                                               row["stratum"], model)))
        pred, ceil = predicted_replicates(y, s, em, k, rng)
        row[f"pred_{model}"], row[f"ceiling_{model}_model"] = pred, ceil
        row[f"pred_minus_obs_{model}"] = pred - r_obs
    row["note"] = "; ".join(notes)
    return row


# ==========================================================================
# the tasks
# ==========================================================================
def neutral_spec(q: dict) -> tuple[dict | str, dict]:
    """The neutral model of a deposit from the synonymous variants of q (a prepared
    deposit): sigma(s) calibrated on them, and the shape z = r / sigma(s)."""
    syn = is_synonymous(q["x"]["hgvs_pro"]).to_numpy()
    if syn.sum() < 2 * rc.SCALE_BINS:
        return "not computable: no synonymous variants recognised", {"n": int(syn.sum())}
    ys, ss = q["x"]["score"].to_numpy(float)[syn], q["s"][syn]
    r = ys - np.median(ys)
    try:
        sc = rc.ErrorScale(ss, r)
    except ValueError as exc:
        return f"not computable: {exc}", {"n": int(syn.sum())}
    return {"scale": sc, "z": r / sc(ss)}, {"n": int(syn.sum()),
                                            "largest_block": sc.largest_block(),
                                            "table": sc.table().to_dict("records")}


def deposit_tasks(specs: dict) -> tuple[list[tuple], dict]:
    tasks, checks = [], {}
    prep = {u: prepared(u, specs[u]) for u in DEPOSITS}
    for urn in DEPOSITS:
        p = prep[urn]
        x, k, s = p["x"], p["k"], p["s"]
        R = x[p["reps"]].to_numpy(float)
        pooled = rc.pooled_scale(s)
        src = SYNONYMOUS_FROM.get(urn, urn)
        neu, neu_rec = neutral_spec(prep[src])
        models = {
            "replicate": {"s": np.full(len(s), pooled), "z": replicate_shape(R, pooled)},
            "per_variant": {"s": s, "z": replicate_shape(R, s)},
            "own": {"s": np.full(len(s), pooled), "z": None},
            "neutral": neu if isinstance(neu, str) else {"s": neu["scale"](s), "z": neu["z"]}}
        head = {"urn": urn, "experiment": EXPERIMENT[urn], "population": "all",
                "stratum": "deposit", "score_vs_mean_max_abs_diff": p["max_diff"],
                "error_column": p["error_column"], "R": p["R"], "error_scale": p["scale"],
                "scale_unclear": p["unclear"], "pooled_s": pooled, "n_neutral": neu_rec["n"],
                "neutral_source": "" if isinstance(neu, str) else src,
                "neutral_largest_block": neu_rec.get("largest_block", np.nan)}
        tasks.append((head, x["id"].tolist(), x["score"].to_numpy(float), s ** 2, R, k, models))
        checks[urn] = {"rows_used": int(len(x)), "k": k, "replicates": specs[urn]["replicates"],
                       "error_column": p["error_column"], "R": p["R"],
                       "error_scale": p["scale"], "scale_unclear": p["unclear"],
                       "score_vs_mean_max_abs_diff": p["max_diff"], "pooled_s": pooled,
                       "synonymous_used": neu_rec["n"], "synonymous_from": src,
                       "neutral_scale": neu_rec.get("table")}
    return tasks, checks


def brca1_tasks() -> tuple[list[tuple], dict]:
    from atlas.error_models import brca1_mean_check, core_frame, neutral_variants
    d = core_frame()
    b = d[d["gene"].eq("BRCA1") & d["has_error"] & d["population"].eq(population.SNV)]
    zn = neutral_variants("BRCA1")["z"].to_numpy(float)
    chk = brca1_mean_check()
    tasks = []
    for stratum in STRATA:
        c = b[_stratum_mask(b["region"], stratum)]
        if len(c) < MIN_N:
            continue
        dd = c["d"].to_numpy(float)
        ones = np.ones(len(c))
        head = {"urn": BRCA1, "experiment": EXPERIMENT[BRCA1], "population": population.SNV,
                "stratum": stratum, "score_vs_mean_max_abs_diff": chk["max_abs_diff"],
                "error_column": "none (d within the cell)", "R": np.nan, "error_scale": "",
                "scale_unclear": False, "pooled_s": np.nan, "n_neutral": int(len(zn)),
                "neutral_source": "atlas.error_models (neutral SNVs)",
                "neutral_largest_block": np.nan}
        reps = np.column_stack([-c["fit_rep1"].to_numpy(float), -c["fit_rep2"].to_numpy(float)])
        models = {"replicate": {"s": ones, "z": dd, "per_variant": True},
                  "per_variant": "not computable: no per-variant error (d within the cell, "
                                 "s = 1, is the replicate column)",
                  "own": "not computable: the deposit has no error column",
                  "neutral": {"s": ones, "z": zn}}
        tasks.append((head, c["variant_id"].tolist(),
                      c["functional_pathogenicity"].to_numpy(float), dd ** 2, reps, 2, models))
    return tasks, {"BRCA1": chk | {"population": "all 3,893 variants are SNVs"}}


def finish(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dep = rows["urn"].ne(BRCA1)
    rows["acceptance_applies"] = np.where(dep, (rows["n"] >= MIN_N_DEPOSIT) & ~rows["scale_unclear"],
                                          rows["n"] >= MIN_N_BRCA1)
    rows["passes"] = pd.Series(rows["pred_minus_obs_replicate"].abs() <= TOL,
                               dtype="boolean").where(rows["acceptance_applies"])
    applies = rows[rows["acceptance_applies"]]
    acc = {"rows_applying": int(len(applies)), "rows_passing": int(applies["passes"].sum()),
           "holds": bool(applies["passes"].all()),
           "largest_abs_difference": float(applies["pred_minus_obs_replicate"].abs().max()),
           "difference_range": [float(applies["pred_minus_obs_replicate"].min()),
                                float(applies["pred_minus_obs_replicate"].max())],
           "rule": f"|replicate prediction - r_obs| <= {TOL} for deposits with n >= "
                   f"{MIN_N_DEPOSIT} and a clear error scale, BRCA1 cells with n >= {MIN_N_BRCA1}",
           "failing": applies.loc[~applies["passes"].astype(bool),
                                  ["urn", "population", "stratum", "n", "r_obs",
                                   "pred_replicate"]].to_dict("records")}
    return rows.reindex(columns=COLUMNS), acc


def definition() -> dict:
    return {"deposits": DEPOSITS + [BRCA1], "experiment": EXPERIMENT,
            "synonymous_from": SYNONYMOUS_FROM,
            "scale_rule": "R = mean(replicate sample variance / k) / mean(error^2); nearer 1 "
                          "(log scale): SE of the mean, nearer 1/k: SD of one replicate; off "
                          f"both by more than x{SCALE_FACTOR:g}: scale_unclear (taken as SE of "
                          "the mean)",
            "predictions": {
                "replicate": "the acceptance; pooled s = sqrt(mean s_i^2), shape z_ij = (r_ij - "
                             "mean_i) / (s sqrt(k - 1)); BRCA1 d within the cell, s = 1",
                "per_variant": "a diagnostic; as replicate with s_i variant by variant",
                "own": "Gaussian, pooled s",
                "neutral": "sigma(s) calibrated on the synonymous variants (five bins by s, "
                           "1.4826 MAD, non-decreasing, log-linear; atlas.rank_ceiling."
                           "ErrorScale), s'_i = sigma(s_i), shape z = r / sigma(s); BRCA1 its "
                           "neutral SNVs, z = r, s = 1"},
            "simulation": "k single replicates with error scale s sqrt(k) from the NPMLE on "
                          "the combined score (point settings); the mean pairwise Spearman rho",
            "variance_formula": "sqrt(1 - mean(s_i^2) / var(score)); BRCA1 mean(d^2), the "
                                "square of its replicate-derived SE",
            "seed": "cell_seed(('rank_validation', urn, population, stratum, model))",
            "estimator": rc.definition()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", default="run", choices=["fetch", "run"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(paths.RESULTS))
    args = ap.parse_args(argv)
    if args.step == "fetch":
        print(json.dumps(fetch(), indent=1))
        return 0
    t0 = time.time()
    specs = specification()
    t1, c1 = deposit_tasks(specs)
    t2, c2 = brca1_tasks()
    tasks = t1 + t2
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_evaluate, tasks))
    else:
        rows = [_evaluate(t) for t in tasks]
    table, acc = finish(pd.DataFrame(rows))
    out = Path(args.out)
    table.to_csv(out / paths.name("rank_ceiling_validation"), sep="\t", index=False)
    rec = {"definition": definition() | {
               "fetch_log": json.loads(FETCH_LOG.read_text()),
               "inputs": [paths.name("mavedb_survey"), paths.name("mavedb_estimator_comparison"),
                          paths.name("score_matrix"), paths.name("assay_aux")]},
           "checks": c1 | c2, "acceptance": acc}
    (out / paths.name("rank_ceiling_validation_summary")).write_text(
        json.dumps(rc.jsonable(rec), indent=1) + "\n")
    show = ["urn", "stratum", "n", "k", "R", "r_obs"] + [f"pred_{m}" for m in MODELS] + [
        "ceiling_replicate_model", "ceiling_neutral_model", "bound_sqrt_r", "bound_sqrt_sb_k",
        "variance_formula", "passes"]
    print(table[show].to_string(float_format=lambda v: f"{v:.3f}"))
    print(json.dumps(rc.jsonable(acc), indent=1))
    print(json.dumps({"workers": args.workers, "seconds_wall": round(time.time() - t0, 1)}))
    return 0 if acc["holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
