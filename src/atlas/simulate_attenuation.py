"""Does the attenuation correction actually recover the true correlation?

The territory map's central claim is that a predictor's ρ should be read
against the ceiling √(assay reliability), and that ρ_observed / ceiling —
"realisation" — estimates what the predictor would score against an error-free
measurement. That is a classical result for Pearson correlation with a normally
distributed error, and this study applies it to Spearman ρ on assay scores whose
distributions are far from normal and, at canonical splice sites, severely
compressed. Whether it holds there is an empirical question, so this module
answers it by simulation.

The generative model mirrors the real data as closely as possible:

* true functional scores are drawn by resampling the *observed* score
  distribution of one gene in one stratum, so the marginal shape — including the
  compression against the damaging extreme at splice ±1–2 — is the real one.
  The resampling must be per gene: the seven assays report on incompatible
  scales (per-gene SDs span 0.066 to 6.8), so a pooled marginal would be a
  mixture whose variance is set by the scale differences rather than by the
  within-assay spread, and the injected noise would be calibrated against the
  wrong quantity. This is the same reason the atlas correlates per gene;
* the predictor is coupled to the true score at a specified Spearman ρ through
  a Gaussian copula (Spearman is invariant to the marginals, so the coupling is
  exact);
* two replicates are formed by adding independent Gaussian error of fixed
  standard deviation, and the "deposited score" is their mean — exactly the
  construction verified for BRCA1;
* reliability is then estimated the way the pipeline estimates it, from the
  replicate correlation via Spearman–Brown, rather than from the known truth.

The quantity under test is therefore the whole chain, estimator included, not
just the algebraic identity. Compression enters through the ratio of true-score
variance to fixed error variance, which is why the marginal must be resampled
rather than assumed normal.

v2 (reviewer-requested hardening)
---------------------------------
The v1 grid and its delivered files (``attenuation_simulation_v1.tsv/json``)
are left untouched — ``main`` no longer rewrites them; ``--legacy-v1``
reproduces them. The v2 run writes ``attenuation_simulation_v2.tsv/json`` and
``attenuation_simulation_v2_curves.tsv`` and extends v1 in four ways:

a. a denser reliability grid: 0.12, 0.15, 0.20 and 0.30 are inserted between
   the v1 anchors (ten targets in all), so the RMSE crossover that brackets
   the 0.45 correction bound can be located more finely;
b. every trial is scored by BOTH reliability estimators the pipeline uses:
   the replicate chain (Spearman–Brown on the two replicates) and the
   SE-based chain (``1 - mean(SE^2)/var(score)``) fed the *known* per-variant
   standard error of the two-replicate mean. The SE chain is what BARD1 and
   PALB2 actually use, so the two estimators are compared on identical
   simulated data;
c. three noise scenarios run in parallel:
   ``gaussian``        the v1 construction: homoscedastic error of fixed SD;
   ``score_dependent`` error SD grows with |true score|,
                       sigma(t) = c * (1 + q*|t|/sd(T)). The shape q is
                       calibrated from the data, not asserted: regressing
                       BRCA1's derived per-variant SE (sd of its two
                       replicates over sqrt(2), ``atlas.assay_aux``) on
                       |score| gives se = 0.170 + 0.072*|score|, i.e.
                       q = b*sd(T)/a ~= 0.39 on the pooled assay. c is then
                       fixed per cell so the mean squared error still hits
                       the reliability target — the scenario changes the
                       *distribution* of the error across scores, not its
                       overall magnitude;
   ``bounded``         homoscedastic Gaussian error, but both replicates are
                       clipped to the observed score range of the cell —
                       assay scores saturate, so extreme errors cannot
                       materialise. The reported SE is the nominal
                       (pre-truncation) one, which is exactly the
                       mis-specification a deposited SE column would carry;
d. an explicit RMSE-crossover table: for every (stratum, estimator, noise)
   combination, the bias and RMSE of the corrected estimate are tabulated
   against the achieved ceiling (the curves file), and the bracket of
   reliability targets across which the correction stops helping is reported
   in the JSON.

Usage (PYTHONPATH=src):
    python -m atlas.simulate_attenuation --out results/            # v2 only
    python -m atlas.simulate_attenuation --legacy-v1 --trials 150  # reproduce v1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from atlas.evaluate import classify_region
from atlas.reliability import spearman_brown

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

# strata whose real marginal shapes are used, with the per-gene n they carry
STRATA = ["coding_or_utr", "splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]
MIN_MARGINAL = 60      # variants needed in a gene x stratum cell to resample it
RHO_TRUE = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8]
# error SDs chosen per stratum to span the reliabilities actually observed
RELIABILITY_TARGETS = [0.1, 0.25, 0.5, 0.65, 0.8, 0.95]

# --------------------------------------------------------------------------
# v2 grid: denser reliability targets, two estimators, three noise scenarios
# --------------------------------------------------------------------------
RELIABILITY_TARGETS_V2 = [0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.50, 0.65, 0.80, 0.95]
NOISE_SCENARIOS = ["gaussian", "score_dependent", "bounded"]
ESTIMATORS = ["replicates", "se"]
SEED_V2 = 20260804          # same seed value as v1, its own generator


def calibrate_score_dependent(se: np.ndarray, score: np.ndarray) -> dict:
    """Fit se = a + b*|score| and return the scale-free shape q = b*sd(score)/a.

    The shape is what the score_dependent scenario needs: error of the
    two-replicate mean for a variant with true score t is modelled as
    m(t) = c * (1 + q*|t|/sd(T)), so q dimensionless and transferable across
    assays whose score scales differ by two orders of magnitude. On BRCA1's
    derived SEs this fits to a = 0.170, b = 0.072, q = 0.386.
    """
    se = np.asarray(se, dtype=float)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(se) & np.isfinite(score)
    x = np.abs(score[ok])
    y = se[ok]
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    sd_t = float(score[ok].std())
    return {"a": float(a), "b": float(b), "sd_score": sd_t,
            "q": float(b * sd_t / a), "n": int(ok.sum())}


def gaussian_copula(n: int, rho_s: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two standard normals whose Spearman correlation is (in expectation) rho_s."""
    r = 2.0 * math.sin(math.pi * rho_s / 6.0)   # Pearson latent giving Spearman rho_s
    cov = np.array([[1.0, r], [r, 1.0]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    return z[:, 0], z[:, 1]


def _error_sd_for(true_scores: np.ndarray, target_reliability: float) -> float:
    """Error SD giving the requested reliability of a two-replicate mean.

    reliability of the mean = var(T) / (var(T) + sigma^2 / 2)
    """
    var_t = float(np.var(true_scores))
    if target_reliability >= 1.0:
        return 0.0
    return math.sqrt(max(2.0 * var_t * (1.0 - target_reliability) / target_reliability, 0.0))


def one_trial(marginal: np.ndarray, n: int, rho_true: float, target_rel: float,
              rng: np.random.Generator) -> dict:
    z_pred, z_true = gaussian_copula(n, rho_true, rng)

    # true score: the real marginal shape, ordered to follow z_true
    draws = rng.choice(marginal, size=n, replace=True)
    true_score = np.sort(draws)[stats.rankdata(z_true, method="ordinal").astype(int) - 1]

    sigma = _error_sd_for(true_score, target_rel)
    rep1 = true_score + rng.normal(0.0, sigma, n)
    rep2 = true_score + rng.normal(0.0, sigma, n)
    observed = (rep1 + rep2) / 2.0

    rho_obs = float(stats.spearmanr(z_pred, observed).statistic)
    rho_ref = float(stats.spearmanr(z_pred, true_score).statistic)

    rep_rho = float(stats.spearmanr(rep1, rep2).statistic)
    rel_hat = spearman_brown(rep_rho)
    ceiling_hat = math.sqrt(max(rel_hat, 0.0)) if np.isfinite(rel_hat) else np.nan
    realisation = rho_obs / ceiling_hat if ceiling_hat and ceiling_hat > 0 else np.nan

    return {
        "n": n, "rho_true_target": rho_true, "reliability_target": target_rel,
        "rho_true_realised": rho_ref,        # truth, on this sample
        "rho_observed": rho_obs,
        "reliability_hat": rel_hat, "ceiling_hat": ceiling_hat,
        "realisation": realisation,
        "err_uncorrected": rho_obs - rho_ref,
        "err_corrected": (realisation - rho_ref) if np.isfinite(realisation) else np.nan,
    }


def run(matrix: Path, n_trials: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(matrix, columns=["gene", "hgvs_c", "functional_pathogenicity"])
    df["region"] = df["hgvs_c"].map(classify_region)
    rng = np.random.default_rng(seed)

    rows = []
    for stratum in STRATA:
        sub = df[df["region"] == stratum]
        for gene, g in sub.groupby("gene"):
            marginal = g["functional_pathogenicity"].dropna().to_numpy()
            if len(marginal) < MIN_MARGINAL:
                continue
            n = len(marginal)          # the real per-gene n for this cell
            for rho_true in RHO_TRUE:
                for target_rel in RELIABILITY_TARGETS:
                    for _ in range(n_trials):
                        rec = one_trial(marginal, n, rho_true, target_rel, rng)
                        rec["stratum"] = stratum
                        rec["gene"] = gene
                        rows.append(rec)
    return pd.DataFrame(rows)


def summarise(sim: pd.DataFrame) -> pd.DataFrame:
    g = sim.groupby(["stratum", "reliability_target", "rho_true_target"])
    return g.agg(
        trials=("err_corrected", "size"),
        ceiling_hat=("ceiling_hat", "mean"),
        rho_true=("rho_true_realised", "mean"),
        rho_obs=("rho_observed", "mean"),
        realisation=("realisation", "mean"),
        bias_uncorrected=("err_uncorrected", "mean"),
        bias_corrected=("err_corrected", "mean"),
        rmse_uncorrected=("err_uncorrected", lambda s: float(np.sqrt(np.mean(s ** 2)))),
        rmse_corrected=("err_corrected", lambda s: float(np.sqrt(np.nanmean(s ** 2)))),
    ).reset_index()


def _ranks_average(a: np.ndarray) -> np.ndarray:
    """Row-wise average (tie-corrected) ranks, one argsort per call.

    scipy.stats.rankdata pays for two argsorts and array-api dispatch; on the
    v2 grid ranking is the dominant cost, so the tie-averaging is done here by
    group means over the sorted rows.
    """
    T, n = a.shape
    order = a.argsort(axis=1, kind="stable")
    sorted_a = np.take_along_axis(a, order, axis=1)
    base = np.broadcast_to(np.arange(n, dtype=float), (T, n))
    # group id of each sorted position: increments wherever the value changes
    change = np.diff(sorted_a, axis=1) != 0
    gid = np.concatenate([np.zeros((T, 1), dtype=int), change.cumsum(axis=1)], axis=1)
    flat = (gid + np.arange(T)[:, None] * n).ravel()
    cnt = np.bincount(flat, minlength=T * n).reshape(T, n)
    sums = np.bincount(flat, weights=base.ravel(), minlength=T * n).reshape(T, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(cnt > 0, sums / cnt, 0.0)
    avg_sorted = np.take_along_axis(avg, gid, axis=1)
    ranks = np.empty((T, n))
    np.put_along_axis(ranks, order, avg_sorted, axis=1)
    return ranks


def _ranks_ordinal(a: np.ndarray) -> np.ndarray:
    """Row-wise ordinal ranks (no ties possible), one argsort per call."""
    order = a.argsort(axis=1, kind="stable")
    ranks = np.empty(a.shape, dtype=float)
    np.put_along_axis(ranks, order,
                      np.broadcast_to(np.arange(a.shape[1], dtype=float), a.shape),
                      axis=1)
    return ranks


def _corr_from_ranks(rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two rank matrices == Spearman rho."""
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    denom = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (rx * ry).sum(axis=1) / denom, np.nan)


def batch_cell_v2(marginal: np.ndarray, n: int, rho_true: float,
                  targets: np.ndarray, q: float, n_trials: int,
                  rng: np.random.Generator, block: int = 400,
                  noises: tuple[str, ...] = tuple(NOISE_SCENARIOS)) -> pd.DataFrame:
    """The v2 trials of one (cell, true-rho) grid row, all three noise scenarios.

    The three scenarios share one draw of the latent coupling and of the true
    scores — they differ only in how measurement error is formed — so their
    comparison is paired on identical truths. Every trial is scored by both
    reliability estimators on the same simulated replicates. ``q`` is the
    calibrated score-dependence shape (see calibrate_score_dependent); only
    ``score_dependent`` uses it. Trials are processed in blocks so the (T, n)
    arrays stay modest.
    """
    r = 2.0 * math.sin(math.pi * rho_true / 6.0)   # latent Pearson for Spearman rho
    cov = np.array([[1.0, r], [r, 1.0]])
    lo, hi = float(marginal.min()), float(marginal.max())
    tvec = np.repeat(np.asarray(targets, dtype=float), n_trials)
    m = len(marginal)

    out = []
    for start in range(0, len(tvec), block):
        tv = tvec[start:start + block]
        T = len(tv)
        z = rng.multivariate_normal([0.0, 0.0], cov, size=(T, n))
        z_pred, z_true = z[..., 0], z[..., 1]

        draws = marginal[rng.integers(0, m, size=(T, n))]
        idx = _ranks_ordinal(z_true).astype(int)
        true_score = np.take_along_axis(np.sort(draws, axis=1), idx, axis=1)

        rz = _ranks_ordinal(z_pred)
        rtrue = _ranks_average(true_score)          # ties from resampling
        rho_ref = _corr_from_ranks(rz, rtrue)
        var_t = true_score.var(axis=1)
        # error variance of the two-replicate mean that hits the reliability target
        err_var_mean = var_t * (1.0 - tv) / tv

        for noise in noises:
            if noise == "score_dependent":
                with np.errstate(invalid="ignore", divide="ignore"):
                    shape = np.where(var_t[:, None] > 0,
                                     1.0 + q * np.abs(true_score) / np.sqrt(var_t)[:, None],
                                     1.0)
                c2 = np.divide(err_var_mean, (shape ** 2).mean(axis=1),
                               out=np.zeros(T), where=(shape ** 2).mean(axis=1) > 0)
                se_known = np.sqrt(c2)[:, None] * shape   # SD of the deposited mean's error
            else:
                se_known = np.sqrt(err_var_mean)[:, None] * np.ones((1, n))
            sigma = math.sqrt(2.0) * se_known             # per-replicate SD

            rep1 = true_score + rng.normal(0.0, 1.0, (T, n)) * sigma
            rep2 = true_score + rng.normal(0.0, 1.0, (T, n)) * sigma
            if noise == "bounded":
                rep1 = np.clip(rep1, lo, hi)
                rep2 = np.clip(rep2, lo, hi)
            observed = (rep1 + rep2) / 2.0

            rho_obs = _corr_from_ranks(rz, _ranks_average(observed))
            rep_rho = _corr_from_ranks(_ranks_average(rep1), _ranks_average(rep2))

            # estimator 1: the replicate chain (Spearman-Brown on the pair)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel_rep = np.where(np.isfinite(rep_rho) & (1.0 + rep_rho != 0.0),
                                   2.0 * rep_rho / (1.0 + rep_rho), np.nan)
            ceil_rep = np.sqrt(np.clip(rel_rep, 0.0, None))

            # estimator 2: the SE chain, fed the known per-variant SE of the mean
            var_obs = observed.var(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel_se = np.where(var_obs > 0,
                                  np.clip(1.0 - (se_known ** 2).mean(axis=1) / var_obs,
                                          0.0, 1.0), np.nan)
            ceil_se = np.sqrt(np.clip(rel_se, 0.0, None))

            rec = pd.DataFrame({
                "n": n, "rho_true_target": rho_true, "reliability_target": tv,
                "noise": noise, "rho_true_realised": rho_ref, "rho_observed": rho_obs,
                "err_uncorrected": rho_obs - rho_ref,
                "reliability_hat_replicates": rel_rep, "ceiling_hat_replicates": ceil_rep,
                "reliability_hat_se": rel_se, "ceiling_hat_se": ceil_se,
            })
            for est in ESTIMATORS:
                ceil = rec[f"ceiling_hat_{est}"].to_numpy()
                with np.errstate(invalid="ignore", divide="ignore"):
                    real = np.where(ceil > 0, rec["rho_observed"].to_numpy() / ceil, np.nan)
                rec[f"realisation_{est}"] = real
                rec[f"err_corrected_{est}"] = real - rho_ref
            out.append(rec)
    return pd.concat(out, ignore_index=True)


def sim_cells(matrix: Path) -> list[dict]:
    """The gene x stratum cells whose real marginals the grid resamples."""
    df = pd.read_parquet(matrix, columns=["gene", "hgvs_c", "functional_pathogenicity"])
    df["region"] = df["hgvs_c"].map(classify_region)
    cells = []
    for stratum in STRATA:
        sub = df[df["region"] == stratum]
        for gene, g in sorted(sub.groupby("gene"), key=lambda kv: kv[0]):
            marginal = g["functional_pathogenicity"].dropna().to_numpy()
            if len(marginal) < MIN_MARGINAL:
                continue
            cells.append({"stratum": stratum, "gene": gene, "marginal": marginal})
    return cells


def _cell_worker(job: dict) -> pd.DataFrame:
    """All v2 grid rows for one cell, with a cell-specific deterministic RNG.

    Per-cell seeding keeps the run reproducible however the cells are
    scheduled across worker processes.
    """
    rng = np.random.default_rng(
        [job["seed"], zlib.crc32(job["stratum"].encode()),
         zlib.crc32(job["gene"].encode())])
    targets = np.asarray(RELIABILITY_TARGETS_V2, dtype=float)
    marginal = job["marginal"]
    frames = []
    for rho_true in RHO_TRUE:
        rec = batch_cell_v2(marginal, len(marginal), rho_true, targets, job["q"],
                            job["n_trials"], rng, noises=tuple(job["noises"]))
        rec["stratum"] = job["stratum"]
        rec["gene"] = job["gene"]
        frames.append(rec)
    return pd.concat(frames, ignore_index=True)


def run_v2(matrix: Path, n_trials: int, seed: int, q: float,
           noises: list[str] | None = None, workers: int = 1) -> pd.DataFrame:
    wanted = list(noises or NOISE_SCENARIOS)
    jobs = [{**cell, "seed": seed, "q": q, "n_trials": n_trials, "noises": wanted}
            for cell in sim_cells(matrix)]
    if workers and workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(_cell_worker, jobs))
    else:
        frames = [_cell_worker(job) for job in jobs]
    return pd.concat(frames, ignore_index=True)


def _long(sim: pd.DataFrame) -> pd.DataFrame:
    """One row per trial x estimator, so summaries group directly."""
    frames = []
    for est in ESTIMATORS:
        f = sim[["noise", "stratum", "gene", "reliability_target", "rho_true_target",
                 "rho_true_realised", "rho_observed", "err_uncorrected",
                 f"reliability_hat_{est}", f"ceiling_hat_{est}", f"realisation_{est}",
                 f"err_corrected_{est}"]].copy()
        f.columns = ["noise", "stratum", "gene", "reliability_target", "rho_true_target",
                     "rho_true_realised", "rho_observed", "err_uncorrected",
                     "reliability_hat", "ceiling_hat", "realisation", "err_corrected"]
        f["estimator"] = est
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def summarise_v2(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby(["noise", "estimator", "stratum", "reliability_target",
                      "rho_true_target"])
    return g.agg(
        trials=("err_corrected", "size"),
        ceiling_hat=("ceiling_hat", "mean"),
        rho_true=("rho_true_realised", "mean"),
        rho_obs=("rho_observed", "mean"),
        realisation=("realisation", "mean"),
        bias_uncorrected=("err_uncorrected", "mean"),
        bias_corrected=("err_corrected", "mean"),
        rmse_uncorrected=("err_uncorrected", lambda s: float(np.sqrt(np.mean(s ** 2)))),
        rmse_corrected=("err_corrected", lambda s: float(np.sqrt(np.nanmean(s ** 2)))),
    ).reset_index()


def curve_table(long: pd.DataFrame) -> pd.DataFrame:
    """Bias and RMSE against the achieved ceiling, per (noise, estimator, stratum).

    Pooled over the true-rho grid: this is the curve the crossover is read from.
    """
    g = long.groupby(["noise", "estimator", "stratum", "reliability_target"])
    return g.agg(
        trials=("err_corrected", "size"),
        achieved_ceiling=("ceiling_hat", "mean"),
        bias_uncorrected=("err_uncorrected", "mean"),
        bias_corrected=("err_corrected", "mean"),
        rmse_uncorrected=("err_uncorrected", lambda s: float(np.sqrt(np.mean(s ** 2)))),
        rmse_corrected=("err_corrected", lambda s: float(np.sqrt(np.nanmean(s ** 2)))),
    ).reset_index()


def crossover_brackets(curves: pd.DataFrame,
                       by: list[str] | None = None) -> list[dict]:
    """Where correction RMSE crosses below uncorrected RMSE, per group.

    The bracket is the pair of adjacent reliability targets between which the
    sign of (rmse_corrected - rmse_uncorrected) flips from non-negative to
    negative, with the corresponding achieved ceilings — so the crossover can
    be quoted either against the design grid or against what an assay would
    actually measure. ``crossover_lo`` is None when the correction already
    helps at the lowest target, and the row is flagged ``never`` when it does
    not help anywhere on the grid.
    """
    by = by or ["noise", "estimator", "stratum"]
    out = []
    for keys, g in curves.groupby(by):
        g = g.sort_values("reliability_target")
        diff = (g["rmse_corrected"] - g["rmse_uncorrected"]).to_numpy()
        t = g["reliability_target"].to_numpy()
        ceil = g["achieved_ceiling"].to_numpy()
        rec = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        idx = next((i for i in range(1, len(diff)) if diff[i - 1] >= 0 > diff[i]), None)
        if idx is None:
            rec.update(crossover_lo=None, crossover_hi=None,
                       achieved_ceiling_lo=None, achieved_ceiling_hi=None,
                       direction=("always_helps_on_grid" if (diff < 0).all()
                                  else "never_helps_on_grid" if (diff >= 0).all()
                                  else "non_monotone"))
        else:
            rec.update(crossover_lo=float(t[idx - 1]), crossover_hi=float(t[idx]),
                       achieved_ceiling_lo=float(ceil[idx - 1]),
                       achieved_ceiling_hi=float(ceil[idx]),
                       direction="crosses")
        out.append(rec)
    return out


def load_score_dependent_calibration(matrix: Path, aux_path: Path) -> dict:
    """Calibrate q from BRCA1's derived per-variant SEs (atlas.assay_aux)."""
    aux = pd.read_parquet(aux_path, columns=["variant_id", "gene", "se"])
    mat = pd.read_parquet(matrix, columns=["variant_id", "functional_pathogenicity"])
    j = (aux[aux["gene"] == "BRCA1"].dropna(subset=["se"])
         .merge(mat, on="variant_id").dropna(subset=["functional_pathogenicity"]))
    cal = calibrate_score_dependent(j["se"].to_numpy(),
                                    j["functional_pathogenicity"].to_numpy())
    cal["source"] = ("BRCA1 derived SE (sd of two fitness replicates / sqrt(2), "
                     "atlas.assay_aux) regressed on |functional_pathogenicity|")
    return cal


def main_v2(args) -> int:
    matrix = Path(args.matrix)
    cal = load_score_dependent_calibration(matrix, Path(args.aux))
    q = cal["q"]
    print(f"score-dependent noise calibrated on BRCA1: se = {cal['a']:.3f} "
          f"+ {cal['b']:.3f}*|score|  (sd {cal['sd_score']:.3f}, q = {q:.3f})")

    sim = run_v2(matrix, args.trials, args.seed, q, workers=args.workers)
    long = _long(sim)
    summary = summarise_v2(long)
    curves = curve_table(long)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "attenuation_simulation_v2.tsv", sep="\t", index=False)
    curves.to_csv(out / "attenuation_simulation_v2_curves.tsv", sep="\t", index=False)

    overall = {}
    for (noise, est), g in long.groupby(["noise", "estimator"]):
        overall.setdefault(noise, {})[est] = {
            "bias_uncorrected_mean": float(g["err_uncorrected"].mean()),
            "bias_corrected_mean": float(g["err_corrected"].mean()),
            "rmse_uncorrected": float(np.sqrt(np.mean(g["err_uncorrected"] ** 2))),
            "rmse_corrected": float(np.sqrt(np.nanmean(g["err_corrected"] ** 2))),
        }
    doc = {
        "trials_per_cell": args.trials,
        "seed": args.seed,
        "reliability_targets": RELIABILITY_TARGETS_V2,
        "rho_true": RHO_TRUE,
        "noise_scenarios": {
            "gaussian": "homoscedastic Gaussian error (the v1 construction)",
            "score_dependent": ("error SD = c*(1 + q*|true score|/sd(true)), q calibrated "
                                "from BRCA1's derived SEs; c set per cell to hit the "
                                "reliability target"),
            "bounded": ("homoscedastic Gaussian error with both replicates clipped to the "
                        "cell's observed score range; the SE estimator is fed the nominal "
                        "pre-truncation SE"),
        },
        "score_dependent_calibration": cal,
        "overall": overall,
        "crossover": crossover_brackets(curves),
        "crossover_pooled_over_strata": crossover_brackets(
            curve_table(long.drop(columns=["stratum"]).assign(stratum="pooled")),
            by=["noise", "estimator", "stratum"]),
    }
    (out / "attenuation_simulation_v2.json").write_text(json.dumps(doc, indent=2) + "\n")

    print(f"\n{len(sim):,} trials: {len(NOISE_SCENARIOS)} noise scenarios x "
          f"{len(ESTIMATORS)} estimators x {sim['stratum'].nunique()} strata x "
          f"{len(RHO_TRUE)} true rho x {len(RELIABILITY_TARGETS_V2)} reliabilities")
    for noise in NOISE_SCENARIOS:
        print(f"\n--- noise: {noise} ---")
        piv = (summary[summary["noise"] == noise]
               .groupby(["estimator", "reliability_target"])
               .agg(ceiling=("ceiling_hat", "mean"),
                    corrected=("bias_corrected", "mean"),
                    rmse_uncorr=("rmse_uncorrected", "mean"),
                    rmse_corr=("rmse_corrected", "mean")))
        print(piv.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\nRMSE crossover brackets (corrected first beats uncorrected):")
    for rec in doc["crossover"]:
        lo, hi = rec["achieved_ceiling_lo"], rec["achieved_ceiling_hi"]
        span = (f"between achieved ceilings {lo:.3f} and {hi:.3f}" if lo is not None
                else rec["direction"])
        print(f"  {rec['noise']:<16s} {rec['estimator']:<11s} {rec['stratum']:<14s} {span}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=str(RESULTS / "score_matrix_atlas_v2.parquet"))
    ap.add_argument("--aux", default=str(RESULTS / "assay_aux_v1.parquet"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--seed", type=int, default=SEED_V2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="processes for the v2 grid (cells are simulated in parallel; "
                         "1 forces serial)")
    ap.add_argument("--legacy-v1", action="store_true",
                    help="reproduce the v1 grid and files instead (default: v2 only, "
                         "which leaves attenuation_simulation_v1.* untouched)")
    args = ap.parse_args(argv)

    if not args.legacy_v1:
        return main_v2(args)

    sim = run(Path(args.matrix), args.trials, args.seed)
    summary = summarise(sim)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "attenuation_simulation_v1.tsv", sep="\t", index=False)

    overall = {
        "trials_per_cell": args.trials,
        "seed": args.seed,
        "bias_uncorrected_mean": float(sim["err_uncorrected"].mean()),
        "bias_corrected_mean": float(sim["err_corrected"].mean()),
        "rmse_uncorrected": float(np.sqrt(np.mean(sim["err_uncorrected"] ** 2))),
        "rmse_corrected": float(np.sqrt(np.nanmean(sim["err_corrected"] ** 2))),
    }
    (out / "attenuation_simulation_v1.json").write_text(json.dumps(overall, indent=2) + "\n")

    print(f"{len(sim):,} trials over {sim['stratum'].nunique()} strata x "
          f"{len(RHO_TRUE)} true rho x {len(RELIABILITY_TARGETS)} reliabilities\n")
    print("Bias in estimating the error-free correlation "
          "(negative = underestimates the predictor):")
    piv = (summary.groupby("reliability_target")
                  .agg(ceiling=("ceiling_hat", "mean"),
                       uncorrected=("bias_uncorrected", "mean"),
                       corrected=("bias_corrected", "mean"),
                       rmse_uncorr=("rmse_uncorrected", "mean"),
                       rmse_corr=("rmse_corrected", "mean")))
    print(piv.to_string(float_format=lambda x: f"{x:+.3f}"))

    print("\nBy stratum (marginal shape), averaged over the grid:")
    print(summary.groupby("stratum")
                 .agg(uncorrected=("bias_uncorrected", "mean"),
                      corrected=("bias_corrected", "mean"),
                      rmse_corr=("rmse_corrected", "mean"))
                 .to_string(float_format=lambda x: f"{x:+.3f}"))

    print(f"\noverall RMSE: uncorrected {overall['rmse_uncorrected']:.3f} "
          f"-> corrected {overall['rmse_corrected']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
