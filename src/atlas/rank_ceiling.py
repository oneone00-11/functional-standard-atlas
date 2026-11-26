"""The rank-scale ceiling of a Spearman rho against a noisy measurement (A4b).

The manuscript compares predictors by Spearman's rho with the functional score. What
bounds that rho is the reliability on the rank scale: the expected Spearman rho between
two independent measurements of the same variants. Its square root is the ceiling. The
variance formula sqrt(1 - mean(SE^2) / var(score)) bounds a Pearson correlation; where
most variants have no effect the two part company (variants with the same true value
are ordered by noise alone, while a few large effects carry the variance).

A cell with one measurement per variant and a per-variant error scale s is measured
again by simulation:

  1 grid g: the 200 evenly spaced quantiles of y, with 120 evenly spaced points on
    [min y - 3 median s, max y + 3 median s], duplicates removed;
  2 likelihood L[i, j] = f((y_i - g_j) / s_i) / s_i, at least 1e-300, f the density
    of the standardised error;
  3 the distribution of the true values: the nonparametric maximum likelihood estimate
    on the grid (NPMLE), by EM from uniform weights, 1,000 iterations for a point
    estimate and 300 in each resample;
  4 per variant a true value t_i drawn from its posterior (proportional to
    L[i, j] w_j), two independent errors added (t + s e1, t + s e2), their Spearman rho
    (average ranks); 20 repeats for a point estimate and 10 in a resample, averaged;
  5 ceiling = sqrt(max(that mean, 0)).

Error models (the standardised error e):
  ``gaussian``   standard normal;
  ``empirical``  the distribution of residuals z: a Gaussian kernel with Silverman's
                 bandwidth h = 0.9 min(sd, IQR / 1.34) n^(-1/5); the density computed on
                 4,001 evenly spaced points over [min z - 5h, max z + 5h] and
                 normalised, 1e-300 outside; a draw is a residual drawn with
                 replacement plus h N(0, 1). The density and the draws are the same
                 distribution; the residuals are not clipped.

Error scales s. A deposited SE is not taken variant by variant: on neutral variants an
SE overstates how much the precision differs between variants (an SE computed from two
or three replicates is itself an estimate), and a per-variant SE taken as known
overstates the agreement of a remeasurement. Two scales instead:
  pooled      one value per cell, sqrt(mean(SE^2)) of its variants (``pooled_scale``;
              ``estimate_cell(pooled=True)``: of the cell for the point and the floor, of
              the resampled variants in each resample);
  calibrated  sigma(SE) (``ErrorScale``), from a noise sample of residuals with their SEs:
              the sample sorted by SE and cut into five equal-count bins as
              numpy.array_split cuts; per bin x_b = the median log SE and sigma_b = 1.4826
              x the median absolute deviation of the residuals from their median;
              sigma_b made non-decreasing in bin order (pool adjacent violators, equal
              weights); sigma(SE) linear in log SE between the points (x_b, sigma_b),
              the first bin's value below x_1 and the last bin's above x_5.
atlas.error_models puts them to use (own: pooled; neutral: calibrated on the gene's
neutral variants).

A cell (``estimate_cell``):
  point     as above;
  interval  bootstrap, B = 200 resamples of the variants with replacement, each
            estimated again (an error model built from the cell's own residuals, as
            BRCA1's replicate half-differences, is rebuilt from the resampled ones, and
            so is a pooled scale);
            the 2.5% and 97.5% quantiles;
  floor     100 data sets without any effect, y0 = s e (e drawn from the same error
            model), each estimated as a resample is; their 95th percentile. Without
            signal the estimate is not 0, hence the floor;
  estimable the bootstrap lower bound above the floor;
  seeds     atlas.matched_ceilings.cell_seed(("rank", group, gene, stratum,
            population, error model, the variant set's sha256)), the set being its
            sorted variant identifiers joined by newlines; the null's key ends in
            "null". The variants are put in that sorted order before anything is
            drawn, so one set of variants gets the same resamples and the same
            ceiling in every table, for every predictor, whatever the number of
            processes. The resamples' indices come first from the seed's own stream;
            each estimate draws from a stream of its own ([seed, 0] for the point,
            [seed, 1, b] for resample b, [null seed, j] for null data set j);
  realisation  rho / ceiling, its interval on the same 200 resamples (each resample
            gives rho and the ceiling).

Self-test: known truths, five kinds x two error distributions x 8 data sets (the
instruction's table), and data sets drawn from the true-value distribution estimated on
real cells (nine genes, 3-10 and 11-50 bp, SNVs) under both error models. The real
cells' error models live in atlas.error_models, which writes the self-test
(results/rank_ceiling_selftest_v3.tsv and .json; the first version had the own model's
cells with the SE taken variant by variant, the second added the neutral model's).
``python -m atlas.rank_ceiling selftest`` runs that.

Usage (PYTHONPATH=src):  python -m atlas.rank_ceiling selftest [--workers N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import paths, population

REPO = Path(__file__).resolve().parents[2]

GRID_QUANTILES = 200
GRID_EVEN = 120
GRID_PAD = 3.0
FLOOR_L = 1e-300
ITER_POINT = 1000
ITER_RESAMPLE = 300
REPEAT_POINT = 20
REPEAT_RESAMPLE = 10
B = 200
NULLS = 100
FLOOR_QUANTILE = 0.95
KDE_POINTS = 4001
KDE_PAD = 5.0
SCALE_BINS = 5
MAD_FACTOR = 1.4826


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rho with average ranks."""
    from atlas.matched_ceilings import spearman_rows
    return float(spearman_rows(np.asarray(a, float)[None, :], np.asarray(b, float)[None, :])[0])


# ==========================================================================
# error models
# ==========================================================================
class ErrorModel:
    """The standardised error: Gaussian (z is None) or the empirical distribution of
    the residuals z (Gaussian kernel, Silverman's bandwidth)."""

    def __init__(self, z: np.ndarray | None = None, name: str | None = None):
        self.z = None if z is None else np.asarray(z, float)
        self.name = name or ("gaussian" if self.z is None else "empirical")
        self.h = np.nan
        if self.z is not None:
            if len(self.z) < 2 or not np.all(np.isfinite(self.z)):
                raise ValueError("an empirical error model needs two or more finite residuals")
            sd = float(self.z.std())
            q75, q25 = np.percentile(self.z, [75, 25])
            self.h = 0.9 * min(sd, (q75 - q25) / 1.34) * len(self.z) ** (-0.2)
            if not self.h > 0:
                raise ValueError("the residuals give no bandwidth")
            self.grid = np.linspace(self.z.min() - KDE_PAD * self.h,
                                    self.z.max() + KDE_PAD * self.h, KDE_POINTS)
            dens = np.zeros(KDE_POINTS)
            for chunk in np.array_split(self.z, max(1, len(self.z) // 2000 + 1)):
                dens += np.exp(-0.5 * ((self.grid[:, None] - chunk[None, :]) / self.h) ** 2).sum(1)
            step = self.grid[1] - self.grid[0]
            self.dens = np.maximum(dens / (dens.sum() * step), FLOOR_L)

    def pdf(self, u: np.ndarray) -> np.ndarray:
        if self.z is None:
            return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
        return np.interp(u, self.grid, self.dens, left=FLOOR_L, right=FLOOR_L)

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.z is None:
            return rng.standard_normal(n)
        return rng.choice(self.z, n) + self.h * rng.standard_normal(n)

    def describe(self) -> dict:
        """n, centre, sd, robust sd (1.4826 MAD), kurtosis, share |z| > 2 and > 3, h."""
        if self.z is None:
            return {"n": np.nan, "centre": np.nan, "sd": 1.0, "robust_sd": np.nan,
                    "kurtosis": np.nan, "share_abs_gt_2": np.nan, "share_abs_gt_3": np.nan,
                    "bandwidth": np.nan}
        z = self.z
        med = float(np.median(z))
        var = float(z.var())
        return {"n": int(len(z)), "centre": med, "sd": float(z.std()),
                "robust_sd": float(1.4826 * np.median(np.abs(z - med))),
                "kurtosis": float(((z - z.mean()) ** 4).mean() / var ** 2) if var > 0 else np.nan,
                "share_abs_gt_2": float((np.abs(z) > 2).mean()),
                "share_abs_gt_3": float((np.abs(z) > 3).mean()), "bandwidth": float(self.h)}


# ==========================================================================
# error scales
# ==========================================================================
def pooled_scale(se) -> float:
    """One error scale for a set of variants: sqrt(mean(SE^2))."""
    se = np.asarray(se, float)
    return float(np.sqrt(np.mean(se * se)))


def robust_sd(x) -> float:
    """1.4826 x the median absolute deviation from the median."""
    x = np.asarray(x, float)
    return float(MAD_FACTOR * np.median(np.abs(x - np.median(x))))


def pava(v) -> tuple[np.ndarray, np.ndarray]:
    """The non-decreasing least-squares fit to v with equal weights (pool adjacent
    violators): (the fitted values, each value's block, numbered from 0)."""
    means, counts = [], []
    for x in map(float, v):
        means.append(x)
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            m, c = means.pop(), counts.pop()
            means[-1] = (means[-1] * counts[-1] + m * c) / (counts[-1] + c)
            counts[-1] += c
    return np.repeat(means, counts), np.repeat(np.arange(len(counts)), counts)


class ErrorScale:
    """sigma(SE), calibrated on a noise sample: residuals r (each a draw of a variant's
    error) with their SEs. The sample sorted by SE (a stable sort: the caller's order
    settles ties) and cut into equal-count bins as numpy.array_split cuts; per bin
    x_b = the median log SE and sigma_b = 1.4826 MAD of the residuals; sigma_b made
    non-decreasing in bin order (pool adjacent violators, equal weights); sigma(SE)
    linear in log SE between the points (x_b, sigma_b), constant beyond the first and
    the last."""

    def __init__(self, se, residuals, bins: int = SCALE_BINS):
        se, r = np.asarray(se, float), np.asarray(residuals, float)
        if len(se) != len(r) or len(se) < 2 * bins:
            raise ValueError("an error scale needs two or more residuals per bin")
        if not (np.all(np.isfinite(se)) and np.all(se > 0) and np.all(np.isfinite(r))):
            raise ValueError("an error scale needs positive SEs and finite residuals")
        order = np.argsort(se, kind="stable")
        self.bins = np.array_split(order, bins)
        self.n = np.array([len(b) for b in self.bins])
        self.x = np.array([np.median(np.log(se[b])) for b in self.bins])
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("the bins' median log SEs do not increase")
        self.sigma_raw = np.array([robust_sd(r[b]) for b in self.bins])
        self.sigma, self.block = pava(self.sigma_raw)
        self.se_min = np.array([se[b].min() for b in self.bins])
        self.se_median = np.array([np.median(se[b]) for b in self.bins])
        self.se_max = np.array([se[b].max() for b in self.bins])
        self.rms_se = np.array([pooled_scale(se[b]) for b in self.bins])

    def __call__(self, se) -> np.ndarray:
        return np.interp(np.log(np.asarray(se, float)), self.x, self.sigma)

    def largest_block(self) -> int:
        """The most bins pooled into one by the monotone fit."""
        return int(np.bincount(self.block).max())

    def outside(self, se) -> dict:
        """The shares of SEs below the first bin's x_b and above the last's."""
        ls = np.log(np.asarray(se, float))
        return {"below_first": float((ls < self.x[0]).mean()),
                "above_last": float((ls > self.x[-1]).mean())}

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"bin": np.arange(1, len(self.bins) + 1), "n": self.n,
                             "se_min": self.se_min, "se_median": self.se_median,
                             "se_max": self.se_max, "x_b": self.x,
                             "sigma_b_raw": self.sigma_raw, "sigma_b": self.sigma,
                             "block": self.block + 1, "rms_se": self.rms_se,
                             "sigma_b_over_rms_se": self.sigma / self.rms_se})


# ==========================================================================
# the estimate
# ==========================================================================
def grid(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    pad = GRID_PAD * float(np.median(s))
    return np.unique(np.concatenate([np.quantile(y, np.linspace(0, 1, GRID_QUANTILES)),
                                     np.linspace(y.min() - pad, y.max() + pad, GRID_EVEN)]))


def likelihood(y: np.ndarray, s: np.ndarray, g: np.ndarray, em: ErrorModel) -> np.ndarray:
    u = (y[:, None] - g[None, :]) / s[:, None]
    return np.maximum(em.pdf(u) / s[:, None], FLOOR_L)


def npmle(L: np.ndarray, iterations: int) -> np.ndarray:
    """EM for the mixing weights on the grid, from uniform weights."""
    n, k = L.shape
    w = np.full(k, 1.0 / k)
    for _ in range(iterations):
        m = L @ w
        w = w * (((1.0 / m) @ L) / n)
    return w


def remeasure(L: np.ndarray, w: np.ndarray, g: np.ndarray, s: np.ndarray, em: ErrorModel,
              rng: np.random.Generator, repeats: int) -> float:
    """The mean Spearman rho between two simulated measurements, over repeats."""
    P = L * w
    P /= P.sum(1, keepdims=True)
    cdf = np.cumsum(P, 1)
    n, last = len(s), len(g) - 1
    rhos = []
    for _ in range(repeats):
        u = rng.random(n)
        t = g[np.minimum((u[:, None] > cdf).sum(1), last)]
        a = t + s * em.draw(n, rng)
        b = t + s * em.draw(n, rng)
        rhos.append(spearman(a, b))
    return float(np.mean(rhos))


def fit(y: np.ndarray, s: np.ndarray, em: ErrorModel, iterations: int):
    g = grid(y, s)
    L = likelihood(y, s, g, em)
    return g, L, npmle(L, iterations)


def estimate(y: np.ndarray, s: np.ndarray, em: ErrorModel, rng: np.random.Generator,
             iterations: int = ITER_POINT, repeats: int = REPEAT_POINT) -> tuple[float, float]:
    """(ceiling, the mean Spearman rho between the two simulated measurements)."""
    g, L, w = fit(np.asarray(y, float), np.asarray(s, float), em, iterations)
    r = remeasure(L, w, g, np.asarray(s, float), em, rng, repeats)
    return float(np.sqrt(max(r, 0.0))), r


# ==========================================================================
# a cell: point, interval, floor
# ==========================================================================
def set_sha(ids) -> str:
    """sha256 of the sorted identifiers joined by newlines."""
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def cell_key(group: str, gene: str, stratum: str, pop: str, model: str, sha: str) -> tuple:
    return ("rank", group, gene, stratum, pop, model, sha)


def seeds(key: tuple) -> tuple[int, int]:
    from atlas.matched_ceilings import cell_seed
    return cell_seed(key), cell_seed(key + ("null",))


def resample_indices(seed: int, n: int, b: int = B) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def estimate_cell(ids, y, s, key: tuple, em: ErrorModel | None = None,
                  residuals: np.ndarray | None = None, b: int = B, nulls: int = NULLS,
                  predictors: dict[str, np.ndarray] | None = None,
                  pooled: bool = False) -> dict:
    """Point, bootstrap interval and floor of one set of variants.

    em is a fixed error model; residuals, instead, are per-variant residuals of the
    cell itself (its empirical model is built from them and rebuilt in each resample).
    s is the variants' error scale; with pooled, it is their SE and the scale is one
    value, pooled_scale of the cell's SEs for the point and the floor, of the resampled
    variants' in each resample. ids, y, s (and the residuals and each predictor's
    scores) are aligned; they are put in the order of the sorted ids first.
    predictors: name -> scores, for the realisation's interval on the same resamples.
    """
    t0 = time.time()
    ids = np.asarray(list(map(str, ids)))
    order = np.argsort(ids, kind="stable")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{key}: an identifier comes twice")
    y = np.asarray(y, float)[order]
    s = np.asarray(s, float)[order]
    n = len(y)
    se = s
    if pooled:
        s = np.full(n, pooled_scale(se))
    res = None if residuals is None else np.asarray(residuals, float)[order]
    if (em is None) == (res is None):
        raise ValueError("give a fixed error model or the cell's residuals")
    model = em if em is not None else ErrorModel(res)
    seed, null_seed = seeds(key)
    point, r = estimate(y, s, model, np.random.default_rng([seed, 0]))
    idx = resample_indices(seed, n, b)
    boot = np.empty(b)
    for i in range(b):
        j = idx[i]
        mb = model if res is None else ErrorModel(res[j])
        sj = np.full(n, pooled_scale(se[j])) if pooled else s[j]
        boot[i], _ = estimate(y[j], sj, mb, np.random.default_rng([seed, 1, i]),
                              ITER_RESAMPLE, REPEAT_RESAMPLE)
    null = np.empty(nulls)
    for i in range(nulls):
        rng = np.random.default_rng([null_seed, i])
        y0 = s * model.draw(n, rng)
        null[i], _ = estimate(y0, s, model, rng, ITER_RESAMPLE, REPEAT_RESAMPLE)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    floor = float(np.quantile(null, FLOOR_QUANTILE))
    out = {"n": int(n), "set_sha": key[-1][:16], "seed": int(seed), "null_seed": int(null_seed),
           "rank_reliability": r, "ceiling": point, "ceiling_lo": float(lo),
           "ceiling_hi": float(hi), "floor": floor, "estimable": bool(lo > floor),
           "bootstrap_B": int(b), "nulls": int(nulls), "seconds": round(time.time() - t0, 1),
           "boot": boot}
    if pooled:
        out["pooled_scale"] = float(s[0])
    if predictors:
        from atlas.matched_ceilings import spearman_rows
        out["realisation"] = {}
        for name, x in predictors.items():
            x = np.asarray(x, float)[order]
            rho = spearman_rows(x[idx], y[idx])
            with np.errstate(invalid="ignore", divide="ignore"):
                real = np.where(boot > 0, rho / boot, np.nan)
            ok = real[np.isfinite(real)]
            out["realisation"][name] = {
                "realisation_lo": float(np.quantile(ok, 0.025)) if len(ok) else np.nan,
                "realisation_hi": float(np.quantile(ok, 0.975)) if len(ok) else np.nan,
                "share_realisation_gt_1": float((ok > 1).mean()) if len(ok) else np.nan,
                "bootstrap_realisation_undefined": int(len(real) - len(ok))}
    return out


# ==========================================================================
# self-test
# ==========================================================================
SELFTEST_N = 8
TRUTH_REPEATS = 40
NEUTRAL_RESIDUALS = 1300
SIGNAL_TRUTH = 0.25
BIAS_TOL = 0.05
ERROR_TOL = 0.10
REAL_BIAS_TOL = 0.05
REAL_SHARE_TOL = 0.25
REAL_STRATA = ("splice_3_10", "splice_11_50")


def _sparse(p0: float, mu: float, sd: float):
    return lambda rng, n: np.where(rng.random(n) < p0, 0.0, rng.normal(mu, sd, n))


def _lognormal_se(scale: float, sd: float):
    return lambda rng, n: scale * np.exp(rng.normal(0, sd, n))


SCENARIOS = {   # name: (n, true values, SE)
    "sparse_15pct": (1500, _sparse(0.85, 1.0, 0.3), _lognormal_se(0.1, 0.5)),
    "sparse_3pct": (1500, _sparse(0.97, 1.0, 0.4), _lognormal_se(0.1, 0.5)),
    "continuous": (500, lambda rng, n: rng.normal(0, 1, n), _lognormal_se(0.35, 0.3)),
    "no_effect": (1500, lambda rng, n: np.zeros(n), _lognormal_se(0.1, 0.5)),
    "no_effect_n300": (300, lambda rng, n: np.zeros(n), _lognormal_se(0.1, 0.5)),
}


def gaussian_noise(rng, n):
    return rng.standard_normal(n)


def heavy_noise(rng, n):
    """95% N(0, 1), 5% N(0, 3.5^2)."""
    return np.where(rng.random(n) < 0.95, rng.standard_normal(n), 3.5 * rng.standard_normal(n))


NOISE = {"gaussian": gaussian_noise, "heavy_tailed": heavy_noise}


def true_ceiling(t: np.ndarray, s: np.ndarray, noise, rng, repeats: int = TRUTH_REPEATS) -> float:
    """sqrt of the mean Spearman rho between two measurements of known true values."""
    r = [spearman(t + s * noise(rng, len(t)), t + s * noise(rng, len(t))) for _ in range(repeats)]
    return float(np.sqrt(max(np.mean(r), 0.0)))


def _known_task(task: tuple) -> dict:
    noise_name, scen, rep = task
    from atlas.matched_ceilings import cell_seed
    rng = np.random.default_rng(cell_seed(("rank_selftest", noise_name, scen, rep)))
    n, tgen, segen = SCENARIOS[scen]
    noise = NOISE[noise_name]
    t, s = tgen(rng, n), segen(rng, n)
    y = t + s * noise(rng, n)
    em = (ErrorModel() if noise_name == "gaussian"
          else ErrorModel(heavy_noise(rng, NEUTRAL_RESIDUALS)))
    t0 = time.time()
    est, r = estimate(y, s, em, rng)
    sec = time.time() - t0
    truth = true_ceiling(t, s, noise, rng)
    return {"part": "known_truth", "error": noise_name, "model": em.name, "case": scen,
            "n": n, "rep": rep, "truth": truth, "estimate": est, "difference": est - truth,
            "seconds": round(sec, 2)}


def known_truth(workers: int = 1) -> pd.DataFrame:
    tasks = [(e, sc, rep) for e in NOISE for sc in SCENARIOS for rep in range(SELFTEST_N)]
    return pd.DataFrame(_map(_known_task, tasks, workers))


def _real_task(task: tuple) -> list[dict]:
    """Data sets drawn from the true-value distribution estimated on one real cell. The
    task carries the cell's own residuals (res, aligned with ids), or a fixed set of
    residuals (an eighth element: the gene's neutral residuals), or neither (Gaussian)."""
    gene, stratum, model_name, ids, y, s, res = task[:7]
    fixed = task[7] if len(task) > 7 else None
    from atlas.matched_ceilings import cell_seed
    sha = set_sha(ids)
    rng = np.random.default_rng(cell_seed(("rank_selftest_real", gene, stratum, model_name,
                                           sha)))
    order = np.argsort(np.asarray(ids, str), kind="stable")
    y, s = np.asarray(y, float)[order], np.asarray(s, float)[order]
    em = (ErrorModel(np.asarray(fixed, float)) if fixed is not None else
          ErrorModel() if res is None else ErrorModel(np.asarray(res, float)[order]))
    g, L, w = fit(y, s, em, ITER_POINT)
    p = w / w.sum()
    rows = []
    for rep in range(SELFTEST_N):
        t = rng.choice(g, len(y), p=p)
        ys = t + s * em.draw(len(y), rng)
        t0 = time.time()
        est, _ = estimate(ys, s, em, rng)
        sec = time.time() - t0
        truth = true_ceiling(t, s, lambda r, n: em.draw(n, r), rng)
        rows.append({"part": "real_cells", "error": model_name, "model": em.name,
                     "case": f"{gene} {stratum}", "gene": gene, "stratum": stratum,
                     "n": int(len(y)), "rep": rep, "truth": truth, "estimate": est,
                     "difference": est - truth, "seconds": round(sec, 2)})
    return rows


def _map(fn, tasks, workers):
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, tasks))
    return [fn(t) for t in tasks]


def real_cell_inputs(strata=REAL_STRATA) -> list[dict]:
    """The nine genes' cells (SNVs) with error information: gene, stratum, ids, y, and
    se (the SE genes' SEs) or d (BRCA1's replicate half-differences)."""
    from atlas.evaluate import classify_region
    from atlas.reliability import _stratum_mask
    m = population.load(population.SNV)
    aux = pd.read_parquet(paths.path("assay_aux"),
                          columns=["variant_id", "gene", "fit_rep1", "fit_rep2", "se",
                                   "se_status"])
    m = m.merge(aux, on=["variant_id", "gene"], how="left")
    cells = []
    for gene in ("BRCA1", "BARD1", "PALB2", "BAP1"):
        g = m[(m["gene"] == gene) & m["functional_pathogenicity"].notna()]
        for stratum in strata:
            c = g[_stratum_mask(g["region"], stratum)]
            if gene == "BRCA1":
                c = c.dropna(subset=["fit_rep1", "fit_rep2"])
                d = ((-c["fit_rep1"]) - (-c["fit_rep2"])).to_numpy(float) / 2
                cells.append({"gene": gene, "stratum": stratum, "ids": c["variant_id"].tolist(),
                              "y": c["functional_pathogenicity"].to_numpy(float), "se": None,
                              "d": d})
            else:
                c = c[c["se"].notna() & c["se_status"].eq("validated")]
                cells.append({"gene": gene, "stratum": stratum, "ids": c["variant_id"].tolist(),
                              "y": c["functional_pathogenicity"].to_numpy(float),
                              "se": c["se"].to_numpy(float), "d": None})
    ext = pd.read_parquet(paths.path("external_scores"),
                          columns=["gene", "variant_urn", "hgvs_c", "functional_pathogenicity",
                                   "standard_error"])
    ext["region"] = ext["hgvs_c"].map(classify_region)
    for gene in ("CTCF", "SFPQ", "RAD51D", "XRCC2", "SBDS"):
        g = ext[(ext["gene"] == gene)].dropna(subset=["functional_pathogenicity",
                                                      "standard_error"])
        for stratum in strata:
            c = g[_stratum_mask(g["region"], stratum)]
            cells.append({"gene": gene, "stratum": stratum, "ids": c["variant_urn"].tolist(),
                          "y": c["functional_pathogenicity"].to_numpy(float),
                          "se": c["standard_error"].to_numpy(float), "d": None})
    return cells


def own_cells(strata=REAL_STRATA) -> list[tuple]:
    """The self-test's real cells with their own error model: the SE genes Gaussian, the
    SE pooled within the cell (one value, pooled_scale); BRCA1 the half-differences of
    its two replicates within the cell (s = 1)."""
    tasks = []
    for c in real_cell_inputs(strata):
        if c["d"] is not None:
            tasks.append((c["gene"], c["stratum"], "own", c["ids"], c["y"],
                          np.ones(len(c["y"])), c["d"]))
        else:
            tasks.append((c["gene"], c["stratum"], "own", c["ids"], c["y"],
                          np.full(len(c["y"]), pooled_scale(c["se"])), None))
    return tasks


def real_cells(tasks: list[tuple], workers: int = 1) -> pd.DataFrame:
    return pd.DataFrame([r for rows in _map(_real_task, tasks, workers) for r in rows])


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["part", "error", "model", "case"]
    g = rows.groupby(keys, sort=False)
    out = g.agg(n=("n", "first"), datasets=("rep", "size"), truth_mean=("truth", "mean"),
                estimate_mean=("estimate", "mean"), estimate_sd=("estimate", "std"),
                mean_bias=("difference", "mean"),
                max_abs_error=("difference", lambda d: float(np.abs(d).max())),
                ).reset_index()
    known = out["part"] == "known_truth"
    signal = known & (out["truth_mean"] >= SIGNAL_TRUTH)
    out["signal"] = pd.Series(signal, dtype="boolean").where(known)
    out["passes"] = pd.Series((out["mean_bias"].abs() <= BIAS_TOL)
                              & (out["max_abs_error"] <= ERROR_TOL), dtype="boolean").where(signal)
    out["bias_above_0.05"] = pd.Series(out["mean_bias"].abs() > REAL_BIAS_TOL,
                                       dtype="boolean").where(~known)
    return out


def acceptance(summary: pd.DataFrame, expected_real: int = 36) -> dict:
    known = summary[(summary["part"] == "known_truth") & summary["signal"].fillna(False)]
    real = summary[summary["part"] == "real_cells"]
    over = int(real["bias_above_0.05"].fillna(False).sum())
    return {"known_truth_signal_cases": int(len(known)),
            "known_truth_signal_cases_passing": int(known["passes"].fillna(False).sum()),
            "known_truth_holds": bool(len(known) and known["passes"].fillna(False).all()),
            "known_truth_mean_bias_range": [float(known["mean_bias"].min()),
                                            float(known["mean_bias"].max())],
            "known_truth_max_abs_error": float(known["max_abs_error"].max()),
            "real_cells_done": int(len(real)), "real_cells_planned": expected_real,
            "real_cells_bias_above_0.05": over,
            "real_cells_limit": REAL_SHARE_TOL * expected_real,
            "real_cells_hold": bool(over <= REAL_SHARE_TOL * expected_real)}


def definition() -> dict:
    return {"grid": f"{GRID_QUANTILES} evenly spaced quantiles of y and {GRID_EVEN} evenly "
                    f"spaced points on [min y - {GRID_PAD:g} median s, max y + {GRID_PAD:g} "
                    "median s], duplicates removed",
            "likelihood_floor": FLOOR_L,
            "em_iterations": {"point": ITER_POINT, "resample": ITER_RESAMPLE},
            "remeasure_repeats": {"point": REPEAT_POINT, "resample": REPEAT_RESAMPLE},
            "bootstrap_B": B, "nulls": NULLS, "floor_quantile": FLOOR_QUANTILE,
            "kde": {"bandwidth": "Silverman: 0.9 min(sd, IQR/1.34) n^(-1/5)",
                    "density_points": KDE_POINTS, "density_range": "min z - 5h to max z + 5h",
                    "outside": FLOOR_L, "draw": "a residual with replacement plus h N(0, 1)"},
            "seed": "atlas.matched_ceilings.cell_seed(('rank', group, gene, stratum, "
                    "population, error model, sha256 of the sorted identifiers)); "
                    "the null's key ends in 'null'"}


def selftest_definition() -> dict:
    return {"known_truth": {
                "cases": {k: {"n": v[0]} for k, v in SCENARIOS.items()},
                "case_text": {"sparse_15pct": "85% 0, 15% N(1, 0.3^2); SE 0.1 exp(N(0, 0.5^2))",
                              "sparse_3pct": "97% 0, 3% N(1, 0.4^2); SE 0.1 exp(N(0, 0.5^2))",
                              "continuous": "N(0, 1); SE 0.35 exp(N(0, 0.3^2))",
                              "no_effect": "all 0; SE 0.1 exp(N(0, 0.5^2))",
                              "no_effect_n300": "all 0, n 300; SE as above"},
                "errors": {"gaussian": "N(0, 1), estimated with the Gaussian model",
                           "heavy_tailed": "95% N(0, 1), 5% N(0, 3.5^2), estimated with the "
                                           f"empirical model on {NEUTRAL_RESIDUALS} further "
                                           "draws of pure noise"},
                "data_sets": SELFTEST_N, "truth": f"sqrt of the mean Spearman rho between two "
                                                  f"simulated measurements of the known true "
                                                  f"values, {TRUTH_REPEATS} repeats",
                "acceptance": f"cases with signal (truth >= {SIGNAL_TRUTH}): mean bias within "
                              f"+-{BIAS_TOL}, no data set off by more than {ERROR_TOL}; "
                              "without effect the estimates are reported only",
                "seed": "cell_seed(('rank_selftest', error, case, data set))"},
            "real_cells": {
                "cells": "BRCA1, BARD1, PALB2, BAP1, CTCF, SFPQ, RAD51D, XRCC2, SBDS at 3-10 "
                         "and 11-50 bp, SNVs, with each error model",
                "error_models": "own: the SE genes Gaussian, the SE pooled within the cell "
                                "(sqrt mean SE^2, one value); BRCA1 its replicate "
                                "half-differences within the cell, s = 1. neutral: the "
                                "gene's neutral residuals over the calibrated scale, "
                                "z = r / sigma(SE), and s = sigma(SE) (BRCA1 z = r, s = 1). "
                                "See atlas.error_models",
                "procedure": "the true-value distribution estimated on the cell (1,000 EM "
                             f"iterations); {SELFTEST_N} data sets, true values drawn from it "
                             "and errors from the model with the cell's s; each estimated and "
                             "compared with the ceiling of its own true values",
                "acceptance": f"stop if more than {REAL_SHARE_TOL:.0%} of the 36 cell x model "
                              f"rows have a mean bias beyond {REAL_BIAS_TOL}",
                "seed": "cell_seed(('rank_selftest_real', gene, stratum, model, set sha256))"}}


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    return obj


SELFTEST_COLUMNS = ["part", "error", "model", "case", "n", "datasets", "truth_mean",
                    "estimate_mean", "estimate_sd", "mean_bias", "max_abs_error", "signal",
                    "passes", "bias_above_0.05"]


def write_selftest(rows: pd.DataFrame, out: Path, key: str, notes: dict) -> dict:
    summary = summarise(rows)[SELFTEST_COLUMNS]
    acc = acceptance(summary)
    rec = {"definition": definition(), "selftest": selftest_definition(), "acceptance": acc,
           **notes, "data_sets": rows.drop(columns=["seconds"]).to_dict("records")}
    summary.to_csv(out / paths.name(key), sep="\t", index=False)
    (out / paths.name(key + "_summary")).write_text(json.dumps(jsonable(rec), indent=1) + "\n")
    return acc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["selftest"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(paths.RESULTS))
    args = ap.parse_args(argv)
    from atlas import error_models          # the real cells' error models
    return error_models.selftest(Path(args.out), args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
