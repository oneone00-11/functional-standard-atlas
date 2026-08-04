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

Usage (PYTHONPATH=src):  python -m atlas.simulate_attenuation --out results/
"""

from __future__ import annotations

import argparse
import json
import math
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=str(RESULTS / "score_matrix_atlas_v2.parquet"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args(argv)

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
