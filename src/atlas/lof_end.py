"""The loss-of-function end (A4b, the second supplement): how many variants of each stratum
sit at the loss-of-function level, and how the neutral ceiling moves were the error to grow
toward that level. Descriptive: no acceptance, no stop.

results/lof_level_by_stratum_v1.tsv: the twelve genes x the seven strata (SNVs with a
functional score; the core genes' from the score matrix, the external genes' from their
deposits, coding variants included, y = -score): n; the variants at the loss-of-function
level (a score above lof_threshold, results/control_classes_v1.tsv), their number and
share; where there are 30 or more of them, their robust SD (1.4826 MAD about their
median) and its ratio to the gap. Beside the nonsense class's spread over the gap
(control_classes), it says whether a stratum's loss-of-function variants spread as the
nonsense variants do.

results/rank_ceiling_lof_sensitivity_v1.tsv: the nine genes with an error model x the
seven strata (SNVs, the layer cell's variant set, n >= 30), point estimates only:
  kappa*   max(1, kappa) of results/control_classes_v1.tsv;
  h(t)     1 + (kappa* - 1) clip((t - t_N) / (t_L - t_N), 0, 1): no growth at the normal
           level, kappa* at the loss-of-function level, linear between;
  L[i, j]  f((y_i - g_j) / (s_i h(g_j))) / (s_i h(g_j)), at least 1e-300, with s_i and f
           the neutral model's (SE genes s_i = sigma(SE_i), BRCA1 s_i = 1);
  remeasure  t_i drawn from its posterior; both measurements' error scale s_i h(t_i).
The grid, 1,000 EM iterations and 20 repeats are the neutral point estimate's; the seed
is the cell's neutral key with "lof" appended (stream [seed, 0]). With kappa* = 1 the
estimate is the neutral point estimate's computation, bit for bit, under the same seed.
Columns: gene, stratum, n, lof_share, kappa, the neutral point estimate (from the layer
tables), the estimate with the error grown, their difference.

Usage (PYTHONPATH=src):  python -m atlas.lof_end [--workers N]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import control_classes as cc
from atlas import error_models as em
from atlas import paths, population
from atlas import rank_ceiling as rc
from atlas.reliability import STRATA, _stratum_mask

MIN_N = 30
MIN_LOF = 30
LEVEL_COLUMNS = ["gene", "group", "stratum", "n", "n_lof_level", "lof_share", "sd_lof_level",
                 "sd_lof_level_over_gap", "sd_nonsense_over_gap"]
SENSITIVITY_COLUMNS = ["gene", "group", "stratum", "n", "set_sha", "lof_share", "kappa",
                       "kappa_star", "ceiling_neutral", "ceiling_neutral_lof_end",
                       "difference", "seed"]


# ==========================================================================
# the level by stratum
# ==========================================================================
def stratum_scores() -> pd.DataFrame:
    """The twelve genes' SNVs with a functional score: gene, group, region, y."""
    from atlas.external_reliability import EXTERNAL, deposit_population, load_deposit
    m = population.load(population.SNV, columns=["variant_id", "gene",
                                                 "functional_pathogenicity"])
    core = m[m["gene"].isin(cc.CORE) & m["functional_pathogenicity"].notna()]
    rows = [pd.DataFrame({"gene": core["gene"], "group": "core", "region": core["region"],
                          "y": core["functional_pathogenicity"]})]
    for urn, spec in EXTERNAL.items():
        if spec["gene"] not in cc.EXTERNAL:
            continue
        d = load_deposit(urn)
        d = d[deposit_population(d["hgvs_c"]) == population.SNV]
        y = -pd.to_numeric(d["score"], errors="coerce")
        ok = y.notna()
        rows.append(pd.DataFrame({"gene": spec["gene"], "group": "external",
                                  "region": d.loc[ok, "region"], "y": y[ok]}))
    return pd.concat(rows, ignore_index=True)


def level_table(scores: pd.DataFrame | None = None, classes: pd.DataFrame | None = None
                ) -> pd.DataFrame:
    scores = stratum_scores() if scores is None else scores
    lv = cc.levels() if classes is None else classes
    full = pd.read_csv(paths.path("control_classes"), sep="\t",
                       float_precision="round_trip").set_index("gene")
    rows = []
    for gene in cc.GENES:
        g = scores[scores["gene"] == gene]
        thr, gap = lv.loc[gene, "lof_threshold"], lv.loc[gene, "gap"]
        for stratum in STRATA:
            y = g.loc[_stratum_mask(g["region"], stratum), "y"].to_numpy(float)
            if not len(y):
                continue
            lof = y[y > thr]
            sd = rc.robust_sd(lof) if len(lof) >= MIN_LOF else np.nan
            rows.append({"gene": gene, "group": g["group"].iloc[0], "stratum": stratum,
                         "n": int(len(y)), "n_lof_level": int(len(lof)),
                         "lof_share": float(len(lof) / len(y)), "sd_lof_level": sd,
                         "sd_lof_level_over_gap": sd / gap,
                         "sd_nonsense_over_gap": float(full.loc[gene, "sd_nonsense_over_gap"])})
    return pd.DataFrame(rows, columns=LEVEL_COLUMNS)


# ==========================================================================
# the ceiling with the error grown toward the loss-of-function level
# ==========================================================================
def amplifier(kappa_star: float, t_n: float, t_l: float):
    """h(t) = 1 + (kappa* - 1) clip((t - t_N) / (t_L - t_N), 0, 1)."""
    def h(t):
        return 1.0 + (kappa_star - 1.0) * np.clip((np.asarray(t, float) - t_n) / (t_l - t_n),
                                                  0.0, 1.0)
    return h


def estimate_grown(y, s, model: rc.ErrorModel, h, rng: np.random.Generator,
                   iterations: int = rc.ITER_POINT, repeats: int = rc.REPEAT_POINT
                   ) -> tuple[float, float]:
    """atlas.rank_ceiling.estimate with the error scale s_i h(t): (ceiling, mean rho)."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    g = rc.grid(y, s)
    S = s[:, None] * h(g)[None, :]
    L = np.maximum(model.pdf((y[:, None] - g[None, :]) / S) / S, rc.FLOOR_L)
    w = rc.npmle(L, iterations)
    P = L * w
    P /= P.sum(1, keepdims=True)
    cdf = np.cumsum(P, 1)
    n, last = len(s), len(g) - 1
    rhos = []
    for _ in range(repeats):
        u = rng.random(n)
        t = g[np.minimum((u[:, None] > cdf).sum(1), last)]
        st = s * h(t)
        a = t + st * model.draw(n, rng)
        b = t + st * model.draw(n, rng)
        rhos.append(rc.spearman(a, b))
    r = float(np.mean(rhos))
    return float(np.sqrt(max(r, 0.0))), r


def _sensitivity_task(task: tuple) -> dict:
    head, ids, y, s, model, kappa_star, t_n, t_l = task
    from atlas.matched_ceilings import cell_seed
    ids = np.asarray(list(map(str, ids)))
    order = np.argsort(ids, kind="stable")
    y, s = np.asarray(y, float)[order], np.asarray(s, float)[order]
    sha = rc.set_sha(ids)
    key = rc.cell_key(head["group"], head["gene"], head["stratum"], population.SNV, "neutral",
                      sha) + ("lof",)
    seed = cell_seed(key)
    est, _ = estimate_grown(y, s, model, amplifier(kappa_star, t_n, t_l),
                            np.random.default_rng([seed, 0]))
    return head | {"set_sha": sha[:16], "ceiling_neutral_lof_end": est, "seed": int(seed)}


def sensitivity_tasks(layer_dir: Path = paths.RESULTS) -> list[tuple]:
    """The layer cells (SNVs, n >= 30) of the nine genes, with the neutral model."""
    from atlas import reliability as rl
    from atlas.external_reliability import deposit_sets
    lv = cc.levels()
    sets = {k: v for k, v in rl.layer_sets(population.load(None),
                                           pd.read_parquet(paths.path("assay_aux"))).items()}
    sets |= deposit_sets()
    layer = pd.concat([pd.read_csv(Path(layer_dir) / paths.name(k), sep="\t",
                                   float_precision="round_trip")
                       for k in ("reliability", "external_reliability")], ignore_index=True)
    layer = layer[(layer["population"] == population.SNV) & layer["ceiling_neutral"].notna()]
    tasks = []
    for r in layer.itertuples(index=False):
        if r.stratum not in STRATA or r.n < MIN_N:
            continue
        x = sets[(r.gene, population.SNV, r.stratum)]
        s = (np.ones(len(x["ids"])) if r.gene == "BRCA1"
             else em.error_scale(r.gene)(x["se"]))
        kappa = float(lv.loc[r.gene, "kappa"])
        head = {"gene": r.gene, "group": "external" if r.gene in cc.EXTERNAL else "core",
                "stratum": r.stratum, "n": int(r.n), "lof_share": float(r.lof_share),
                "kappa": kappa, "kappa_star": max(1.0, kappa),
                "ceiling_neutral": float(r.ceiling_neutral)}
        tasks.append((head, x["ids"], x["y"], s, em.neutral_model(r.gene), max(1.0, kappa),
                      float(lv.loc[r.gene, "t_N"]), float(lv.loc[r.gene, "t_L"])))
    return tasks


def sensitivity_table(tasks: list[tuple], workers: int = 1) -> pd.DataFrame:
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_sensitivity_task, tasks))
    else:
        rows = [_sensitivity_task(t) for t in tasks]
    t = pd.DataFrame(rows)
    t["difference"] = t["ceiling_neutral_lof_end"] - t["ceiling_neutral"]
    order = {g: i for i, g in enumerate(cc.GENES)}
    t = t.sort_values(["gene", "stratum"], key=lambda c: c.map(order) if c.name == "gene"
                      else c.map({s: i for i, s in enumerate(STRATA)}), kind="stable")
    return t[SENSITIVITY_COLUMNS].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(paths.RESULTS))
    ap.add_argument("--layer-dir", default=str(paths.RESULTS))
    args = ap.parse_args(argv)
    out = Path(args.out)
    levels = level_table()
    levels.to_csv(out / paths.name("lof_level_by_stratum"), sep="\t", index=False)
    sens = sensitivity_table(sensitivity_tasks(Path(args.layer_dir)), args.workers)
    sens.to_csv(out / paths.name("rank_ceiling_lof_sensitivity"), sep="\t", index=False)
    print(levels[levels["stratum"].isin(["splice_1_2", "splice_3_10"])].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print(sens.drop(columns=["set_sha", "seed"]).to_string(index=False,
                                                           float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
