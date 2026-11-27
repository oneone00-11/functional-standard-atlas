"""The neutral model's sensitivity (A4c.2): how far the neutral ceiling moves under other
choices for its three settings. The main model is not changed (atlas.error_models: the
neutral class is the synonymous SNVs with spliceai_ds below 0.1, the SE in five bins, no
growth of the error toward the loss-of-function level); the alternatives appear in this
table only.

Cells: those of results/rank_ceiling_lof_sensitivity_v1.tsv: the nine genes with an error
model x the seven strata, SNVs, the layer cell's variant set, n >= 30 (57 cells). Point
estimates only, as the neutral point estimate (atlas.rank_ceiling.estimate: the grid,
1,000 EM iterations, 20 repeats); no bootstrap and no nulls.

Eight estimates per cell:
  base        the neutral class synonymous with spliceai_ds < 0.1; five SE bins; no growth
  bins_3      as base, three bins
  bins_8      as base, eight bins
  syn_all     every synonymous SNV (no SpliceAI filter); five bins
  ds_lt_0_2   synonymous with spliceai_ds < 0.2; five bins
  ds_lt_0_05  synonymous with spliceai_ds < 0.05; five bins
  ds_lt_0_02  synonymous with spliceai_ds < 0.02; five bins
  lof_end     as base, the error grown toward the loss-of-function level by kappa*
              (h(t) as atlas.lof_end: kappa* = max(1, kappa), linear from t_N to t_L)
A neutral class is the gene's synonymous SNVs with a functional score and error
information (the core genes from the score matrix, the external genes from
results/external_coding_consequence_v1.parquet), sorted by identifier, r = y - its median;
with another class the scale sigma(SE) and the shape z are made again on it as the main
model makes them (atlas.rank_ceiling.ErrorScale: equal-count bins by SE, 1.4826 MAD,
non-decreasing, linear in log SE). base is the main model itself (atlas.error_models'
error_scale and neutral_model). BRCA1 has no SE: bins_3 and bins_8 are empty, the others
are computed with s = 1 and the class's residuals.

Seed: the cell's neutral key (rank, group, gene, stratum, snv, neutral, the set's sha256)
with "sensitivity" appended; every estimate of a row starts again from
numpy.random.default_rng([seed, 0]), so the differences within a row carry no noise
between seeds.

Checks: (1) base with the cell's own neutral seed (stream [seed, 0]) is the layer table's
ceiling_neutral, bit for bit; (2) lof_end with kappa* = 1 is base, bit for bit; (3) the
explicit construction with five bins and the threshold 0.1 is base, bit for bit.

Outputs: results/rank_ceiling_sensitivity_v1.tsv: gene, group, stratum, n, set_sha,
lof_share, neutral_in_range, ceiling_neutral, floor_neutral and estimable_neutral (the
layer tables', results/reliability_v4.tsv and results/external_reliability_v4.tsv), kappa,
kappa_star, the eight estimates, seven differences d_<estimate> = estimate - base, seed;
results/rank_ceiling_sensitivity_v1.json: the neutral classes' sizes at the five
thresholds, per stratum each difference's median, minimum and maximum over the cells that
enter a summary (estimable_neutral and neutral_in_range) and over all cells, the checks.
results/rank_ceiling_lof_sensitivity_v1.tsv is not changed and not updated. If a check
does not hold, both files go to <out>/_failed/ and the exit status is 1.

Usage (PYTHONPATH=src):  python -m atlas.neutral_sensitivity [--workers N]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import control_classes as cc
from atlas import error_models as em
from atlas import lof_end, paths, population
from atlas import rank_ceiling as rc
from atlas.reliability import STRATA

MIN_N = 30
# estimate -> (SpliceAI threshold of the neutral class, None for none; SE bins; growth)
SPEC = {"base": (0.1, 5, False), "bins_3": (0.1, 3, False), "bins_8": (0.1, 8, False),
        "syn_all": (None, 5, False), "ds_lt_0_2": (0.2, 5, False),
        "ds_lt_0_05": (0.05, 5, False), "ds_lt_0_02": (0.02, 5, False),
        "lof_end": (0.1, 5, True)}
ESTIMATES = list(SPEC)
ALTERNATIVES = ESTIMATES[1:]
THRESHOLDS = {"none": None, "0.2": 0.2, "0.1": 0.1, "0.05": 0.05, "0.02": 0.02}
COLUMNS = (["gene", "group", "stratum", "n", "set_sha", "lof_share", "neutral_in_range",
            "ceiling_neutral", "floor_neutral", "estimable_neutral", "kappa", "kappa_star"]
           + ESTIMATES + [f"d_{e}" for e in ALTERNATIVES] + ["seed"])


# ==========================================================================
# the neutral class and its model, written out
# ==========================================================================
def neutral_class(gene: str, threshold: float | None) -> pd.DataFrame:
    """The gene's synonymous SNVs with a functional score and error information, those with
    spliceai_ds below threshold (None: all), sorted by identifier: id, y, se, r."""
    if gene in em.EXTERNAL:
        x = em.external_coding()
        g = x[x["gene"].eq(gene) & x["consequence"].eq("synonymous")
              & x["standard_error"].notna() & x["functional_pathogenicity"].notna()]
        if threshold is not None:
            g = g[g["spliceai_ds"] < threshold]
        out = pd.DataFrame({"id": g["variant_urn"], "y": g["functional_pathogenicity"],
                            "se": g["standard_error"]})
    else:
        d = em.core_frame()
        g = d[d["gene"].eq(gene) & d["population"].eq(population.SNV)
              & d["consequence"].eq("synonymous") & d["has_error"]
              & d["functional_pathogenicity"].notna()]
        if threshold is not None:
            g = g[g["spliceai_ds"] < threshold]
        out = pd.DataFrame({"id": g["variant_id"], "y": g["functional_pathogenicity"],
                            "se": g["se"] if gene != "BRCA1" else np.nan})
    out = out.sort_values("id", kind="stable").reset_index(drop=True)
    out["r"] = out["y"] - out["y"].median()
    return out


def neutral_spec(gene: str, threshold: float | None, bins: int
                 ) -> tuple[rc.ErrorScale | None, rc.ErrorModel]:
    """(sigma(SE), the shape) made on the class: z = r / sigma(SE) (BRCA1: no scale, z = r)."""
    nv = neutral_class(gene, threshold)
    if gene == "BRCA1":
        return None, rc.ErrorModel((nv["r"] / 1.0).to_numpy(float), name="neutral")
    se = nv["se"].to_numpy(float)
    scale = rc.ErrorScale(se, nv["r"].to_numpy(float), bins=bins)
    return scale, rc.ErrorModel((nv["r"] / scale(se)).to_numpy(float), name="neutral")


def class_sizes() -> dict:
    return {g: {k: int(len(neutral_class(g, t))) for k, t in THRESHOLDS.items()}
            for g in em.GENES}


# ==========================================================================
# the cells
# ==========================================================================
def tasks(layer_dir: Path = paths.RESULTS) -> list[dict]:
    """The layer cells (SNVs, n >= 30) of the nine genes, as atlas.lof_end takes them, with
    every input of the eight estimates built here."""
    from atlas import reliability as rl
    from atlas.external_reliability import deposit_sets
    lv = cc.levels()
    sets = rl.layer_sets(population.load(None), pd.read_parquet(paths.path("assay_aux")))
    sets |= deposit_sets()
    layer = pd.concat([pd.read_csv(Path(layer_dir) / paths.name(k), sep="\t",
                                   float_precision="round_trip")
                       for k in ("reliability", "external_reliability")], ignore_index=True)
    layer = layer[(layer["population"] == population.SNV) & layer["ceiling_neutral"].notna()]
    specs = {}
    out = []
    for r in layer.itertuples(index=False):
        if r.stratum not in STRATA or r.n < MIN_N:
            continue
        gene = r.gene
        x = sets[(gene, population.SNV, r.stratum)]
        kappa = float(lv.loc[gene, "kappa"])
        inputs = {}
        for name, (thr, bins, grow) in SPEC.items():
            if (gene == "BRCA1" and name in ("bins_3", "bins_8")) or name == "lof_end":
                continue                           # lof_end: base's inputs, grown
            if (thr, bins) not in specs.setdefault(gene, {}):
                specs[gene][(thr, bins)] = neutral_spec(gene, thr, bins)
            scale, model = specs[gene][(thr, bins)]
            s = np.ones(len(x["ids"])) if scale is None else scale(x["se"])
            inputs[name] = {"s": s, "model": model, "grow": grow}
        main_s = (np.ones(len(x["ids"])) if gene == "BRCA1"
                  else em.error_scale(gene)(x["se"]))
        out.append({"head": {"gene": gene, "group": "external" if gene in cc.EXTERNAL else "core",
                             "stratum": r.stratum, "n": int(r.n), "lof_share": float(r.lof_share),
                             "neutral_in_range": bool(r.neutral_in_range),
                             "ceiling_neutral": float(r.ceiling_neutral),
                             "floor_neutral": float(r.floor_neutral),
                             "estimable_neutral": bool(r.estimable_neutral),
                             "kappa": kappa, "kappa_star": max(1.0, kappa)},
                    "ids": x["ids"], "y": x["y"], "inputs": inputs,
                    "main": {"s": main_s, "model": em.neutral_model(gene)},
                    "levels": (float(lv.loc[gene, "t_N"]), float(lv.loc[gene, "t_L"])),
                    "seed_neutral": float(r.seed_neutral)})
    return out


def _estimate(y, s, model, rng, grow=None) -> float:
    if grow is None:
        return rc.estimate(y, s, model, rng)[0]
    return lof_end.estimate_grown(y, s, model, grow, rng)[0]


def run(task: dict) -> dict:
    from atlas.matched_ceilings import cell_seed
    head = task["head"]
    ids = np.asarray(list(map(str, task["ids"])))
    order = np.argsort(ids, kind="stable")
    y = np.asarray(task["y"], float)[order]
    sha = rc.set_sha(ids)
    key = rc.cell_key(head["group"], head["gene"], head["stratum"], population.SNV, "neutral",
                      sha)
    seed = cell_seed(key + ("sensitivity",))
    seed_neutral = cell_seed(key)
    t_n, t_l = task["levels"]
    out = dict(head) | {"set_sha": sha[:16], "seed": int(seed)}
    main_s = np.asarray(task["main"]["s"], float)[order]
    main_model = task["main"]["model"]
    out["base"] = _estimate(y, main_s, main_model, np.random.default_rng([seed, 0]))
    out["lof_end"] = _estimate(y, main_s, main_model, np.random.default_rng([seed, 0]),
                               lof_end.amplifier(head["kappa_star"], t_n, t_l))
    for name, x in task["inputs"].items():
        if name in ("base", "lof_end"):
            continue
        s = np.asarray(x["s"], float)[order]
        out[name] = _estimate(y, s, x["model"], np.random.default_rng([seed, 0]))
    # the checks
    ex = task["inputs"]["base"]
    ex_s = np.asarray(ex["s"], float)[order]
    out["check_seed_neutral"] = int(seed_neutral)
    out["check_base_own_seed"] = _estimate(y, main_s, main_model,
                                           np.random.default_rng([seed_neutral, 0]))
    out["check_lof_end_kappa_1"] = _estimate(y, main_s, main_model,
                                             np.random.default_rng([seed, 0]),
                                             lof_end.amplifier(1.0, t_n, t_l))
    out["check_explicit"] = _estimate(y, ex_s, ex["model"], np.random.default_rng([seed, 0]))
    out["check_explicit_inputs_same"] = bool(
        np.array_equal(ex_s.view(np.uint64), main_s.view(np.uint64))
        and np.array_equal(ex["model"].z.view(np.uint64), main_model.z.view(np.uint64)))
    return out


def _bits(a, b) -> bool:
    a, b = np.float64(a), np.float64(b)
    return bool(a.view(np.uint64) == b.view(np.uint64))


def table(rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    t = pd.DataFrame(rows)
    for e in ALTERNATIVES:
        if e not in t:
            t[e] = np.nan
        t[f"d_{e}"] = t[e] - t["base"]
    checks = {
        "base_with_the_own_neutral_seed_is_the_layer_ceiling": {
            "cells": int(len(t)),
            "differing": [f"{r.gene} {r.stratum}" for r in t.itertuples()
                          if not _bits(r.check_base_own_seed, r.ceiling_neutral)],
            "seeds_differing": [f"{r.gene} {r.stratum}" for r in t.itertuples()     # as the
                                if float(r.check_seed_neutral) != r.seed_neutral_layer]},  # TSV
        "lof_end_with_kappa_star_1_is_base": {
            "cells": int(len(t)),
            "differing": [f"{r.gene} {r.stratum}" for r in t.itertuples()
                          if not _bits(r.check_lof_end_kappa_1, r.base)]},
        "explicit_five_bins_threshold_0.1_is_base": {
            "cells": int(len(t)),
            "differing": [f"{r.gene} {r.stratum}" for r in t.itertuples()
                          if not _bits(r.check_explicit, r.base)],
            "inputs_differing": [f"{r.gene} {r.stratum}" for r in t.itertuples()
                                 if not r.check_explicit_inputs_same]}}
    for c in checks.values():
        c["holds"] = not any(v for k, v in c.items() if k.endswith("differing"))
    order = {g: i for i, g in enumerate(cc.GENES)}
    t = t.sort_values(["gene", "stratum"], kind="stable",
                      key=lambda c: c.map(order) if c.name == "gene"
                      else c.map({s: i for i, s in enumerate(STRATA)}))
    return t[COLUMNS].reset_index(drop=True), checks


def _span(v: pd.Series) -> dict:
    v = pd.to_numeric(v, errors="coerce").dropna()
    if not len(v):
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": int(len(v)), "median": float(v.median()), "min": float(v.min()),
            "max": float(v.max())}


def differences(t: pd.DataFrame) -> list[dict]:
    out = []
    for stratum in STRATA:
        d = t[t["stratum"] == stratum]
        if len(d):
            out.append({"stratum": stratum, "cells": int(len(d)), "genes": d["gene"].tolist(),
                        **{f"d_{e}": _span(d[f"d_{e}"]) for e in ALTERNATIVES}})
    return out


def summary(t: pd.DataFrame, checks: dict, sizes: dict) -> dict:
    enter = t["estimable_neutral"].astype(bool) & t["neutral_in_range"].astype(bool)
    return {"definition": {
                "estimates": {k: {"spliceai_below": v[0], "se_bins": v[1],
                                  "lof_growth": v[2]} for k, v in SPEC.items()},
                "estimator": f"atlas.rank_ceiling.estimate: {rc.ITER_POINT} EM iterations, "
                             f"{rc.REPEAT_POINT} repeats; point estimates only",
                "seed": "the cell's neutral key with 'sensitivity' appended; every estimate "
                        "of a row from numpy.random.default_rng([seed, 0])",
                "brca1": "no SE: bins_3 and bins_8 empty; s = 1",
                "layer_columns": [paths.name("reliability"), paths.name("external_reliability")],
                "enters_a_summary": "estimable_neutral and neutral_in_range"},
            "neutral_class_sizes": sizes,
            "differences_cells_entering": differences(t[enter]),
            "differences_all_cells": differences(t),
            "cells": {"all": int(len(t)), "entering": int(enter.sum())},
            "checks": checks}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(paths.RESULTS))
    ap.add_argument("--layer-dir", default=str(paths.RESULTS))
    args = ap.parse_args(argv)
    t0 = time.time()
    work = tasks(Path(args.layer_dir))
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(run, work))
    else:
        rows = [run(w) for w in work]
    for r, w in zip(rows, work):
        r["seed_neutral_layer"] = w["seed_neutral"]
    t, checks = table(rows)
    rec = summary(t, checks, class_sizes())
    holds = all(c["holds"] for c in checks.values())
    out = Path(args.out) if holds else Path(args.out) / "_failed"     # a failed check: not
    out.mkdir(parents=True, exist_ok=True)                             # where it is delivered
    t.to_csv(out / paths.name("rank_ceiling_sensitivity"), sep="\t", index=False)
    (out / paths.name("rank_ceiling_sensitivity_summary")).write_text(
        json.dumps(rc.jsonable(rec), indent=1) + "\n")
    print(json.dumps(rc.jsonable(checks), indent=1))
    print(t.drop(columns=["set_sha", "seed"]).to_string(index=False,
                                                        float_format=lambda v: f"{v:.3f}"))
    print(f"[sensitivity] {len(t)} cells, {args.workers} processes, "
          f"{time.time() - t0:.1f} s; checks {'hold' if holds else 'FAIL'}")
    return 0 if holds else 1


if __name__ == "__main__":
    raise SystemExit(main())
