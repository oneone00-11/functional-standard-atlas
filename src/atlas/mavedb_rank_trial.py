"""A trial (A4b.5): the rank-scale ceiling of the MaveDB deposits, beside the variance-scale
ceiling of results/mavedb_ceilings_v2.tsv. Neither Figure 3 nor the ceilings table changes;
which scale they take is decided on these results.

Deposits: the rows of mavedb_ceilings_v2.tsv whose status is ok and whose score table is in
the local caches (data/raw/mavedb_audit_v2/scores/, data/raw/mavedb_ceilings_v2/); nothing is
fetched. The error column and its reading are the table's (method se: the column; ci: the
interval's half-width over 1.96; sd_as_se: the SD taken as the SE), on the rows with a
score and an error, as atlas.mavedb_ceilings reads them.

Per deposit: the own model of atlas.error_models, Gaussian with one pooled scale
sqrt(mean SE^2) for the deposit, point estimate only (atlas.rank_ceiling.estimate, 1,000
EM iterations, 20 repeats; the seed from the key (rank, mavedb, urn, deposit, all, own,
the set's sha256), stream [seed, 0]), beside v2's variance-scale ceiling.

Output: results/mavedb_rank_ceiling_trial_v1.tsv.

Usage (PYTHONPATH=src):  python -m atlas.mavedb_rank_trial [--workers N]
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import paths
from atlas import rank_ceiling as rc
from atlas.mavedb_ceilings import CI_Z, SCORE

CACHES = (paths.REPO / "data" / "raw" / "mavedb_audit_v2" / "scores",
          paths.REPO / "data" / "raw" / "mavedb_ceilings_v2")
COLUMNS = ["urn", "title", "stratum", "platform", "method", "error_column", "n", "pooled_se",
           "ceiling_v2", "ceiling_rank_own", "difference", "rank_reliability_own", "seed",
           "cache", "seconds"]


def cached(urn: str) -> Path | None:
    name = urn.replace(":", "_") + ".csv"
    for d in CACHES:
        if (d / name).exists():
            return d / name
    return None


def errors(df: pd.DataFrame, method: str, column: str) -> pd.Series:
    """The deposit's SE as mavedb_ceilings reads it."""
    if method == "ci":
        a, b = column.split("+")
        return (pd.to_numeric(df[b], errors="coerce") - pd.to_numeric(df[a], errors="coerce")
                ).abs() / (2 * CI_Z)
    if method in ("se", "sd_as_se"):
        return pd.to_numeric(df[column], errors="coerce")
    raise ValueError(f"method {method!r} has no error column")


def deposits() -> pd.DataFrame:
    t = pd.read_csv(paths.path("mavedb_ceilings"), sep="\t", float_precision="round_trip")
    t = t[t["status"] == "ok"].copy()
    t["cache"] = t["urn"].map(cached)
    return t[t["cache"].notna()].reset_index(drop=True)


def _task(r: dict) -> dict:
    from atlas.matched_ceilings import cell_seed
    t0 = time.time()
    df = pd.read_csv(r["cache"], low_memory=False)
    x = pd.DataFrame({"id": df["accession"].astype(str),
                      "y": pd.to_numeric(df[SCORE], errors="coerce"),
                      "se": errors(df, r["method"], r["error_column"])}).dropna()
    order = np.argsort(x["id"].to_numpy(), kind="stable")
    y, se = x["y"].to_numpy(float)[order], x["se"].to_numpy(float)[order]
    s = np.full(len(y), rc.pooled_scale(se))
    sha = rc.set_sha(x["id"])
    seed = cell_seed(rc.cell_key("mavedb", r["urn"], "deposit", "all", "own", sha))
    ceil, rel = rc.estimate(y, s, rc.ErrorModel(), np.random.default_rng([seed, 0]))
    return {"urn": r["urn"], "title": r["title"], "stratum": r["stratum"],
            "platform": r["platform"], "method": r["method"], "error_column": r["error_column"],
            "n": int(len(y)), "pooled_se": float(s[0]), "ceiling_v2": r["ceiling"],
            "ceiling_rank_own": ceil, "difference": ceil - r["ceiling"],
            "rank_reliability_own": rel, "seed": int(seed),
            "cache": Path(r["cache"]).relative_to(paths.REPO).as_posix(),
            "seconds": round(time.time() - t0, 1)}


def trial(workers: int = 1) -> pd.DataFrame:
    rows = deposits().to_dict("records")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(_task, rows))
    else:
        out = [_task(r) for r in rows]
    return pd.DataFrame(out, columns=COLUMNS)


def by_stratum(t: pd.DataFrame) -> pd.DataFrame:
    """The difference's median, quartiles and the number above 0.1 in absolute value."""
    g = t.groupby("stratum")["difference"]
    return pd.DataFrame({"deposits": g.size(), "median": g.median(), "q25": g.quantile(0.25),
                         "q75": g.quantile(0.75),
                         "abs_above_0.1": g.apply(lambda d: int((d.abs() > 0.1).sum()))})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(paths.RESULTS))
    args = ap.parse_args(argv)
    t = trial(args.workers)
    t.drop(columns=["seconds"]).to_csv(Path(args.out) / paths.name("mavedb_rank_ceiling_trial"),
                                       sep="\t", index=False)
    print(by_stratum(t).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"{len(t)} deposits, {t['seconds'].sum():.0f} s in all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
