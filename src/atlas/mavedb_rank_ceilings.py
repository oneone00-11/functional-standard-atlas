"""The rank-scale ceiling of the MaveDB deposits (A4c.3), beside the variance-scale ceiling
of results/mavedb_ceilings_v2.tsv. Figure 3's ceilings are on the variance scale and the
text's on the rank scale; every deposit with a ceiling gets a rank-scale value, the two
scales side by side. The survey's counts, the audit, the ceilings table and the trial
(atlas.mavedb_rank_trial, results/mavedb_rank_ceiling_trial_v1.tsv) are not changed.

Deposits: the 820 rows of mavedb_ceilings_v2.tsv whose status is ok (methods sd_as_se,
se, ci and replicates). Score tables: the local caches where they hold one
(data/raw/mavedb_audit_v2/scores/, data/raw/mavedb_ceilings_v2/), else fetched with
atlas.mavedb_ceilings.fetch_tables into data/raw/mavedb_rank_ceilings_v1/ (not tracked;
the survey's pause between requests; the time fetched and the bytes in its fetch_log.json).
Each table's sha256 is recorded.

Per deposit:
  1 the variance-scale ceiling computed again from the table by v2's method and error
    column (atlas.mavedb_ceilings: errors 1 - mean(SE^2) / var(score); replicates the mean
    pairwise Spearman of the columns raised by Spearman-Brown); reproduces_v2: the same n
    and a ceiling within 1e-9 of v2's;
  2 each variant's SE: se and sd_as_se the column, ci the interval's half-width over 1.96
    (atlas.mavedb_rank_trial.errors); replicates the sample SD (ddof 1) of v2's replicate
    columns over sqrt(k), on the rows with a score and every replicate (with the largest
    difference between the score and the replicates' mean);
  3 the rank-scale ceiling as the trial computes it: one pooled scale s = sqrt(mean SE^2),
    Gaussian, atlas.rank_ceiling.estimate (1,000 EM iterations, 20 repeats); the key
    (rank, mavedb, urn, deposit, all, own, the set's sha256), the accession the identifier,
    stream [seed, 0];
  4 the floor floor_own: 100 data sets without effect, y0 = s e, each with 300 EM
    iterations and 10 repeats, their 95th percentile; the null seed from the key with
    "null" appended, data set j from stream [null seed, j], as atlas.rank_ceiling's
    estimate_cell draws its nulls (100 here: 500 would take a day over the deposits);
  5 above_floor: the point estimate above floor_own.
No bootstrap: a deposit gets its point estimate and the floor.

Outputs: results/mavedb_rank_ceilings_v1.tsv (one row per deposit) and .json (on the
deposits whose status is ok and which reproduce v2: the numbers by stratum x method, the
deposits not fetched and those not reproducing v2; per stratum, over the methods with an
error column (se, sd_as_se, ci), both ceilings' median and quartiles, the difference's,
the differences below -0.1, the deposits lower on the rank scale, those below 0.90 on
each scale, those not above the floor; the 22 replicate deposits one by one, whose v2
ceiling is already on the rank scale). If the trial's 70 deposits are not reproduced bit
for bit, both files go to <out>/_failed/ and the exit status is 1.

Usage (PYTHONPATH=src):  python -m atlas.mavedb_rank_ceilings fetch
                         python -m atlas.mavedb_rank_ceilings run [--workers N] [--store DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import paths
from atlas import rank_ceiling as rc
from atlas.mavedb_ceilings import (SCORE, classify_rel, fetch_tables,
                                   reliability_from_replicates)
from atlas.mavedb_rank_trial import CACHES as TRIAL_CACHES
from atlas.mavedb_rank_trial import errors

FETCHED = paths.REPO / "data" / "raw" / "mavedb_rank_ceilings_v1"
CACHES = TRIAL_CACHES + (FETCHED,)
NULLS = 100
REPRODUCE_TOL = 1e-9
LOW = 0.90
ERROR_METHODS = ("se", "sd_as_se", "ci")
COLUMNS = ["urn", "title", "stratum", "platform", "method", "error_column", "n", "n_v2",
           "n_variance_here", "table_source", "table_sha256", "fetched_utc", "ceiling_v2",
           "ceiling_variance_here", "reproduces_v2", "pooled_se", "rank_reliability_own",
           "ceiling_rank_own", "floor_own", "above_floor", "difference",
           "replicate_mean_max_abs_diff", "seed", "null_seed", "status", "note"]


# ==========================================================================
# the deposits and their tables
# ==========================================================================
def deposits() -> pd.DataFrame:
    t = pd.read_csv(paths.path("mavedb_ceilings"), sep="\t", float_precision="round_trip")
    return t[t["status"] == "ok"].reset_index(drop=True)


def table_path(urn: str) -> Path | None:
    name = urn.replace(":", "_") + ".csv"
    for d in CACHES:
        if (d / name).exists():
            return d / name
    return None


def fetch_log(d: Path) -> dict:
    # the audit keeps its log beside scores/
    p = (d.parent if d.name == "scores" else d) / "fetch_log.json"
    return json.loads(p.read_text()) if p.exists() else {}


def fetch(urns: list[str] | None = None) -> dict:
    """The tables no cache holds, fetched into FETCHED."""
    urns = deposits()["urn"].tolist() if urns is None else urns
    missing = [u for u in urns if table_path(u) is None]
    print(f"[fetch] {len(urns)} deposits, {len(missing)} to fetch", flush=True)
    return fetch_tables(missing, FETCHED)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ==========================================================================
# one deposit
# ==========================================================================
def variance_ceiling(df: pd.DataFrame, method: str, column: str) -> dict:
    """v2's estimate computed again from the table (atlas.mavedb_ceilings.from_table)."""
    score = pd.to_numeric(df[SCORE], errors="coerce")
    var = float(score.var())
    if method == "replicates":
        rel, n = reliability_from_replicates(df, column.split(";"))
    else:
        sub = pd.concat([score.rename(SCORE), errors(df, method, column).rename("se")],
                        axis=1).dropna()
        n = len(sub)
        rel = (1 - float((sub["se"] ** 2).mean()) / float(sub[SCORE].var())
               if n >= 50 and var > 0 else float("nan"))
    status = classify_rel(rel, n, var)
    return {"n": int(n), "reliability": rel, "status": status,
            "ceiling": float(np.sqrt(rel)) if status == "ok" else float("nan")}


def replicate_errors(df: pd.DataFrame, column: str) -> tuple[pd.Series, float]:
    """SE = the replicates' sample SD (ddof 1) / sqrt(k) on the rows with a score and
    every replicate (NaN elsewhere), and the largest |score - the replicates' mean|."""
    cols = column.split(";")
    reps = df[cols].apply(pd.to_numeric, errors="coerce")
    score = pd.to_numeric(df[SCORE], errors="coerce")
    ok = reps.notna().all(axis=1) & score.notna()
    se = (reps.std(axis=1, ddof=1) / np.sqrt(len(cols))).where(ok)
    gap = (score - reps.mean(axis=1)).abs()[ok]
    return se, float(gap.max()) if len(gap) else float("nan")


def variant_set(df: pd.DataFrame, method: str, column: str) -> tuple[pd.DataFrame, float]:
    """(id, y, se) of the rows with a score and an error, as the trial builds it."""
    if method == "replicates":
        se, gap = replicate_errors(df, column)
    else:
        se, gap = errors(df, method, column), float("nan")
    x = pd.DataFrame({"id": df["accession"].astype(str),
                      "y": pd.to_numeric(df[SCORE], errors="coerce"), "se": se}).dropna()
    return x, gap


def rank_ceiling(urn: str, x: pd.DataFrame, nulls: int = NULLS) -> dict:
    """The point estimate (stream [seed, 0]) and the floor (streams [null seed, j])."""
    from atlas.matched_ceilings import cell_seed
    order = np.argsort(x["id"].to_numpy(), kind="stable")
    y, se = x["y"].to_numpy(float)[order], x["se"].to_numpy(float)[order]
    n = len(y)
    s = np.full(n, rc.pooled_scale(se))
    key = rc.cell_key("mavedb", urn, "deposit", "all", "own", rc.set_sha(x["id"]))
    seed, null_seed = rc.seeds(key)
    model = rc.ErrorModel()
    ceil, rel = rc.estimate(y, s, model, np.random.default_rng([seed, 0]))
    null = np.empty(nulls)
    for j in range(nulls):
        rng = np.random.default_rng([null_seed, j])
        y0 = s * model.draw(n, rng)
        null[j], _ = rc.estimate(y0, s, model, rng, rc.ITER_RESAMPLE, rc.REPEAT_RESAMPLE)
    floor = float(np.quantile(null, rc.FLOOR_QUANTILE))
    return {"n": int(n), "pooled_se": float(s[0]), "rank_reliability_own": rel,
            "ceiling_rank_own": ceil, "floor_own": floor, "above_floor": bool(ceil > floor),
            "seed": int(seed), "null_seed": int(null_seed), "null": null}


def one(r: dict, store: Path | None = None) -> dict:
    t0 = time.time()
    base = {k: r[k] for k in ("urn", "title", "stratum", "platform", "method", "error_column")}
    base |= {"ceiling_v2": r["ceiling"], "n_v2": int(r["n_variants"])}
    p = table_path(r["urn"])
    if p is None:
        return base | {"status": "fetch_failed", "note": "no score table in the caches"}
    src = {"table_source": p.parent.relative_to(paths.REPO).as_posix(),
           "table_sha256": sha256_file(p),
           "fetched_utc": fetch_log(p.parent).get(r["urn"], {}).get("fetched_utc")}
    df = pd.read_csv(p, low_memory=False)
    cols = r["error_column"].replace("+", ";").split(";")
    missing = [c for c in [SCORE, "accession"] + cols if c not in df.columns]
    if missing:
        return base | src | {"status": "column_missing", "note": "missing: " + ";".join(missing)}
    var = variance_ceiling(df, r["method"], r["error_column"])
    x, gap = variant_set(df, r["method"], r["error_column"])
    out = base | src | {"n_variance_here": var["n"], "ceiling_variance_here": var["ceiling"],
                        "reproduces_v2": bool(var["n"] == r["n_variants"]
                                              and abs(var["ceiling"] - r["ceiling"])
                                              <= REPRODUCE_TOL),
                        "replicate_mean_max_abs_diff": gap}
    if len(x) < 50:
        return out | {"n": int(len(x)), "status": "too_few_rows",
                      "note": "fewer than 50 rows with a score and an error"}
    rk = rank_ceiling(r["urn"], x)
    null = rk.pop("null")
    out |= rk | {"difference": rk["ceiling_rank_own"] - var["ceiling"], "status": "ok",
                 "note": {"sd_as_se": "the SD taken as the SE",
                          "ci": "SE = the 95% interval's half-width over 1.96",
                          "replicates": "SE = the replicates' SD (ddof 1) / sqrt(k)"}.get(
                              r["method"], ""), "seconds": round(time.time() - t0, 1)}
    if store is not None:
        name = r["urn"].replace(":", "_")
        np.save(Path(store) / f"{name}_null.npy", null)
        tmp = Path(store) / f"{name}.pkl.tmp{os.getpid()}"
        tmp.write_bytes(pickle.dumps(out))
        os.replace(tmp, Path(store) / f"{name}.pkl")
    return out


def run(workers: int = 1, store: Path | None = None) -> pd.DataFrame:
    dep = deposits()
    rows = dep.to_dict("records")
    done = {}
    if store is not None:
        Path(store).mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(rows):
            p = Path(store) / (r["urn"].replace(":", "_") + ".pkl")
            if p.exists():
                done[i] = pickle.loads(p.read_bytes())
    todo = sorted((i for i in range(len(rows)) if i not in done),
                  key=lambda i: -float(rows[i]["n_variants"]))
    print(f"[mavedb] {len(rows)} deposits: {len(done)} in the store, {len(todo)} to compute "
          f"({workers} workers)", flush=True)
    t0 = time.time()
    if workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(one, rows[i], store): i for i in todo}
            for k, f in enumerate(as_completed(futures), 1):
                i = futures[f]
                done[i] = f.result()
                print(f"[mavedb] {k}/{len(todo)} {rows[i]['urn']} n={done[i].get('n')} "
                      f"{done[i]['status']} {time.time() - t0:.0f} s", flush=True)
    else:
        for k, i in enumerate(todo, 1):
            done[i] = one(rows[i], store)
    return pd.DataFrame([done[i] for i in range(len(rows))]).reindex(columns=COLUMNS
                                                                     + ["seconds"])


# ==========================================================================
# the summary
# ==========================================================================
def _q(v: pd.Series) -> dict:
    v = pd.to_numeric(v, errors="coerce").dropna()
    if not len(v):
        return {"n": 0, "median": None, "q25": None, "q75": None}
    return {"n": int(len(v)), "median": float(v.median()), "q25": float(v.quantile(0.25)),
            "q75": float(v.quantile(0.75))}


def summary(t: pd.DataFrame) -> dict:
    ok = t[(t["status"] == "ok") & t["reproduces_v2"].astype("boolean").fillna(False)]
    failed = t[t["status"] != "ok"]
    not_rep = t[(t["status"] == "ok") & ~t["reproduces_v2"].astype("boolean").fillna(False)]
    by = ok.groupby(["stratum", "method"]).size()
    per_stratum = {}
    for stratum, d in ok[ok["method"].isin(ERROR_METHODS)].groupby("stratum"):
        diff = d["difference"]
        per_stratum[stratum] = {
            "deposits": int(len(d)),
            "ceiling_variance": _q(d["ceiling_variance_here"]),
            "ceiling_rank_own": _q(d["ceiling_rank_own"]),
            "difference": _q(diff),
            "difference_below_minus_0.1": int((diff < -0.1).sum()),
            "rank_below_variance": int((d["ceiling_rank_own"] < d["ceiling_variance_here"]).sum()),
            "variance_below_0.90": {"n": int((d["ceiling_variance_here"] < LOW).sum()),
                                    "share": float((d["ceiling_variance_here"] < LOW).mean())},
            "rank_below_0.90": {"n": int((d["ceiling_rank_own"] < LOW).sum()),
                                "share": float((d["ceiling_rank_own"] < LOW).mean())},
            "not_above_the_floor": int((~d["above_floor"].astype(bool)).sum())}
    rep = ok[ok["method"] == "replicates"]
    return {
        "deposits": int(len(t)), "used": int(len(ok)),
        "by_stratum_and_method": [{"stratum": s, "method": m, "deposits": int(v)}
                                  for (s, m), v in by.items()],
        "not_computed": {"n": int(len(failed)),
                         "deposits": failed[["urn", "status", "note"]].to_dict("records")},
        "not_reproducing_v2": {"n": int(len(not_rep)),
                               "deposits": not_rep[["urn", "method", "n_v2", "n_variance_here",
                                                    "ceiling_v2", "ceiling_variance_here"]
                                                   ].to_dict("records")},
        "by_stratum_error_methods": per_stratum,
        "not_above_the_floor": ok[~ok["above_floor"].astype(bool)][
            ["urn", "title", "stratum", "method", "n", "ceiling_variance_here",
             "ceiling_rank_own", "floor_own"]].to_dict("records"),
        "replicates": rep[["urn", "title", "stratum", "error_column", "n", "ceiling_v2",
                           "ceiling_rank_own", "difference", "floor_own", "above_floor",
                           "replicate_mean_max_abs_diff"]].to_dict("records"),
    }


def trial_check(t: pd.DataFrame) -> dict:
    """The trial's 70 deposits: n, pooled_se and ceiling_rank_own bit for bit."""
    tr = pd.read_csv(paths.path("mavedb_rank_ceiling_trial"), sep="\t",
                     float_precision="round_trip").set_index("urn")
    here = t.set_index("urn").loc[tr.index]
    differ = {}
    for c in ("n", "pooled_se", "ceiling_rank_own"):
        a, b = here[c].to_numpy(float), tr[c].to_numpy(float)
        bad = ~((a.view(np.uint64) == b.view(np.uint64)) | (np.isnan(a) & np.isnan(b)))
        differ[c] = tr.index[bad].tolist()
    return {"deposits": int(len(tr)), "differing": differ,
            "reproduce_v2": int(here["reproduces_v2"].astype("boolean").fillna(False).sum()),
            "holds": not any(differ.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "run"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default=str(paths.RESULTS))
    args = ap.parse_args(argv)
    if args.step == "fetch":
        log = fetch()
        bad = {u: v for u, v in log.items() if "error" in v}
        print(f"[fetch] {len(log)} in the log, {len(bad)} failed", flush=True)
        return 0
    t0 = time.time()
    t = run(args.workers, Path(args.store) if args.store else None)
    trial = trial_check(t)
    new = FETCHED / "fetch_log.json"
    log = json.loads(new.read_text()) if new.exists() else {}
    times = sorted(v["fetched_utc"] for v in log.values() if "error" not in v)
    tables = {r["urn"]: {"source": r["table_source"], "sha256": r["table_sha256"],
                         "fetched_utc": r["fetched_utc"]}
              for r in t.to_dict("records") if isinstance(r["table_source"], str)}
    t["table_sha256"] = t["table_sha256"].str[:16]              # the table: the first 16
    rec = {"definition": {
               "deposits": "mavedb_ceilings_v2.tsv, status ok",
               "variance_scale": "computed again by v2's method and error column; "
                                 f"reproduces_v2: the same n and within {REPRODUCE_TOL:g}",
               "se": {"se": "the column", "sd_as_se": "the SD taken as the SE",
                      "ci": "the 95% interval's half-width over 1.96 (1.959963984540054)",
                      "replicates": "the replicates' sample SD (ddof 1) / sqrt(k), on the rows "
                                    "with a score and every replicate"},
               "rank_scale": "one pooled scale sqrt(mean SE^2), Gaussian; "
                             f"{rc.ITER_POINT} EM iterations, {rc.REPEAT_POINT} repeats; key "
                             "(rank, mavedb, urn, deposit, all, own, sha256 of the sorted "
                             "accessions), stream [seed, 0]",
               "floor": f"{NULLS} data sets y0 = s e, {rc.ITER_RESAMPLE} EM iterations and "
                        f"{rc.REPEAT_RESAMPLE} repeats each, the 95th percentile; the null "
                        "seed from the key with 'null' appended, stream [null seed, j]",
               "no_bootstrap": "point estimate and floor only",
               "summaries": "status ok and reproduces_v2"},
           "fetch": {"cache": FETCHED.relative_to(paths.REPO).as_posix(),
                     "fetched": len(times), "failed": sorted(u for u, v in log.items()
                                                             if "error" in v),
                     "first_utc": times[0] if times else None,
                     "last_utc": times[-1] if times else None,
                     "from_the_caches": int(t["table_source"].isin(
                         [d.relative_to(paths.REPO).as_posix() for d in TRIAL_CACHES]).sum())},
           "checks": {"trial_70": trial},
           "summary": summary(t),
           "tables": tables}
    out = Path(args.out) if trial["holds"] else Path(args.out) / "_failed"   # a failed check:
    out.mkdir(parents=True, exist_ok=True)                                  # not delivered
    t.drop(columns=["seconds"]).to_csv(out / paths.name("mavedb_rank_ceilings"), sep="\t",
                                       index=False)
    (out / paths.name("mavedb_rank_ceilings_summary")).write_text(
        json.dumps(rc.jsonable(rec), indent=1) + "\n")
    s = rec["summary"]
    print(json.dumps(rc.jsonable({"trial_70": trial, "used": s["used"],
                                  "not_computed": s["not_computed"]["n"],
                                  "not_reproducing_v2": s["not_reproducing_v2"]["n"]}), indent=1))
    secs = pd.to_numeric(t["seconds"], errors="coerce")
    print(f"[mavedb] {len(t)} deposits, {args.workers} processes, {time.time() - t0:.1f} s "
          f"(per deposit with a rank-scale ceiling, {secs.notna().sum()} of them: "
          f"{secs.sum():.0f} s summed, the largest {secs.max():.0f} s)", flush=True)
    return 0 if trial["holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
