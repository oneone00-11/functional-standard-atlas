"""Pangolin on a table of variants, by score_fullprec.py's own functions, one thread per
process.

score_fullprec.py stays byte for byte the file that the core genes and the companion
release's external columns were scored with; its sha256 is checked against
--expect-sha256 when given and recorded either way. This driver loads it and uses its
setup (the twelve models), build_fullprec, aggregate, Args and DISTANCE (50; masking
"True", Pangolin's default) unchanged, with the reference and gene database given by
--reference and --gtf-db (defaults: data/refs/grch38_external.fa and
grch38_external.gtf.db, built by models/pangolin/build_external_refs.py).

One thread per process: the run requires OMP_NUM_THREADS = MKL_NUM_THREADS = 1 and
calls torch.set_num_threads(1). The companion release found that on one thread the
scorer reproduces its anchor variants bit for bit and on 2, 4, 6 or 8 threads it does
not. Speed comes from workers instead: --worker k --workers n scores the rows k, k + n,
... of --input, each worker appending to a cache of its own (variant_id, repr(value),
scored_utc) so that a stopped worker resumes; a value does not depend on n. A variant
the scorer cannot score, or raises on, is NaN, as in the atlas's run, and counted.
--collect merges the workers' caches into --output (variant_id, pangolin_score,
scored_utc) with <output>.environment.json beside it: interpreter, python, platform and
pip freeze read after the run (CONVENTIONS.md rule 3), the scorer's, reference's and
database's sha256, the configuration, and each worker's record.

Run with the Pangolin environment, from the repository root, one process per worker:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 <pangolin venv>/bin/python \\
        models/pangolin/score_external.py --input data/raw/external_v1/score_input.parquet \\
        --cache-dir models/pangolin/cache/external_v1 --worker 0 --workers 6
and then, once every worker has finished:
    <pangolin venv>/bin/python models/pangolin/score_external.py --collect \\
        --input data/raw/external_v1/score_input.parquet \\
        --cache-dir models/pangolin/cache/external_v1 \\
        --output models/pangolin/cache/external_v1/pangolin.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
REPO = MODEL_DIR.parents[1]
SCORER = MODEL_DIR / "score_fullprec.py"
REFERENCE = REPO / "data" / "refs" / "grch38_external.fa"
GTF_DB = REPO / "data" / "refs" / "grch38_external.gtf.db"
COLUMN = "pangolin_score"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_cache(path: Path) -> dict[str, tuple[float, str]]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines()[1:]:
            if line:
                vid, val, when = line.split("\t")
                out[vid] = (float(val), when)
    return out


def append_cache(path: Path, vid: str, value: float) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("variant_id\tvalue\tscored_utc\n")
    with open(path, "a") as f:
        f.write(f"{vid}\t{float(value)!r}\t{utc()}\n")


def worker_rows(df, worker: int, workers: int):
    """The rows a worker scores: positions worker, worker + workers, ..."""
    if not 0 <= worker < workers:
        raise ValueError(f"worker {worker} of {workers}")
    return df.iloc[worker::workers]


def load_scorer(path: Path = SCORER):
    spec = importlib.util.spec_from_file_location("pangolin_fullprec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _freeze() -> list[str]:
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                         capture_output=True, text=True).stdout.split("\n")
    return [x for x in out if x]


def score(args, df) -> int:
    if (os.environ.get("OMP_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS")) != ("1", "1"):
        raise SystemExit("set OMP_NUM_THREADS=1 MKL_NUM_THREADS=1: one thread per process")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import gffutils
    import pangolin.pangolin as PP
    pf = load_scorer()
    rows = worker_rows(df, args.worker, args.workers)
    cache = Path(args.cache_dir) / f"w{args.worker}_of_{args.workers}.tsv"
    done = read_cache(cache)
    todo = rows[~rows["variant_id"].isin(done)]
    print(f"[pangolin w{args.worker}/{args.workers}] {len(rows)} variants, {len(done)} cached, "
          f"{len(todo)} to score; torch threads {torch.get_num_threads()}", flush=True)
    _, models = pf.setup(PP)
    gtf = gffutils.FeatureDB(str(args.gtf_db))
    a = pf.Args(pf.DISTANCE)
    a.reference_file = str(args.reference)
    fullprec = pf.build_fullprec(PP)
    start, t0, errors, unscorable = utc(), time.time(), 0, 0
    for i, r in enumerate(todo.itertuples(index=False), 1):
        try:
            sc = fullprec(0, str(r.chrom), int(r.pos), r.ref, r.alt, gtf, models, a)
            if sc == -1:
                value, unscorable = np.nan, unscorable + 1
            else:
                value = pf.aggregate(sc)
        except Exception:                                          # noqa: BLE001
            value, errors = math.nan, errors + 1
        append_cache(cache, r.variant_id, value)
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  w{args.worker}: {i}/{len(todo)}  {el / 60:.1f} min, {el / i:.2f} s each",
                  flush=True)
    rec = {"worker": args.worker, "workers": args.workers, "rows": int(len(rows)),
           "torch_threads": torch.get_num_threads(),
           "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
           "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
           "run_started_utc": start, "run_ended_utc": utc(),
           "seconds_this_run": round(time.time() - t0, 1), "n_scored_this_run": int(len(todo)),
           "n_unscorable_this_run": unscorable, "n_exceptions_this_run": errors,
           "interpreter": sys.executable, "python": platform.python_version(),
           "platform": platform.platform(), "pip_freeze": _freeze()}
    Path(str(cache) + ".environment.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[pangolin w{args.worker}] done: {len(todo)} scored, {unscorable} unscorable, "
          f"{errors} exceptions", flush=True)
    return 0


def collect(args, df) -> int:
    import pandas as pd
    caches = sorted(Path(args.cache_dir).glob(f"w*_of_{args.workers}.tsv"))
    done: dict = {}
    for c in caches:
        done |= read_cache(c)
    missing = [v for v in df["variant_id"] if v not in done]
    if missing:
        raise SystemExit(f"{len(missing)} variants not in the workers' caches")
    res = pd.DataFrame({"variant_id": df["variant_id"],
                        COLUMN: [done[v][0] for v in df["variant_id"]],
                        "scored_utc": [done[v][1] for v in df["variant_id"]]})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out, index=False)
    pf_sha = sha256(SCORER)
    workers = [json.loads(Path(str(c) + ".environment.json").read_text()) for c in caches]
    env = {"written_by": "models/pangolin/score_external.py",
           "scorer": "models/pangolin/score_fullprec.py", "scorer_sha256": pf_sha,
           "driver_sha256": sha256(Path(__file__)),
           "tool": "Pangolin (git 5cf94b8), max(gain, |loss|) over transcript records, "
                   "full precision",
           "configuration": {"distance": 50, "mask": "True (Pangolin's default)",
                             "threads_per_process": 1, "workers": args.workers},
           "reference": {"path": str(args.reference), "sha256": sha256(Path(args.reference))},
           "gtf_db": {"path": str(args.gtf_db), "sha256": sha256(Path(args.gtf_db))},
           "input": {"path": args.input, "sha256": sha256(Path(args.input)), "rows": len(df)},
           "n_nan": int(res[COLUMN].isna().sum()),
           "scored_utc_range": [min(res["scored_utc"]), max(res["scored_utc"])],
           "workers_records": workers, "output_sha256": sha256(out)}
    Path(str(out) + ".environment.json").write_text(json.dumps(env, indent=1) + "\n")
    print(f"[pangolin] wrote {out}: {len(res)} rows, {env['n_nan']} NaN")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--reference", default=str(REFERENCE))
    ap.add_argument("--gtf-db", default=str(GTF_DB))
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--expect-sha256", default=None,
                    help="stop unless score_fullprec.py has this sha256")
    ap.add_argument("--limit", type=int, default=None, help="the first N rows only")
    args = ap.parse_args(argv)
    if args.expect_sha256 and sha256(SCORER) != args.expect_sha256:
        raise SystemExit(f"{SCORER.name} is not {args.expect_sha256}")
    import pandas as pd
    df = pd.read_parquet(args.input, columns=["variant_id", "chrom", "pos", "ref", "alt"])
    if args.limit:
        df = df.head(args.limit)
    if args.collect:
        if not args.output:
            raise SystemExit("--collect needs --output")
        return collect(args, df)
    return score(args, df)


if __name__ == "__main__":
    raise SystemExit(main())
