"""SpliceAI on a table of variants, by score_fullprec.py's own functions.

score_fullprec.py scores the atlas matrix and stays byte for byte the file that the
core genes and the companion release's external columns were scored with; its sha256
is checked against --expect-sha256 when given and recorded either way. This driver loads
it and calls its build_fullprec, aggregate, ANNOTATION ("grch38", SpliceAI's bundled
annotation), DISTANCE (50) and MASK (0) unchanged, on any table with variant_id, chrom,
pos, ref and alt, against --reference (default data/refs/grch38_external.fa, Ensembl
release 112 chromosomes 1, 7, 16, 17, built by models/pangolin/build_external_refs.py).

Each value is appended to --cache as it is scored (variant_id, repr(value), scored_utc),
so a stopped run resumes. A variant the scorer raises on is NaN, as in the atlas's run,
and counted. When every row is cached the cache is written to --output (parquet:
variant_id, spliceai_ds, scored_utc), with <output>.environment.json beside it: the
interpreter, python, platform and pip freeze read after the run (CONVENTIONS.md rule 3),
the scorer's and the reference's sha256, the configuration, start, end and counts.

Run with the SpliceAI environment, from the repository root:
    TF_CPP_MIN_LOG_LEVEL=3 models/spliceai/.venv/bin/python \\
        models/spliceai/score_external.py --input data/raw/external_v1/score_input.parquet \\
        --cache models/spliceai/cache/external_v1/spliceai.tsv \\
        --output models/spliceai/cache/external_v1/spliceai.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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
COLUMN = "spliceai_ds"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_cache(path: Path) -> dict[str, tuple[float, str]]:
    """variant_id -> (value, scored_utc), from a cache written by this script."""
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


def load_scorer(path: Path = SCORER):
    spec = importlib.util.spec_from_file_location("spliceai_fullprec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--reference", default=str(REFERENCE))
    ap.add_argument("--expect-sha256", default=None,
                    help="stop unless score_fullprec.py has this sha256")
    ap.add_argument("--limit", type=int, default=None, help="score the first N rows only")
    args = ap.parse_args(argv)

    import pandas as pd
    scorer_sha = sha256(SCORER)
    if args.expect_sha256 and scorer_sha != args.expect_sha256:
        raise SystemExit(f"{SCORER.name} is {scorer_sha}, not {args.expect_sha256}")
    S = load_scorer()
    from spliceai.utils import Annotator

    df = pd.read_parquet(args.input, columns=["variant_id", "chrom", "pos", "ref", "alt"])
    if args.limit:
        df = df.head(args.limit)
    cache, out = Path(args.cache), Path(args.output)
    done = read_cache(cache)
    todo = df[~df["variant_id"].isin(done)]
    print(f"[spliceai] {len(df)} variants, {len(done)} cached, {len(todo)} to score",
          flush=True)
    start, t0, errors = utc(), time.time(), 0
    ann = Annotator(args.reference, S.ANNOTATION)
    fullprec = S.build_fullprec()
    for i, r in enumerate(todo.itertuples(index=False), 1):
        rec = S.Rec(str(r.chrom), int(r.pos), r.ref, r.alt)
        try:
            value = S.aggregate(fullprec(rec, ann, S.DISTANCE, S.MASK))
        except Exception:                                          # noqa: BLE001
            value, errors = math.nan, errors + 1
        append_cache(cache, r.variant_id, value)
        if i % 200 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el / 60:.1f} min, {el / i:.2f} s each", flush=True)
    seconds = time.time() - t0
    done = read_cache(cache)
    missing = [v for v in df["variant_id"] if v not in done]
    if missing:
        raise SystemExit(f"{len(missing)} variants not in the cache")
    res = pd.DataFrame({"variant_id": df["variant_id"],
                        COLUMN: [done[v][0] for v in df["variant_id"]],
                        "scored_utc": [done[v][1] for v in df["variant_id"]]})
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out, index=False)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                            capture_output=True, text=True).stdout.split("\n")
    env = {"written_by": "models/spliceai/score_external.py",
           "scorer": "models/spliceai/score_fullprec.py", "scorer_sha256": scorer_sha,
           "driver_sha256": sha256(Path(__file__)),
           "tool": "SpliceAI 1.3.1, the four delta scores' maximum at full precision",
           "configuration": {"annotation": S.ANNOTATION, "distance": S.DISTANCE,
                             "mask": S.MASK},
           "reference": {"path": str(Path(args.reference).resolve().relative_to(REPO.resolve()))
                         if Path(args.reference).resolve().is_relative_to(REPO.resolve())
                         else args.reference,
                         "sha256": sha256(Path(args.reference))},
           "input": {"path": args.input, "sha256": sha256(Path(args.input)), "rows": len(df)},
           "interpreter": sys.executable, "python": platform.python_version(),
           "platform": platform.platform(), "pip_freeze": [x for x in freeze if x],
           "run_started_utc": start, "run_ended_utc": utc(),
           "seconds_this_run": round(seconds, 1), "n_scored_this_run": int(len(todo)),
           "n_exceptions_this_run": errors, "n_nan": int(res[COLUMN].isna().sum()),
           "scored_utc_range": [min(res["scored_utc"]), max(res["scored_utc"])],
           "output_sha256": sha256(out)}
    Path(str(out) + ".environment.json").write_text(json.dumps(env, indent=1) + "\n")
    print(f"[spliceai] wrote {out}: {len(res)} rows, {env['n_nan']} NaN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
