"""AlphaGenome's splicing score by its official definition, ported from the companion
release's scripts/stage_E17/score_official.py (tag v3.1.0-submission).

The definition is the official variant-scoring notebook as quoted in
NOTEBOOK_SOURCE.json (the companion release's file; copied to data/raw/companion_v3.1.0/
by `python -m atlas.external_scores companion`; its sha256 is checked), and the scoring
code is executed from there: cell 5 (the imports) and cell 9 (the three scorers) as
written; per variant, the lines of cell 15 that score one variant (an interval of
dna_client.SEQUENCE_LENGTH_1MB centred on it, dna_model.score_variant with the three
scorers, Homo sapiens), taken out of the notebook's loop so that two threads can each
score one; then variant_scorers.tidy_scores and cell 13's compute_merged_splicing_score,
as written. The model is created as in cell 7 except for the key, which comes only from
read_key(): the environment variable ALPHAGENOME_API_KEY, or ~/.config/alphagenome/key
with permissions 0600. The key is never printed, logged or written: only whether one was
found and where.

Columns: SPLICE_SITES, SPLICE_SITE_USAGE, SPLICE_JUNCTIONS and the merged score, which
the notebook calls alphagenome_splicing and this repository alphagenome_official, so that
it is not confused with the matrix's alphagenome column (models/alphagenome/score.py's
older definition).

Modes
  --check-start  the 10 variants of the release's drift check
                 (alphagenome_official/drift_2b_end_scores.tsv; their coordinates from
                 the release's analysis_set_v1.parquet), afresh; compared with the
                 release's values to the last bit, and the result recorded (a
                 difference is reported, it does not stop the run)
  --external     every row of --input
  --check-end    the same 10, afresh, after --external: equal to --check-start's to the
                 last bit, or the run stops
  --mock         deterministic offline scores: no network and no key (for the tests)

Guards, as in the release's script: at most two threads; a pause before each new
attempt, doubling (5 s, 10 s, ...), six attempts per variant; the run stops on an
authentication error, when three variants in a row have used all their attempts, on a
raw score below zero (the scorers are unsigned, so the maximum would no longer be the
maximum absolute score), and on Ctrl+C. Each scored variant is appended to a cache, one
file per mode, so a stopped --external run resumes; a check starts from an empty cache.
Writes <out>/<mode>_scores.tsv (values written with repr, so to the last bit) and
<mode>_scores.tsv.provenance.json: calls, retries, failures, timing, client version, the
client's line naming the model a request with no model_version gets, the definition's
sha256, and pip freeze read after the run (CONVENTIONS.md rule 3). Research use only:
the scores are not used to train any other model.

Run with its own environment (models/alphagenome/.venv, from requirements_official.txt
with --require-hashes --no-deps), from the repository root:
    models/alphagenome/.venv/bin/python models/alphagenome/score_official.py --check-start
    models/alphagenome/.venv/bin/python models/alphagenome/score_official.py --external \\
        --input data/raw/external_v1/score_input.parquet
    models/alphagenome/.venv/bin/python models/alphagenome/score_official.py --check-end
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
REPO = MODEL_DIR.parents[1]
COMPANION = REPO / "data" / "raw" / "companion_v3.1.0"
AG = COMPANION / "phase1" / "data" / "evidence" / "alphagenome_official"
SOURCE = AG / "NOTEBOOK_SOURCE.json"
SOURCE_SHA256_PREFIX = "8f050e9b0c224e0c"
DRIFT_REF = AG / "drift_2b_end_scores.tsv"
ANALYSIS_SET = COMPANION / "phase1" / "data" / "evidence" / "analysis_set_v1.parquet"
REQ = MODEL_DIR / "requirements_official.txt"
OUT = MODEL_DIR / "cache" / "official_v1"
KEY_ENV = "ALPHAGENOME_API_KEY"
KEY_FILE = Path.home() / ".config" / "alphagenome" / "key"

COMPONENTS = ["SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS"]
NOTEBOOK_COMPOSITE = "alphagenome_splicing"
COMPOSITE = "alphagenome_official"
MAX_WORKERS = 2
MAX_ATTEMPTS, BACKOFF = 6, 5.0
IN_A_ROW = 3
FATAL = ("UNAUTHENTICATED", "PERMISSION_DENIED", "API key", "api key")
DNA_MODEL = "alphagenome/models/v1/dna_model.py"
DEFAULT_LINE = "ALL_FOLDS = enum.auto()  # Default model version if None explicitly set."
MODES = ("check_start", "external", "check_end")
TABLE_COLUMNS = (COMPONENTS + [COMPOSITE]
                 + [f"{p}_{c}" for c in COMPONENTS for p in ("rows", "nan", "min_raw")]
                 + ["scored_utc"])


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ==========================================================================
# the key: read in one place, never written
# ==========================================================================
def read_key() -> str | None:
    """The key, from the environment or from a file outside the repository whose
    permissions are 0600; None when there is neither."""
    env = os.environ.get(KEY_ENV)
    if env:
        return env.strip()
    if KEY_FILE.exists():
        mode = stat.S_IMODE(KEY_FILE.stat().st_mode)
        if mode != 0o600:
            raise SystemExit(f"{KEY_FILE} has permissions {mode:o}; they must be 600")
        return KEY_FILE.read_text().strip() or None
    return None


def key_report() -> dict:
    key = read_key()
    found = bool(key)
    del key
    source = (KEY_ENV if os.environ.get(KEY_ENV) else
              "~/" + str(KEY_FILE.relative_to(Path.home())) if found else None)
    return {"api_key_present": found, "api_key_source": source}


def clean(msg: str) -> str:
    """An error message with the key, should it ever appear, cut out."""
    try:
        key = read_key()
    except SystemExit:
        key = None
    if key and key in msg:
        msg = msg.replace(key, "<key>")
    del key
    return msg[:300]


# ==========================================================================
# the notebook's code, executed as written
# ==========================================================================
class Notebook:
    """The quoted cells, executed as written; the model created with read_key()."""

    def __init__(self, source: Path = SOURCE):
        self.source_sha256 = sha256_of(source)
        if not self.source_sha256.startswith(SOURCE_SHA256_PREFIX):
            raise SystemExit(f"{source.name} is not the release's definition")
        src = json.loads(source.read_text())
        cells = {c["cell"]: c["source"] for c in src["quoted_cells"]}
        self.ns: dict = {}
        exec(cells[5], self.ns)                                    # imports
        key = read_key()
        if not key:
            raise SystemExit(f"no AlphaGenome API key ({KEY_ENV} or {KEY_FILE}): stopping")
        self.ns["dna_model"] = self.ns["dna_client"].create(key)   # cell 7, the key aside
        del key
        exec(cells[9], self.ns)                                    # the three scorers
        lines = cells[15].split("\n")
        start = next(i for i, l in enumerate(lines) if l.startswith("for variant in variants:"))
        end = next(i for i, l in enumerate(lines) if l.strip() == "all_scores.append(scores)")
        self.body = "\n".join(lines[start + 1:end])                # interval = ...; scores = ...
        exec("def score_one(variant):\n" + self.body + "\n  return scores\n", self.ns)
        exec(cells[13].split("\n\n\nmerged_scores")[0], self.ns)   # the function, not its call
        self.uns_keys = None

    def variant(self, chrom: str, pos: int, ref: str, alt: str):
        return self.ns["genome"].Variant(chromosome=f"chr{chrom}", position=int(pos),
                                         reference_bases=ref, alternate_bases=alt)

    def probe(self, chrom, pos, ref, alt) -> None:
        """One call, for the response's own metadata keys."""
        scores = self.ns["score_one"](self.variant(chrom, pos, ref, alt))
        self.uns_keys = sorted({k for ad in scores for k in ad.uns.keys()})

    def score(self, chrom, pos, ref, alt) -> dict:
        scores = self.ns["score_one"](self.variant(chrom, pos, ref, alt))
        df = self.ns["variant_scorers"].tidy_scores([scores])
        merged = self.ns["compute_merged_splicing_score"](df.copy())
        if len(merged) != 1:
            raise RuntimeError(f"{len(merged)} merged rows for one variant")
        row = merged.iloc[0]
        out = {c: float(row[c]) if c in merged.columns else 0.0 for c in COMPONENTS}
        out[COMPOSITE] = float(row[NOTEBOOK_COMPOSITE])
        for c in COMPONENTS:
            part = df.loc[df["output_type"] == c, "raw_score"]
            out[f"rows_{c}"] = int(len(part))
            out[f"nan_{c}"] = int(part.isna().sum())
            out[f"min_raw_{c}"] = float(part.min()) if part.notna().any() else None
        return out


class MockNotebook:
    """Deterministic offline scores, from a hash of the variant: for the tests."""
    source_sha256 = "mock"
    body = "mock"
    uns_keys = ["mock"]

    def probe(self, *a) -> None:
        return None

    def score(self, chrom, pos, ref, alt) -> dict:
        d = hashlib.sha256(f"{chrom}:{pos}:{ref}:{alt}".encode()).digest()
        vals = [int.from_bytes(d[i:i + 4], "big") / 2 ** 32 for i in (0, 4, 8)]
        out = dict(zip(COMPONENTS, vals))
        out[COMPOSITE] = vals[0] + vals[1] + vals[2] / 5
        for c, v in zip(COMPONENTS, vals):
            out[f"rows_{c}"], out[f"nan_{c}"], out[f"min_raw_{c}"] = 3, 0, v / 2
        return out


# ==========================================================================
# the gentle runner
# ==========================================================================
class Stopped(SystemExit):
    """The run stopped before its end; .state holds what it had done."""

    def __init__(self, why: str, state: dict):
        super().__init__(why)
        self.state = state


def run(rows, score, record, *, workers: int, attempts: int = MAX_ATTEMPTS,
        backoff: float = BACKOFF, in_a_row: int = IN_A_ROW,
        stop: threading.Event | None = None) -> dict:
    """score(row) for every row, `workers` (at most MAX_WORKERS) at a time, and
    record(row, result) for each result. An error is tried again after a pause
    (backoff seconds, doubling), up to `attempts` calls; a variant that uses them all is
    a failure. Raises Stopped on an authentication error, when `in_a_row` variants in a
    row have used all their attempts, and on Ctrl+C; the variants not yet started are
    then cancelled and no new call begins. Returns the calls, retries and failures."""
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be 1..{MAX_WORKERS}, not {workers}")
    stop = stop or threading.Event()
    lock = threading.Lock()
    state = {"calls": 0, "retries": 0, "failures": [], "in_a_row": 0, "why": None,
             "waits": []}

    def halt(why: str) -> None:                # with the lock held
        state["why"] = state["why"] or why
        stop.set()

    def one(r) -> None:
        for attempt in range(1, attempts + 1):
            if stop.is_set():
                return
            with lock:
                state["calls"] += 1
            try:
                out = score(r)
            except Exception as e:                              # noqa: BLE001
                msg = clean(f"{type(e).__name__}: {e}")
                if any(f in msg for f in FATAL):
                    with lock:
                        halt(f"authentication error: stopping: {msg}")
                    return
                if attempt < attempts:
                    wait = backoff * 2 ** (attempt - 1)
                    with lock:
                        state["retries"] += 1
                        state["waits"].append(wait)
                    stop.wait(wait)
                    continue
                with lock:
                    state["failures"].append({"variant_id": r.variant_id, "error": msg})
                    state["in_a_row"] += 1
                    if state["in_a_row"] >= in_a_row:
                        halt(f"{in_a_row} variants in a row failed after {attempts} attempts "
                             f"each: stopping; the cache keeps every variant scored")
                return
            with lock:
                state["in_a_row"] = 0
            record(r, out)
            return

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for fut in as_completed([pool.submit(one, r) for r in rows]):
            fut.result()
            if stop.is_set():
                break
    except KeyboardInterrupt:
        with lock:
            halt("interrupted (Ctrl+C): stopping; the cache keeps every variant scored")
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
    if state["why"]:
        raise Stopped(state["why"], state)
    return state


# ==========================================================================
# checks and records
# ==========================================================================
def read_table(path: Path):
    import pandas as pd
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).set_index("variant_id")


def differing(a_path: Path, b_path: Path, ids: list[str], b_composite: str = COMPOSITE) -> list[str]:
    """The variants among ids whose three components or merged score, as written, are
    not those of b to the last character (both written with repr: to the last bit)."""
    a, b = read_table(a_path), read_table(b_path)
    cols_b = COMPONENTS + [b_composite]
    return [v for v in ids
            if list(a.loc[v, COMPONENTS + [COMPOSITE]]) != list(b.loc[v, cols_b])]


def negative_raw(table) -> list[str]:
    """Components whose smallest raw score is below zero."""
    import pandas as pd
    return [c for c in COMPONENTS
            if (pd.to_numeric(table[f"min_raw_{c}"], errors="coerce") < 0).any()]


def drift_variants():
    """The release's drift-check variants, with their coordinates."""
    import pandas as pd
    ref = pd.read_csv(DRIFT_REF, sep="\t", dtype=str)
    s = pd.read_parquet(ANALYSIS_SET, columns=["variant_id", "gene", "chrom", "pos", "ref", "alt"])
    d = ref[["variant_id"]].merge(s, on="variant_id", how="left")
    if d["chrom"].isna().any() or len(d) != 10:
        raise SystemExit("the drift variants are not all in the release's analysis set")
    d["chrom"], d["pos"] = d["chrom"].astype(str), d["pos"].astype(int)
    return d


def model_default() -> dict:
    """Which model a request with no model_version gets, in the client's own words: the
    line of dna_model.py that says so; the file's sha256, which must be the one the
    wheel's RECORD lists; the wheel's sha256 as requirements_official.txt pins it."""
    from importlib import metadata
    dist = metadata.distribution("alphagenome")
    entry = next(f for f in dist.files if str(f) == DNA_MODEL)
    data = Path(entry.locate()).read_bytes()
    digest = hashlib.sha256(data)
    if (entry.hash.mode, entry.hash.value) != (
            "sha256", base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()):
        raise SystemExit(f"{DNA_MODEL} is not the file the alphagenome wheel installed")
    lines = data.decode().split("\n")
    hits = [n for n, line in enumerate(lines, 1) if DEFAULT_LINE in line]
    pins = [l for l in REQ.read_text().splitlines()
            if l.startswith(f"alphagenome=={dist.version} --hash=sha256:")]
    if len(hits) != 1 or len(pins) != 1:
        raise SystemExit(f"{len(hits)} lines of {DNA_MODEL} hold the default line, and "
                         f"{len(pins)} lines of {REQ.name} pin alphagenome {dist.version}")
    return {"file": DNA_MODEL, "line": hits[0], "text": lines[hits[0] - 1],
            "file_sha256": digest.hexdigest(), "file_sha256_in_wheel_record": True,
            "wheel": f"alphagenome=={dist.version}",
            "wheel_sha256": pins[0].split("--hash=sha256:")[1]}


def environment(mock: bool) -> dict:
    freeze = sorted(x for x in subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True,
        text=True).stdout.split("\n") if x)
    env = {"interpreter": sys.executable, "python": sys.version, "platform": platform.platform(),
           "pip_freeze": freeze, "requirements": str(REQ.relative_to(REPO)),
           "requirements_sha256": sha256_of(REQ)}
    if not mock:
        import alphagenome
        env |= {"client": {"package": "alphagenome", "version": alphagenome.__version__},
                "model_version_requested": None,
                "model_version_requested_note": "dna_client.create(...) as in the notebook's "
                "cell 7, with no model_version: the service's default",
                "default_when_not_requested": model_default(),
                "model_version_in_response": "not_provided"} | key_report()
    return env


# ==========================================================================
# command line
# ==========================================================================
def load_cache(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line:
                r = json.loads(line)
                done[r["variant_id"]] = r
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    for m in MODES:
        mode.add_argument(f"--{m.replace('_', '-')}", dest="mode", action="store_const", const=m)
    ap.add_argument("--input", default=str(REPO / "data/raw/external_v1/score_input.parquet"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--mock", action="store_true", help="offline deterministic scores (tests)")
    args = ap.parse_args(argv)
    import pandas as pd

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"{args.mode}.jsonl"
    if args.mode != "external" and cache.exists() and cache.read_text().strip():
        raise SystemExit(f"{cache} is not empty: a check scores its variants afresh")
    if args.mode == "check_end" and not (out_dir / "check_start_scores.tsv").exists():
        raise SystemExit("--check-end compares with --check-start, which has not run")
    if args.mode == "external":
        todo = pd.read_parquet(args.input, columns=["variant_id", "gene", "chrom", "pos",
                                                    "ref", "alt"])
        todo["chrom"], todo["pos"] = todo["chrom"].astype(str), todo["pos"].astype(int)
        key = ["gene", "variant_id", "chrom", "pos", "ref", "alt"]
    else:
        todo = drift_variants()
        key = ["gene", "variant_id", "chrom", "pos", "ref", "alt"]
    nb = MockNotebook() if args.mock else Notebook()
    done = load_cache(cache)
    rows = [r for r in todo.itertuples() if r.variant_id not in done]
    print(f"[{args.mode}] {len(todo)} variants, {len(done)} cached, {len(rows)} to score "
          f"with {args.workers} worker(s)", flush=True)
    start, t0, lock = utc(), time.time(), threading.Lock()
    first = todo.iloc[0]
    try:
        nb.probe(first["chrom"], first["pos"], first["ref"], first["alt"])
    except Exception as e:                                         # noqa: BLE001
        raise SystemExit(f"the first call failed: stopping: {clean(f'{type(e).__name__}: {e}')}"
                         ) from None

    def record(r, res):
        rec = {"variant_id": r.variant_id, "scored_utc": utc()} | res
        with lock:
            with open(cache, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done[r.variant_id] = rec
            if len(done) % 100 == 0:
                print(f"  {len(done)} scored, {time.time() - t0:.0f} s", flush=True)

    stats = run(rows, lambda r: nb.score(r.chrom, r.pos, r.ref, r.alt), record,
                workers=args.workers)
    if stats["failures"]:
        raise SystemExit(f"{len(stats['failures'])} variants failed after {MAX_ATTEMPTS} "
                         f"attempts: {stats['failures'][:3]}")
    table = todo[key].merge(pd.DataFrame([done[v] for v in todo["variant_id"]]), on="variant_id")
    out = out_dir / f"{args.mode}_scores.tsv"
    table[key + TABLE_COLUMNS].to_csv(out, sep="\t", index=False,
                                      float_format=lambda x: repr(float(x)))
    checks, stops = {}, []
    if negative_raw(table):
        stops.append(f"a raw score below zero in {negative_raw(table)}")
    ids = todo["variant_id"].tolist()
    if args.mode == "check_start":
        diff = differing(out, DRIFT_REF, ids, b_composite=NOTEBOOK_COMPOSITE)
        checks["against_release"] = {"against": str(DRIFT_REF.relative_to(REPO)),
                                     "against_sha256": sha256_of(DRIFT_REF), "variants": len(ids),
                                     "identical": len(ids) - len(diff), "differing": diff}
    if args.mode == "check_end":
        ref = out_dir / "check_start_scores.tsv"
        diff = differing(out, ref, ids)
        checks["against_check_start"] = {"variants": len(ids), "identical": len(ids) - len(diff),
                                         "differing": diff, "against_sha256": sha256_of(ref)}
        if diff:
            stops.append(f"{len(diff)} of {len(ids)} differ from --check-start")
        ext = out_dir / "external_scores.tsv.provenance.json"
        if ext.exists():
            ended = json.loads(ext.read_text())["end_utc"]
            early = [v for v in ids if done[v]["scored_utc"] <= ended]
            checks["after_external"] = {"external_end_utc": ended,
                                        "every_scored_utc_later": not early}
            if early:
                stops.append("the end check did not score its variants after --external")
    prov = {"written_by": "models/alphagenome/score_official.py", "mode": args.mode,
            "mock": args.mock, "definition": str(SOURCE.relative_to(REPO)),
            "definition_sha256": nb.source_sha256,
            "code_run": {"cells_as_written": [5, 9, 13, "15 (the lines scoring one variant, "
                                              "out of the loop)"],
                         "cell_7": "dna_client.create(read_key()): the key aside, as written",
                         "per_variant_lines": nb.body},
            "response_metadata_keys": nb.uns_keys,
            "start_utc": start, "end_utc": utc(), "seconds_scoring": round(time.time() - t0, 1),
            "n_variants": int(len(table)), "n_scored_this_run": len(rows),
            "n_from_cache": int(len(todo) - len(rows)),
            "scored_utc_range": [min(table["scored_utc"]), max(table["scored_utc"])],
            "calls": stats["calls"] + 1, "calls_note": "including the one probe call",
            "retries": stats["retries"], "failures": stats["failures"],
            "workers": args.workers,
            "n_by_gene": {g: int(n) for g, n in table["gene"].value_counts().items()},
            "summary": {c: {"variants_with_no_row": int((table[f"rows_{c}"] == 0).sum()),
                            "nan_raw_scores": int(table[f"nan_{c}"].sum())} for c in COMPONENTS},
            "environment": environment(args.mock)} | checks | {
            "output": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out),
            "output_sha256": sha256_of(out)}
    Path(str(out) + ".provenance.json").write_text(json.dumps(prov, indent=1,
                                                              ensure_ascii=False) + "\n")
    print(f"[{args.mode}] wrote {out}: {len(table)} variants, {len(rows)} scored this run, "
          f"{stats['retries']} retries")
    for k, v in checks.items():
        print(f"  {k}: " + (f"{v['identical']}/{v['variants']} identical" if "identical" in v
                            else json.dumps(v)))
    if stops:
        raise SystemExit("; ".join(stops) + ": stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
