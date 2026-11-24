"""AlphaGenome Atlas: the companion release's four columns (avi, avi_splice_sites,
avi_splice_site_usage, avi_splice_junctions) for the atlas's SNVs and the external genes'.

Ported from the companion release (tag v3.1.0-submission, read with git show):
phase1/src/evid_score_avi.py's run() and _aggregate(), and the guards
scripts/stage_E18/run_avi.py wraps around the client. The Atlas is AlphaGenome's
precomputed variant scores behind its own API (client 0.9.0, alphagenome.atlas), a
different service from the per-variant scoring of models/alphagenome/.

Per variant, each column is the maximum over tracks and over rows (SPLICE_JUNCTIONS
returns one row per junction) of the scorer's X, which client 0.9.0 fills with the
service's raw scores; the calibrated scores in the 'quantiles' layer are not read (as
the release's alphagenome_official/ATLAS_COLUMNS_CHECK.json records).

Modes
  --list      the scorers the service offers, recorded as served (<out>/scorers.json)
  --anchors   the release's 39 anchors (external/e18_anchors_avi.parquet's
              coordinates), compared with its four columns to the bit
  --run       every row of --input (variant_id, gene, set, chrom, pos, ref, alt)
  --mock      deterministic offline responses: no network and no key (for the tests)

Guards, as the release's: batches of 250 variants; a failed request is tried again, six
attempts per batch, after a pause that doubles (a quota error: 20 s, 40 s, ...; any other
error: 5 s, 10 s, ..., plus run()'s 2 s); an authentication error stops the run at once,
and so does a batch whose every attempt fails; every scored batch goes to a checkpoint,
so a stopped run resumes. Before anything is scored the scorers the service offers are
listed and recorded as served, and each invocation's start, end and errors are kept
(<out>/segments.json). The key is read only by read_key() (ALPHAGENOME_API_KEY, or
~/.config/alphagenome/key with permissions 0600) and never printed or written; records
say only whether one was found and where.

Writes <out>/atlas_scores.parquet, its .provenance.json and <out>/environment.json (pip
freeze read after the run, CONVENTIONS.md rule 3). Licence: AlphaGenome Atlas and API
outputs are for non-commercial research use only and must not be used to train other
models.

Run in the Atlas environment (alphagenome 0.9.0; see NOTES.md), from the repository root:
    P=~/work/envs/alphagenome-atlas/bin/python
    $P models/alphagenome_atlas/score.py --anchors
    $P models/alphagenome_atlas/score.py --run --input data/raw/alphagenome_atlas_v1/score_input.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
REPO = MODEL_DIR.parents[1]
COMPANION = REPO / "data" / "raw" / "companion_v3.1.0"
ANCHORS = COMPANION / "phase1" / "data" / "evidence" / "external" / "e18_anchors_avi.parquet"
OUT = MODEL_DIR / "cache" / "atlas_v1"
ANCHORS_OUT = MODEL_DIR / "cache" / "anchors_v1"
KEY_ENV = "ALPHAGENOME_API_KEY"
KEY_FILE = Path.home() / ".config" / "alphagenome" / "key"

MIN_CLIENT = (0, 9, 0)
AVI_SCORER = "AVI_SCORE"
SPLICE_SCORERS = ["SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS"]
AVI_HINTS = ("avi", "variant_impact", "variant impact")
COLUMNS = ["avi", "avi_splice_sites", "avi_splice_site_usage", "avi_splice_junctions"]
BATCH = 250
MAX_RETRIES = 6
BACKOFF_BASE = 20.0          # seconds, after a quota error; the quota is per minute
RETRY_PAUSE = 2.0            # run()'s pause after any other error
BACKOFF = 5.0                # the guard's, before run()'s, after any other error
FATAL = ("UNAUTHENTICATED", "PERMISSION_DENIED", "API key", "api key")
INPUT_COLUMNS = ["variant_id", "gene", "set", "chrom", "pos", "ref", "alt"]
PRODUCT = "AlphaGenome Atlas -- AlphaGenome Variant Impact (AVI)"
LICENCE = ("AlphaGenome Terms of Service -- NON-COMMERCIAL research use only; outputs "
           "must not be used to train other models")
LICENCE_SOURCE = "https://deepmind.google.com/science/alphagenome/terms"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ==========================================================================
# the key: read in one place, never written
# ==========================================================================
def read_key() -> str | None:
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
    try:
        key = read_key()
    except SystemExit:
        key = None
    if key and key in msg:
        msg = msg.replace(key, "<key>")
    del key
    return msg[:300]


# ==========================================================================
# the client, and the release's guards around it
# ==========================================================================
class Stop(SystemExit):
    """Not an Exception, so the retry loop lets it through."""


class Guarded:
    """scripts/stage_E18/run_avi.py's wrapper: an authentication error stops at once;
    any other error backs off before the retry; a batch whose every attempt fails
    stops the run."""

    def __init__(self, client):
        self._c = client
        self.in_a_row, self.errors, self.scorers = 0, [], None

    def scorer_metadata(self):
        md = self._c.scorer_metadata()
        self.scorers = sorted(md)
        return md

    def query_variants(self, *args, **kwargs):
        try:
            out = self._c.query_variants(*args, **kwargs)
        except Exception as e:                                  # noqa: BLE001
            msg = clean(f"{type(e).__name__}: {e}")
            if any(f in msg for f in FATAL):
                raise Stop(f"authentication error: stopping: {msg}")
            self.in_a_row += 1
            self.errors.append({"utc": utc(), "error": msg})
            if self.in_a_row >= MAX_RETRIES:
                raise Stop(f"a batch failed all {MAX_RETRIES} attempts: stopping; the "
                           "checkpoint keeps every batch scored")
            if "RESOURCE_EXHAUSTED" not in msg and "Quota exceeded" not in msg:
                time.sleep(BACKOFF * 2 ** (self.in_a_row - 1))
            raise
        self.in_a_row = 0
        return out


def real_client():
    from alphagenome.atlas import atlas
    key = read_key()
    if not key:
        raise SystemExit(f"no AlphaGenome API key ({KEY_ENV} or {KEY_FILE}): stopping")
    client = atlas.create(key)
    del key
    return client


class _Meta:
    def __init__(self, name):
        self.name = name
        self.description = f"mock scorer {name}"


class _AnnData:
    def __init__(self, x, variants):
        self.X = x
        self.obs = pd.DataFrame({"variant": variants})


class MockClient:
    """Deterministic offline responses shaped like client 0.9.0's: three tracks per
    scorer, two rows per variant for SPLICE_JUNCTIONS."""

    def scorer_metadata(self):
        return {n: _Meta(n) for n in [AVI_SCORER, *SPLICE_SCORERS, "OTHER_SCORER"]}

    def query_variants(self, variants, requested_scorers, progress_bar=False):
        out = {}
        for s in requested_scorers:
            keys, rows = [], []
            for v in variants:
                k = f"{v.chromosome}:{v.position}:{v.reference_bases}>{v.alternate_bases}"
                for j in range(2 if s == "SPLICE_JUNCTIONS" else 1):
                    d = hashlib.sha256(f"{s}|{k}|{j}".encode()).digest()
                    rows.append([int.from_bytes(d[i:i + 4], "big") / 2 ** 32 for i in (0, 4, 8)])
                    keys.append(k)
            out[s] = _AnnData(np.asarray(rows, dtype=np.float32), keys)
        return out


class _Variant:
    def __init__(self, chromosome, position, reference_bases, alternate_bases):
        self.chromosome, self.position = chromosome, position
        self.reference_bases, self.alternate_bases = reference_bases, alternate_bases


def client_version(mock: bool) -> str:
    if mock:
        return "mock"
    import alphagenome
    return getattr(alphagenome, "__version__", "unknown")


# ==========================================================================
# the release's run(), pointed at a table
# ==========================================================================
def _describe(meta) -> dict:
    """Whatever the service says about a scorer, as plain data (the release's)."""
    out = {}
    for attr in ("name", "description", "version", "output_type", "organism",
                 "scorer_type", "requires_gene", "ontology_terms", "num_tracks"):
        v = getattr(meta, attr, None)
        if v is None:
            continue
        try:
            out[attr] = v if isinstance(v, (str, int, float, bool)) else str(v)
        except Exception:                                    # noqa: BLE001
            pass
    if not out:
        out["repr"] = str(meta)[:400]
    return out


def _aggregate(ad, variants_order: list[str]) -> np.ndarray:
    """One value per variant: the maximum over tracks, and over rows where a
    scorer returns several (SPLICE_JUNCTIONS emits one row per junction).

    The maximum is the atlas's own aggregation over splice tracks, so the columns
    are comparable in construction even though the score type is not the same one
    the v0.7.0 composite was built on (`alphagenome_atlas_quantile` since E17).
    """
    import pandas as _pd
    x = np.asarray(ad.X, dtype=float)
    per_row = np.nanmax(x, axis=1) if x.ndim == 2 and x.shape[1] > 1 else x.ravel()
    key = ad.obs["variant"].astype(str).to_numpy()
    s = _pd.Series(per_row).groupby(key).max()
    return s.reindex(variants_order).to_numpy(dtype=float)


def list_scorers(client, out: Path, version: str) -> dict:
    md = client.scorer_metadata()
    rec = {"listed_utc": utc(), "client_version": version, "n_scorers": len(md),
           "scorers": {name: _describe(meta) for name, meta in sorted(md.items())},
           "avi_candidates": sorted(n for n in md if any(h in n.lower() for h in AVI_HINTS))}
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorers.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    return {"record": rec, "metadata": md}


def run(todo: pd.DataFrame, client, out: Path, *, version: str, batch: int = BATCH,
        sleep=time.sleep, genome_variant=None) -> dict:
    """Score every row of todo (resuming from a checkpoint) and write the output."""
    listed = list_scorers(client, out, version)
    md = listed["metadata"]
    if AVI_SCORER not in md:
        raise SystemExit(f"the service offers no {AVI_SCORER}; candidates: "
                         f"{listed['record']['avi_candidates']}: stopping")
    wanted = [AVI_SCORER] + [s for s in SPLICE_SCORERS if s in md]
    colname = {AVI_SCORER: "avi"}
    for s in SPLICE_SCORERS:
        colname[s] = "avi_splice_" + s.replace("SPLICE_", "").lower()
    cols = [colname[s] for s in wanted]
    if cols != COLUMNS:
        raise SystemExit(f"the service offers {wanted}: not the four columns: stopping")
    print(f"[atlas] scorers: {wanted}", flush=True)
    if genome_variant is None:
        from alphagenome.data import genome
        genome_variant = genome.Variant

    todo = todo.drop_duplicates("variant_id").reset_index(drop=True)
    out_path = out / "atlas_scores.parquet"
    ckpt = out / "atlas_scores.partial.parquet"
    done = {}
    for src in (ckpt, out_path):
        if not src.exists():
            continue
        prev = pd.read_parquet(src)
        if not all(c in prev.columns for c in cols):
            continue
        for r in prev[["variant_id"] + cols].itertuples(index=False):
            if not any(pd.isna(x) for x in r[1:]):
                done[r[0]] = tuple(r[1:])
        break
    if done:
        print(f"[resume] {len(done):,} of {len(todo):,} already scored", flush=True)

    failures, t0, n_requests, calls, retries = [], time.time(), 0, 0, 0
    pending = todo[~todo["variant_id"].isin(done)]
    for start in range(0, len(pending), batch):
        chunk = pending.iloc[start:start + batch]
        vs = [genome_variant(chromosome=f"chr{r.chrom}", position=int(r.pos),
                             reference_bases=r.ref, alternate_bases=r.alt)
              for r in chunk.itertuples()]
        order = [f"chr{r.chrom}:{int(r.pos)}:{r.ref}>{r.alt}" for r in chunk.itertuples()]
        res = None
        for attempt in range(MAX_RETRIES):
            calls += 1
            try:
                res = client.query_variants(vs, requested_scorers=wanted, progress_bar=False)
                n_requests += len(vs)
                break
            except Exception as e:                            # noqa: BLE001
                msg = clean(str(e))
                exhausted = "RESOURCE_EXHAUSTED" in msg or "Quota exceeded" in msg
                if attempt == MAX_RETRIES - 1:
                    failures.append({"first_variant_id": chunk["variant_id"].iloc[0],
                                     "n": len(chunk), "attempts": MAX_RETRIES,
                                     "error": msg[:300]})
                    print(f"  batch at {start} FAILED after {MAX_RETRIES}: {msg[:100]}",
                          flush=True)
                    break
                wait = BACKOFF_BASE * (2 ** attempt) if exhausted else RETRY_PAUSE
                retries += 1
                print(f"  batch at {start} {'quota' if exhausted else 'error'}, "
                      f"retry {attempt + 1}/{MAX_RETRIES - 1} in {wait:.0f}s", flush=True)
                sleep(wait)
        if res is None:
            continue
        vals = {s: _aggregate(res[s], order) for s in wanted}
        for k, r in enumerate(chunk.itertuples()):
            done[r.variant_id] = tuple(float(vals[s][k]) for s in wanted)
        frame = todo[todo["variant_id"].isin(done)].copy()
        arr = np.asarray([done[v] for v in frame["variant_id"]], dtype=float)
        for c_i, c in enumerate(cols):
            frame[c] = arr[:, c_i]
        frame.to_parquet(ckpt, index=False)
        el = time.time() - t0
        n_left = len(pending) - min(start + batch, len(pending))
        print(f"  {len(done):>6}/{len(todo)}  {el / 60:5.1f} min, "
              f"~{n_left * el / max(start + batch, 1) / 60:5.1f} min left", flush=True)

    frame = todo.copy()
    arr = np.asarray([done.get(v, (np.nan,) * len(cols)) for v in frame["variant_id"]],
                     dtype=float)
    for c_i, c in enumerate(cols):
        frame[c] = arr[:, c_i]
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    complete = not failures and not frame[cols].isna().any().any()
    if complete:
        ckpt.unlink(missing_ok=True)
    return {"columns": cols, "scorers_requested": wanted, "n_variants": int(len(frame)),
            "n_scored": {c: int(frame[c].notna().sum()) for c in cols},
            "by_set": {str(name): {"n": int(len(g)), **{c: int(g[c].notna().sum()) for c in cols}}
                       for name, g in frame.groupby("set")},
            "n_requests": n_requests, "n_requests_note": "variants sent in batches that "
            "succeeded (the release's count)", "calls": calls,
            "retries": retries,
            "n_from_checkpoint": len(done) - n_requests,
            "failed_batches": failures, "runtime_minutes": round((time.time() - t0) / 60, 1),
            "complete": complete, "output": out_path}


# ==========================================================================
# checks and records
# ==========================================================================
def anchor_input(path: Path = ANCHORS) -> pd.DataFrame:
    a = pd.read_parquet(path)
    if len(a) != 39:
        raise SystemExit(f"{path.name} has {len(a)} rows, not the release's 39")
    a = a.assign(set="anchor")
    a["chrom"], a["pos"] = a["chrom"].astype(str), a["pos"].astype(int)
    return a[INPUT_COLUMNS]


def anchor_check(got: pd.DataFrame, path: Path = ANCHORS) -> dict:
    """Each column, scored here against the release's anchors: equal as float64, NaN
    equal to NaN (the release's _e18.anchor_check)."""
    stored = pd.read_parquet(path).set_index("variant_id")
    g = got.set_index("variant_id")
    out = {}
    for c in COLUMNS:
        ids = [i for i in stored.index if i in g.index]
        a = g.loc[ids, c].to_numpy(dtype=float)
        b = stored.loc[ids, c].to_numpy(dtype=float)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        out[c] = {"n_anchors": int(len(stored)), "n_checked": len(ids),
                  "n_identical": int(same.sum()),
                  "differing": [{"variant_id": i, "stored": repr(float(y)), "scored": repr(float(x))}
                                for i, x, y, s in zip(ids, a, b, same) if not s]}
    out["holds"] = all(out[c]["n_identical"] == out[c]["n_anchors"] for c in COLUMNS)
    return out


def environment(mock: bool) -> dict:
    freeze = sorted(x for x in subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True,
        text=True).stdout.split("\n") if x)
    env = {"interpreter": sys.executable, "python": platform.python_version(),
           "platform": platform.platform(), "pip_freeze": freeze,
           "client_version": client_version(mock)}
    if not mock:
        try:
            from alphagenome.atlas import atlas  # noqa: F401
            env["atlas_module"] = True
        except Exception:                                       # noqa: BLE001
            env["atlas_module"] = False
        env |= key_report()
    return env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", dest="mode", action="store_const", const="list")
    mode.add_argument("--anchors", dest="mode", action="store_const", const="anchors")
    mode.add_argument("--run", dest="mode", action="store_const", const="run")
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--mock", action="store_true", help="offline deterministic responses (tests)")
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else (ANCHORS_OUT if args.mode == "anchors" else OUT)
    version = client_version(args.mock)
    if not args.mock and tuple(int(x) for x in version.split(".")[:3]) < MIN_CLIENT:
        raise SystemExit(f"alphagenome {version} has no Atlas API (0.9.0 or later): stopping")
    raw = MockClient() if args.mock else real_client()
    client = Guarded(raw)
    if args.mode == "list":
        rec = list_scorers(client, out, version)["record"]
        print(f"[atlas] the service offers {rec['n_scorers']} scorers; recorded in "
              f"{out / 'scorers.json'}")
        return 0
    if args.mode == "anchors":
        todo = anchor_input()
    else:
        if not args.input:
            raise SystemExit("--run needs --input")
        todo = pd.read_parquet(args.input, columns=INPUT_COLUMNS)
        todo["chrom"], todo["pos"] = todo["chrom"].astype(str), todo["pos"].astype(int)
    segf = out / "segments.json"
    out.mkdir(parents=True, exist_ok=True)
    segments = json.loads(segf.read_text()) if segf.exists() else []
    segments.append({"start_utc": utc()})
    segf.write_text(json.dumps(segments, indent=1) + "\n")
    stats = None
    try:
        stats = run(todo, client, out, version=version, batch=args.batch,
                    genome_variant=_Variant if args.mock else None)
    finally:
        segments[-1].update({"end_utc": utc(), "errors": client.errors,
                             "scorers_offered": client.scorers,
                             "n_requests": stats["n_requests"] if stats else None})
        segf.write_text(json.dumps(segments, indent=1) + "\n")
    got = pd.read_parquet(stats["output"])
    checks = {"anchors": anchor_check(got)} if args.mode == "anchors" else {}
    prov = {"written_by": "models/alphagenome_atlas/score.py", "mode": args.mode,
            "mock": args.mock, "product": PRODUCT, "client_version": version,
            "ported_from": {"run": "phase1/src/evid_score_avi.py::run (tag v3.1.0-submission)",
                            "guards": "scripts/stage_E18/run_avi.py::Guarded"},
            "aggregation": "per variant, the maximum over tracks and over rows of X (the "
                           "service's raw scores); SPLICE_JUNCTIONS returns one row per "
                           "junction; the 'quantiles' layer is not read",
            "scorer_list": str(out / "scorers.json"), "accessed_utc": utc(),
            "segments": segments,
            **{k: v for k, v in stats.items() if k != "output"},
            "input": str(args.input) if args.input else str(ANCHORS),
            "input_sha256": sha256_of(Path(args.input) if args.input else ANCHORS),
            "licence": LICENCE, "licence_source": LICENCE_SOURCE,
            "output": str(stats["output"]), "output_sha256": sha256_of(stats["output"])} | checks
    Path(str(stats["output"]) + ".provenance.json").write_text(
        json.dumps(prov, indent=1, default=str) + "\n")
    (out / "environment.json").write_text(json.dumps(environment(args.mock), indent=1) + "\n")
    print(f"[atlas] wrote {stats['output']}: {stats['n_variants']} variants, "
          f"{stats['n_requests']} requested this run, {len(stats['failed_batches'])} failed batches")
    if checks:
        a = checks["anchors"]
        print("  anchors: " + ", ".join(f"{c} {a[c]['n_identical']}/{a[c]['n_anchors']}"
                                        for c in COLUMNS))
        if not a["holds"]:
            raise SystemExit("an anchor differs from the release: stopping")
    if not stats["complete"]:
        raise SystemExit("variants unscored; the checkpoint is kept: run again to continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
