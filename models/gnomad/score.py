"""gnomAD v4 allele-frequency baseline scorer (GRCh38, GraphQL API).

Contract (see models/_template/README.md):
    python score.py --input <variant_table.parquet> --output scores.parquet \
        --metric global

Score column: `gnomad_af_global` or `gnomad_af_popmax` (run once per metric).
Higher AF = more likely BENIGN, so the value is FLIPPED (score = -AF) to
satisfy the contract's "larger = more damaging" rule; recorded in NOTES.md.
Concordance vs the companion's raw AF columns is therefore expected at
rho = -1.0.

Definitions (identical to companion benchmark scripts/72_gnomad.py):
  - global: max(genome.af, exome.af)
  - popmax: max over continental populations {afr, amr, eas, nfe, sas, fin,
    mid, asj} of ac/an, taken over genome and exome, then flipped.

Variants ABSENT from gnomAD are left unscored (dropped, never imputed as 0)
— same conservative choice as the companion benchmark; see NOTES.md.

Source: https://gnomad.broadinstitute.org/api (GraphQL), dataset gnomad_r4,
queried per genomic region (one request per variant block, ~7 requests for
the whole matrix — no rate-limit concern). Raw responses cached per block.

Use --mock to run offline (deterministic pseudo-scores, for tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://gnomad.broadinstitute.org/api"
DATASET = "gnomad_r4"
MODEL_DIR = Path(__file__).resolve().parent
METRICS = {"global": "gnomad_af_global", "popmax": "gnomad_af_popmax"}
POPMAX_IDS = {"afr", "amr", "eas", "nfe", "sas", "fin", "mid", "asj"}

QUERY = """
query($chrom: String!, $start: Int!, $stop: Int!) {
  region(chrom: $chrom, start: $start, stop: $stop, reference_genome: GRCh38) {
    variants(dataset: %s) {
      variant_id
      genome { af populations { id ac an } }
      exome  { af populations { id ac an } }
    }
  }
}
""" % DATASET


def variant_blocks(df: pd.DataFrame, gap: int = 50_000) -> list[tuple[str, int, int]]:
    """Cluster variant positions into (chrom, start0, end) fetch blocks."""
    blocks = []
    for chrom, sub in df.groupby("chrom"):
        pos = sorted(int(p) for p in sub["pos"].unique())
        start = prev = pos[0]
        for p in pos[1:]:
            if p - prev > gap:
                blocks.append((str(chrom), start - 1, prev + 1))
                start = p
            prev = p
        blocks.append((str(chrom), start - 1, prev + 1))
    return blocks


def fetch_region(chrom: str, start0: int, end: int, retries: int = 3) -> list[dict]:
    """gnomAD variants in one region (gnomAD region coords are 1-based)."""
    s = requests.Session()
    s.trust_env = False
    last_exc: Exception | None = None
    for i in range(retries):
        try:
            r = s.post(
                API,
                json={"query": QUERY,
                      "variables": {"chrom": str(chrom), "start": start0 + 1, "stop": end}},
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            if r.status_code == 200:
                j = r.json()
                if j.get("data", {}).get("region") is not None:
                    return j["data"]["region"]["variants"]
                if j.get("errors"):
                    raise RuntimeError(str(j["errors"][0].get("message"))[:200])
            last_exc = RuntimeError(f"HTTP {r.status_code}")
            time.sleep(2 * (i + 1))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"gnomAD region fetch failed for {chrom}:{start0}-{end}") from last_exc


def _popmax(seq_obj: dict | None) -> float | None:
    if not seq_obj:
        return None
    best = None
    for p in seq_obj.get("populations") or []:
        if p["id"].lower() in POPMAX_IDS and p.get("an"):
            af = p["ac"] / p["an"]
            best = af if best is None else max(best, af)
    return best


def variant_af(variant: dict, metric: str) -> float | None:
    """Raw (unflipped) AF per the companion definitions."""
    gen, exo = variant.get("genome"), variant.get("exome")
    if metric == "global":
        afs = [x.get("af") for x in (gen, exo) if x and x.get("af") is not None]
        return max(afs) if afs else None
    pms = [x for x in (_popmax(gen), _popmax(exo)) if x is not None]
    return max(pms) if pms else None


def score_mock(df: pd.DataFrame, col: str) -> pd.Series:
    """Deterministic pseudo-scores (flipped micro-AF scale), for tests."""
    def one(row) -> float:
        d = hashlib.sha256(
            f"mock-{col}:{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}".encode()
        ).hexdigest()
        return -(int(d[:8], 16) % 1000) / 1e6  # -0.001 .. 0 (flipped AF)

    return df.apply(one, axis=1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--metric", required=True, choices=sorted(METRICS))
    ap.add_argument("--limit", type=int, default=None, help="score only first N variants")
    ap.add_argument("--mock", action="store_true", help="offline deterministic scores (tests)")
    ap.add_argument("--cache-dir", default=str(MODEL_DIR / "cache"))
    args = ap.parse_args(argv)

    col = METRICS[args.metric]
    df = pd.read_parquet(args.input)
    required = {"variant_id", "chrom", "pos", "ref", "alt"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"input missing columns: {sorted(missing)}")
    if args.limit:
        df = df.head(args.limit).copy()

    if args.mock:
        df[col] = score_mock(df, col)
    else:
        cache_dir = Path(args.cache_dir) / "regions"
        cache_dir.mkdir(parents=True, exist_ok=True)
        blocks = variant_blocks(df)
        print(f"gnomad ({args.metric}): {len(blocks)} region block(s)", flush=True)
        lut: dict = {}
        for i, (chrom, start0, end) in enumerate(blocks, 1):
            cache_path = cache_dir / f"{chrom}_{start0}_{end}.json"
            if cache_path.exists():
                variants = json.loads(cache_path.read_text())
            else:
                variants = fetch_region(chrom, start0, end)
                cache_path.write_text(json.dumps(variants))
            for v in variants:
                c, p, ref, alt = v["variant_id"].split("-")
                af = variant_af(v, args.metric)
                if af is not None:
                    lut[(c, int(p), ref, alt)] = -af  # FLIP: larger = more damaging
            print(f"  block {i}/{len(blocks)} chr{chrom}: {len(variants)} gnomAD variants",
                  flush=True)
            time.sleep(0.3)
        df[col] = df.apply(
            lambda r: lut.get((str(r["chrom"]), int(r["pos"]), str(r["ref"]), str(r["alt"]))),
            axis=1,
        )

    out = df.dropna(subset=[col])
    out.to_parquet(args.output, index=False)

    run_log = {
        "model": f"gnomAD v4 AF baseline ({args.metric})",
        "dataset": DATASET,
        "api": API,
        "score_definition": f"{col} = -AF (flipped: larger = more damaging); "
        "variants absent from gnomAD left unscored",
        "mock": args.mock,
        "scored": len(out),
        "not_in_gnomad": len(df) - len(out),
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / f"run_log_{col}.json").write_text(json.dumps(run_log, indent=2) + "\n")
    print(f"{col}: scored {len(out)}/{len(df)} "
          f"({len(df) - len(out)} absent from gnomAD) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
