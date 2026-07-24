"""Download MaveDB score sets and register them in the hash manifest.

Usage (with PYTHONPATH=src):

    python -m atlas.fetch_mavedb --config config/assays.yaml
    python -m atlas.fetch_mavedb --urn urn:mavedb:00000097-0-2

Metadata endpoint: GET https://api.mavedb.org/api/v1/score-sets/{urn}
Scores endpoint:   GET https://api.mavedb.org/api/v1/score-sets/{urn}/scores  (CSV)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests
import yaml

from atlas.manifest import register_file

API_BASE = "https://api.mavedb.org/api/v1/score-sets"
DEFAULT_OUT = Path("data/raw/mavedb")
TIMEOUT = 120


def fetch_json(url: str) -> dict:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def fetch_score_set(urn: str, out_dir: Path, gene: str | None = None) -> dict:
    """Download metadata + scores for one URN and register both in the manifest."""
    urn_dir = out_dir / urn.replace(":", "_")
    urn_dir.mkdir(parents=True, exist_ok=True)

    meta = fetch_json(f"{API_BASE}/{urn}")
    title = meta.get("title", "")

    meta_path = urn_dir / "metadata.json"
    meta_path.write_bytes(
        __import__("json").dumps(meta, indent=2, sort_keys=True).encode("utf-8")
    )

    scores_path = urn_dir / "scores.csv"
    scores_path.write_bytes(fetch_bytes(f"{API_BASE}/{urn}/scores"))

    manifest_path = out_dir / "MANIFEST.json"
    for fp in (meta_path, scores_path):
        register_file(
            manifest_path,
            fp,
            urn=urn,
            gene=gene or "",
            title=title,
        )

    n_scores = sum(1 for _ in open(scores_path, "rb")) - 1
    return {"urn": urn, "title": title, "n_score_rows": max(n_scores, 0)}


def load_registry(config_path: str | Path) -> list[dict]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    assays = cfg.get("assays", [])
    return [a for a in assays if a.get("urn")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="assay registry YAML (default: fetch all with urn)")
    ap.add_argument("--urn", help="fetch a single URN (overrides --config)")
    ap.add_argument("--gene", help="gene symbol, only used with --urn")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.urn:
        jobs = [{"urn": args.urn, "gene": args.gene}]
    elif args.config:
        jobs = load_registry(args.config)
    else:
        ap.error("provide --config or --urn")

    print(f"Fetching {len(jobs)} score set(s) into {out_dir}/")
    failures = 0
    for job in jobs:
        urn = job["urn"]
        try:
            info = fetch_score_set(urn, out_dir, gene=job.get("gene"))
            print(f"  OK  {urn}  rows={info['n_score_rows']:>6}  {info['title'][:60]}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"  FAIL {urn}: {exc}", file=sys.stderr)
        time.sleep(0.3)  # be polite to the API

    if failures:
        print(f"{failures} fetch(es) failed", file=sys.stderr)
        return 1
    print("Done. Manifest:", out_dir / "MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
