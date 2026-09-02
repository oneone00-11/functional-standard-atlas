"""Quantities the manuscript cites that live in no results/ file.

The guardrail suite size is the clear case: the manuscript says the pipeline is
covered by N tests, and N appears nowhere in `results/`. Left out of the number
checker's pool, a stale N does not fail -- it matches some unrelated pipeline
value by coincidence and is reported as verified, which is worse than no check
at all. Both 102 (the previous count) and 118 (the current one) match exactly
one pool value each, so the coincidence is not hypothetical.

    python -m atlas.pipeline_facts --write
    python -m atlas.pipeline_facts --check
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FACTS = REPO / "manifests" / "pipeline_facts.json"

NOTE = ("Quantities the manuscript cites that are derived from the pipeline but appear "
        "inside no results/ file. Regenerate with `make facts`. "
        "atlas.check_manuscript_numbers compares the manuscript's stated guardrail count "
        "against guardrail_tests_collected rather than looking it up in the value pool: "
        "both 102 and 118 -- the counts this manuscript carried before -- match exactly "
        "one unrelated pipeline value each, so a stale count would be reported as "
        "verified. How many tests PASS depends on the checkout (a release archive ships "
        "data/ and results/, a git archive does not), so the passing count must be "
        "measured from the release at release time.")


def collected_tests() -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", str(REPO / "tests"),
                          "--collect-only", "-q"],
                         capture_output=True, text=True, cwd=REPO,
                         env={**__import__("os").environ,
                              "PYTHONPATH": str(REPO / "src")}).stdout
    m = re.search(r"(\d+) tests? collected", out)
    if not m:
        sys.exit("[facts] could not read a collected-test count from pytest")
    return int(m.group(1))


def load() -> dict:
    return json.loads(FACTS.read_text()) if FACTS.exists() else {}


def write() -> dict:
    d = load()
    skipped = int(d.get("guardrail_tests_skipped_without_denylist", 1))
    n = collected_tests()
    d = {
        "note": NOTE,
        "guardrail_tests_collected": n,
        "guardrail_tests_passing": n - skipped,
        "guardrail_tests_skipped_without_denylist": skipped,
        # Measured by running the suite from a clean extract of the release
        # archive -- the only number that makes the manuscript's "from a clean
        # extract" true. Re-measure at every release; `--write` preserves it.
        "guardrail_tests_passing_from_archive":
            int(d.get("guardrail_tests_passing_from_archive", 0)) or None,
        "archive_measured_for_version":
            d.get("archive_measured_for_version"),
        "tests_covering_the_analyses_introduced_here":
            int(d.get("tests_covering_the_analyses_introduced_here", 26)),
    }
    FACTS.parent.mkdir(parents=True, exist_ok=True)
    FACTS.write_text(json.dumps(d, indent=1) + "\n")
    return d


def check() -> list[str]:
    d = load()
    if not d:
        return [f"{FACTS} is missing"]
    n = collected_tests()
    problems = []
    if d.get("guardrail_tests_collected") != n:
        problems.append(f"collected {n}, recorded {d.get('guardrail_tests_collected')}")
    expect = n - int(d.get("guardrail_tests_skipped_without_denylist", 1))
    if d.get("guardrail_tests_passing") != expect:
        problems.append(f"passing should be {expect}, recorded {d.get('guardrail_tests_passing')}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.write:
        print(f"[facts] {json.dumps(write(), indent=1)}")
        return
    problems = check()
    if problems:
        print("[facts] stale:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print(f"[facts] OK -- {load()['guardrail_tests_collected']} tests collected")


if __name__ == "__main__":
    main()
