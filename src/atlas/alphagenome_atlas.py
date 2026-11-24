"""AlphaGenome Atlas's four columns for the atlas's SNVs and the external genes'.

v5 has no AlphaGenome Atlas column. The companion release (tag ``v3.1.0-submission``)
has four, avi, avi_splice_sites, avi_splice_site_usage and avi_splice_junctions, scored
with client 0.9.0. The committed scorer models/alphagenome_atlas/score.py (the release's
run() and guards, ported) scores here the 46,392 SNVs of the frozen matrix and the 5,095
intronic SNVs of the external genes (results/external_scores_v1.parquet): 51,487. Non-SNVs
are not queried: the Atlas is precomputed for SNVs.

The Atlas's merged score was selected on a validation set that includes BRCA1, RAD51C
and DDX3X experiments (the companion paper's reference 25); BRCA1 and RAD51C are core
genes here. The provenance record says so; how the rebuild treats it is decided there.

Subcommands (from the repository root, PYTHONPATH=src):

  companion  copy the release files this needs with ``git show``, read-only, into
             data/raw/companion_v3.1.0/ (registered in its MANIFEST.json): the ported
             code, the columns check, the scorer list, the anchors and the two score
             files compared with, each with its records
  inputs     data/raw/alphagenome_atlas_v1/score_input.parquet: variant_id, gene, set
             (core or external), chrom, pos, ref, alt; the core SNVs in the matrix's
             order, then the external ones in their table's order
  assemble   results/alphagenome_atlas_scores_v1.parquet (51,487 rows: variant_id, gene,
             set and the four columns), its provenance record, and
             results/alphagenome_atlas_concordance_v1.tsv: the four columns, row by row
             and to the bit, against the release's avi.parquet (its 8,853 analysis-set
             rows) and external/newgenes_avi.parquet (4,522 rows), and the anchors' check

Research use only: the Atlas's scores are not used to train any other model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.alphagenome_official import _rel, fetch_companion
from atlas.external_scores import COMPANION_DIR, DOI, TAG, TAG_COMMIT
from atlas.manifest import sha256_file

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
FROZEN = REPO / "data" / "frozen" / "frozen-matrix-v1.parquet"
EXTERNAL = RESULTS / "external_scores_v1.parquet"
WORK = REPO / "data" / "raw" / "alphagenome_atlas_v1"
SCORE_INPUT = WORK / "score_input.parquet"
MODEL = REPO / "models" / "alphagenome_atlas"
SCORER = MODEL / "score.py"
OUT = MODEL / "cache" / "atlas_v1"
ANCHORS_OUT = MODEL / "cache" / "anchors_v1"

OUT_TABLE = RESULTS / "alphagenome_atlas_scores_v1.parquet"
OUT_PROV = RESULTS / "alphagenome_atlas_scores_v1.provenance.json"
OUT_CONC = RESULTS / "alphagenome_atlas_concordance_v1.tsv"

EV = "phase1/data/evidence"
AVI = f"{EV}/avi.parquet"
NEWGENES = f"{EV}/external/newgenes_avi.parquet"
ANCHORS = f"{EV}/external/e18_anchors_avi.parquet"
FILES = (
    "phase1/src/evid_score_avi.py", "scripts/stage_E18/run_avi.py", "scripts/stage_E18/_e18.py",
    f"{EV}/alphagenome_official/ATLAS_COLUMNS_CHECK.json", f"{EV}/avi_scorers_20260920.json",
    AVI, f"{AVI}.provenance.json",
    NEWGENES, f"{NEWGENES}.provenance.json", f"{NEWGENES}.environment.json",
    ANCHORS, f"{ANCHORS}.provenance.json", f"{ANCHORS}.environment.json",
)
COLUMNS = ["avi", "avi_splice_sites", "avi_splice_site_usage", "avi_splice_junctions"]
EXPECTED = {"core": 46392, "external": 5095, "variants": 51487, "analysis": 8853,
            "newgenes": 4522, "anchors": 39}
DEPENDENCY = ("The Atlas's merged score (AVI) was selected on a validation set that "
              "includes BRCA1, RAD51C and DDX3X experiments (the companion paper's reference "
              "25); BRCA1 and RAD51C are core genes of this atlas. Recorded here; how the "
              "rebuild treats it is decided there.")
LICENCE = ("AlphaGenome Terms of Service: non-commercial research use only; the outputs "
           "are not used to train any other model")


def companion(rel: str, base: Path = COMPANION_DIR) -> Path:
    path = base / rel
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m atlas.alphagenome_atlas "
                                "companion`")
    return path


def is_snv(ref: pd.Series, alt: pd.Series) -> pd.Series:
    return (ref.str.len() == 1) & (alt.str.len() == 1)


def score_input(matrix: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    core = matrix[is_snv(matrix["ref"], matrix["alt"])][
        ["variant_id", "gene", "chrom", "pos", "ref", "alt"]].assign(set="core")
    ext = external[is_snv(external["ref"], external["alt"])][
        ["variant_id", "gene", "chrom", "pos", "ref", "alt"]].assign(set="external")
    t = pd.concat([core, ext], ignore_index=True)
    t["chrom"], t["pos"] = t["chrom"].astype(str), t["pos"].astype("int64")
    return t[["variant_id", "gene", "set", "chrom", "pos", "ref", "alt"]]


def write_inputs(work: Path = WORK) -> dict:
    t = score_input(pd.read_parquet(FROZEN), pd.read_parquet(EXTERNAL))
    n = {"core": int((t["set"] == "core").sum()), "external": int((t["set"] == "external").sum()),
         "variants": int(len(t))}
    if any(n[k] != EXPECTED[k] for k in n) or not t["variant_id"].is_unique:
        raise SystemExit(f"the input is {n} (unique ids: {t['variant_id'].is_unique}), "
                         f"not {EXPECTED}")
    work.mkdir(parents=True, exist_ok=True)
    t.to_parquet(work / SCORE_INPUT.name, index=False)
    rec = {"score_input": _rel(work / SCORE_INPUT.name),
           "sha256": sha256_file(work / SCORE_INPUT.name), **n,
           "sources": {"frozen_matrix": sha256_file(FROZEN), "external": sha256_file(EXTERNAL)},
           "order": "the core SNVs in the matrix's order, then the external SNVs"}
    (work / "inputs.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def bitwise(a: pd.Series, b: pd.Series) -> np.ndarray:
    x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    return (x == y) | (np.isnan(x) & np.isnan(y))


def concordance(t: pd.DataFrame, avi: pd.DataFrame, newgenes: pd.DataFrame) -> pd.DataFrame:
    """The four columns against the release's, row by row (by variant_id)."""
    rows = []
    ours = t.set_index("variant_id")
    for name, ref in (("release_analysis_set", avi[avi["set"] == "analysis"]),
                      ("release_newgenes", newgenes)):
        r = ref.set_index("variant_id")
        mine = ours.reindex(r.index)
        for c in COLUMNS:
            same = bitwise(mine[c], r[c])
            diff = np.abs(mine[c].to_numpy(dtype=float) - r[c].to_numpy(dtype=float))
            rows.append({"comparison": name, "column": c, "n": int(len(r)),
                         "n_ours_missing": int(mine[c].isna().sum()),
                         "n_identical": int(same.sum()), "n_differing": int((~same).sum()),
                         "max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0,
                         "holds": bool(same.all())})
    return pd.DataFrame(rows)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def home_as_tilde(env: dict, home: Path | None = None) -> dict:
    """The scorer's environment record with the home directory written as ~ (in the
    interpreter's path, and in any other string or list of strings), so the record in
    results/ names no local account; the scorer's own record is left as written."""
    h = str(home or Path.home()) + "/"

    def tilde(v):
        if isinstance(v, str) and v.startswith(h):
            return "~/" + v[len(h):]
        return [tilde(x) for x in v] if isinstance(v, list) else v
    return {k: tilde(v) for k, v in env.items()}


def assemble(out: Path = OUT, anchors_out: Path = ANCHORS_OUT) -> dict:
    inp = pd.read_parquet(SCORE_INPUT)
    got = pd.read_parquet(out / "atlas_scores.parquet")
    if not inp["variant_id"].equals(got["variant_id"]):
        raise SystemExit("the scorer's output is not the input's rows in order")
    t = inp[["variant_id", "gene", "set"]].copy()
    for c in COLUMNS:
        t[c] = got[c].to_numpy()
    avi = pd.read_parquet(companion(AVI))
    newgenes = pd.read_parquet(companion(NEWGENES))
    if (int((avi["set"] == "analysis").sum()), len(newgenes)) != (EXPECTED["analysis"],
                                                                 EXPECTED["newgenes"]):
        raise SystemExit("the release's score files are not 8,853 and 4,522 rows")
    conc = concordance(t, avi, newgenes)
    run = _read_json(out / "atlas_scores.parquet.provenance.json")
    anc = _read_json(anchors_out / "atlas_scores.parquet.provenance.json")
    a = anc["anchors"]
    conc = pd.concat([conc, pd.DataFrame([
        {"comparison": "release_anchors", "column": c, "n": a[c]["n_anchors"],
         "n_ours_missing": a[c]["n_anchors"] - a[c]["n_checked"], "n_identical": a[c]["n_identical"],
         "n_differing": a[c]["n_checked"] - a[c]["n_identical"], "max_abs_diff": np.nan,
         "holds": a[c]["n_identical"] == a[c]["n_anchors"]} for c in COLUMNS])],
        ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    t.to_parquet(OUT_TABLE, index=False)
    conc.to_csv(OUT_CONC, sep="\t", index=False, float_format="%.6g")
    env = home_as_tilde(_read_json(out / "environment.json"))
    prov = {
        "written_by": "atlas.alphagenome_atlas assemble",
        "output": _rel(OUT_TABLE), "output_sha256": sha256_file(OUT_TABLE),
        "rows": int(len(t)), "by_set": t["set"].value_counts().to_dict(),
        "unscored": {c: int(t[c].isna().sum()) for c in COLUMNS},
        "product": run["product"], "client_version": run["client_version"],
        "scorer": {"file": _rel(SCORER), "sha256": sha256_file(SCORER),
                   "ported_from": run["ported_from"]},
        "scorers_requested": run["scorers_requested"],
        "scorer_list": _read_json(out / "scorers.json"),
        "aggregation": run["aggregation"],
        "run": {k: run.get(k) for k in ("segments", "n_requests", "n_requests_note", "calls",
                                        "retries", "n_from_checkpoint", "failed_batches",
                                        "runtime_minutes", "accessed_utc")},
        "anchors": {"run": {k: anc.get(k) for k in ("segments", "calls", "retries",
                                                     "accessed_utc")}, "check": a},
        "concordance": {"file": _rel(OUT_CONC),
                        "holds": bool(conc["holds"].all())},
        "input": _read_json(WORK / "inputs.json"),
        "environment": env,
        "release": {"tag": TAG, "commit": TAG_COMMIT, "doi": DOI,
                    "files": {rel: sha256_file(companion(rel)) for rel in FILES}},
        "dependency": DEPENDENCY,
        "licence": LICENCE, "licence_source": run["licence_source"],
    }
    OUT_PROV.write_text(json.dumps(prov, indent=1, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {_rel(OUT_TABLE)} ({len(t)} rows), "
          f"{_rel(OUT_CONC)}; concordance "
          f"{'holds' if prov['concordance']['holds'] else 'DOES NOT hold'}")
    if not prov["concordance"]["holds"]:
        raise SystemExit("a row differs from the release: stopping")
    return prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["companion", "inputs", "assemble"])
    args = ap.parse_args(argv)
    if args.command == "companion":
        print(json.dumps(fetch_companion(files=FILES), indent=1))
    elif args.command == "inputs":
        print(json.dumps(write_inputs(), indent=1))
    else:
        assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())
