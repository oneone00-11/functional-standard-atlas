"""AlphaGenome by its official definition, for every variant of the frozen matrix.

The matrix's alphagenome column (models/alphagenome/score.py: client 0.7.0, a 16 kb
interval) is an older definition. The official one is the definition the companion
benchmark's release uses (tag ``v3.1.0-submission``): models/alphagenome/score_official.py,
which A2 ported and which scored the external genes to the bit of the release. Here the
same scorer, unchanged and in its own environment (CONVENTIONS.md rule 2), scores all
64,178 variants of data/frozen/frozen-matrix-v1.parquet; this module writes its input and
assembles what it wrote.

Subcommands (run from the repository root, PYTHONPATH=src):

  companion  copy the release files needed here beyond atlas.external_scores's
             (alphagenome_official/all_scores.tsv and its provenance record) with
             ``git show``, read-only, into data/raw/companion_v3.1.0/, registered in
             its MANIFEST.json as the others are
  inputs     data/raw/alphagenome_official_atlas_v1/score_input.parquet: every variant
             of the frozen matrix (variant_id, gene, chrom, pos, ref, alt, kind), the
             SNVs first and then the deletions, insertions and complex variants, each
             part in the matrix's order; trial_input.parquet: 20 of them drawn with a
             fixed seed (10 SNVs; 10 others: a deletion, an insertion and a complex
             variant at +-1-2 bp, then 7 at random); inputs.json: both files' sha256
  assemble   results/alphagenome_official_scores_v1.parquet (one row per variant of
             the matrix; a variant the scorer could not score keeps NaN, nothing is
             filled in), its provenance record, and
             results/alphagenome_official_concordance_v1.tsv:
               release  the release analysis set's 8,853 SNVs, the three components and
                        the merged score against the release's all_scores.tsv (read
                        round-trip), row by row, to the bit;
               v5       Spearman rho with the matrix's older column
                        (results/alphagenome_scores.parquet), overall and by gene x
                        stratum, for every variant and for the SNVs; reported only

Run, in this order (see models/alphagenome/NOTES.md):
    PYTHONPATH=src .venv/bin/python -m atlas.alphagenome_official companion
    PYTHONPATH=src .venv/bin/python -m atlas.alphagenome_official inputs
    S=models/alphagenome/score_official.py; P=models/alphagenome/.venv/bin/python
    $P $S --check-start --out models/alphagenome/cache/official_atlas_v1
    $P $S --external --input data/raw/alphagenome_official_atlas_v1/trial_input.parquet \\
        --out models/alphagenome/cache/official_atlas_v1_trial
    $P $S --external --input data/raw/alphagenome_official_atlas_v1/score_input.parquet \\
        --out models/alphagenome/cache/official_atlas_v1
    $P $S --check-end --out models/alphagenome/cache/official_atlas_v1
    PYTHONPATH=src .venv/bin/python -m atlas.alphagenome_official assemble

Research use only: AlphaGenome's outputs are not used to train any other model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from atlas.evaluate import classify_region
from atlas.external_scores import COMPANION_DIR, DOI, TAG, TAG_COMMIT, _git
from atlas.manifest import register_file, sha256_file

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
FROZEN = REPO / "data" / "frozen" / "frozen-matrix-v1.parquet"
WORK = REPO / "data" / "raw" / "alphagenome_official_atlas_v1"
SCORE_INPUT = WORK / "score_input.parquet"
TRIAL_INPUT = WORK / "trial_input.parquet"
INPUTS_RECORD = WORK / "inputs.json"
MODEL = REPO / "models" / "alphagenome"
SCORER = MODEL / "score_official.py"
SCORER_SHA256_PREFIX = "3bb934200a35624f"      # A2's commit; unchanged here
OUT = MODEL / "cache" / "official_atlas_v1"
TRIAL_OUT = MODEL / "cache" / "official_atlas_v1_trial"
V5 = RESULTS / "alphagenome_scores.parquet"

OUT_TABLE = RESULTS / "alphagenome_official_scores_v1.parquet"
OUT_PROV = RESULTS / "alphagenome_official_scores_v1.provenance.json"
OUT_CONC = RESULTS / "alphagenome_official_concordance_v1.tsv"

AG_DIR = "phase1/data/evidence/alphagenome_official"
ANALYSIS_SET = "phase1/data/evidence/analysis_set_v1.parquet"
FILES = (f"{AG_DIR}/all_scores.tsv", f"{AG_DIR}/all_scores.tsv.provenance.json")

COMPONENTS = ["SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS"]
COMPOSITE = "alphagenome_official"
RELEASE_COMPOSITE = "alphagenome_splicing"
COLUMNS = {COMPOSITE: COMPOSITE, "SPLICE_SITES": "alphagenome_splice_sites",
           "SPLICE_SITE_USAGE": "alphagenome_splice_site_usage",
           "SPLICE_JUNCTIONS": "alphagenome_splice_junctions"}
EXPECTED = {"variants": 64178, "snv": 46392, "other": 17786, "release_set": 8853}
SEED = 20261124
TRIAL = {"snv": 10, "other": 10, "random_other": 7}
KINDS = ("snv", "deletion", "insertion", "complex")
STRATA = ("coding_or_utr", "splice_1_2", "splice_3_10", "splice_11_50", "splice_deep")
GAP_MINUTES = 10        # a longer pause between two scoring times begins a segment


# ==========================================================================
# the release's files
# ==========================================================================
def fetch_companion(out: Path = COMPANION_DIR, files=FILES) -> dict:
    """The release files this module reads, copied with git show (no optional locks:
    nothing is written in the companion checkout), each registered with its release
    path, SHA-256, tag, commit and DOI."""
    commit = _git("rev-parse", f"{TAG}^{{commit}}").decode().strip()
    if commit != TAG_COMMIT:
        raise SystemExit(f"{TAG} points to {commit}, not {TAG_COMMIT}")
    for rel in files:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_git("show", f"{TAG}:{rel}"))
        register_file(out / "MANIFEST.json", dest, release_path=rel, tag=TAG,
                      commit=TAG_COMMIT, doi=DOI, read_with=f"git show {TAG}:{rel}")
    return {"files": list(files), "tag": TAG, "commit": commit, "doi": DOI}


def _rel(path: Path) -> str:
    """A path relative to the repository where it lies inside it, else as it stands."""
    path = Path(path)
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def companion(rel: str, base: Path = COMPANION_DIR) -> Path:
    path = base / rel
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m atlas.alphagenome_official "
                                "companion` (and atlas.external_scores companion)")
    return path


# ==========================================================================
# the input
# ==========================================================================
def kind(ref: str, alt: str) -> str:
    """snv, deletion (ref = alt + more), insertion (alt = ref + more) or complex."""
    if len(ref) == 1 and len(alt) == 1:
        return "snv"
    if len(ref) > len(alt) and ref.startswith(alt):
        return "deletion"
    if len(alt) > len(ref) and alt.startswith(ref):
        return "insertion"
    return "complex"


def score_input(matrix: pd.DataFrame) -> pd.DataFrame:
    """Every variant, SNVs first, then the others; each part in the matrix's order."""
    t = matrix[["variant_id", "gene", "chrom", "pos", "ref", "alt"]].copy()
    t["chrom"], t["pos"] = t["chrom"].astype(str), t["pos"].astype("int64")
    t["kind"] = [kind(r, a) for r, a in zip(t["ref"], t["alt"])]
    order = np.argsort((t["kind"] != "snv").to_numpy(), kind="stable")
    return t.iloc[order].reset_index(drop=True)


def trial_input(t: pd.DataFrame, hgvs_c: pd.Series, seed: int = SEED) -> pd.DataFrame:
    """TRIAL['snv'] SNVs at random; among the others, one deletion, one insertion and
    one complex variant at +-1-2 bp (classify_region on the c. name), then
    TRIAL['random_other'] more at random. hgvs_c is indexed by variant_id."""
    rng = np.random.default_rng(seed)
    region = t["variant_id"].map(hgvs_c).map(classify_region)
    snv = t.index[t["kind"] == "snv"]
    picks = list(np.sort(rng.choice(snv, TRIAL["snv"], replace=False)))
    canonical = t[(t["kind"] != "snv") & (region == "splice_1_2")]
    others = []
    for k in ("deletion", "insertion", "complex"):
        pool = canonical.index[canonical["kind"] == k]
        if not len(pool):
            raise SystemExit(f"no {k} at +-1-2 bp to try")
        others.append(int(pool[rng.integers(len(pool))]))
    rest = t.index[(t["kind"] != "snv") & ~t.index.isin(others)]
    others += list(np.sort(rng.choice(rest, TRIAL["random_other"], replace=False)))
    out = t.loc[picks + others].copy()
    out["region"] = region.loc[out.index]
    out["trial_role"] = (["random_snv"] * TRIAL["snv"]
                         + ["canonical_deletion", "canonical_insertion", "canonical_complex"]
                         + ["random_other"] * TRIAL["random_other"])
    return out.reset_index(drop=True)


def write_inputs(frozen: Path = FROZEN, work: Path = WORK) -> dict:
    m = pd.read_parquet(frozen)
    t = score_input(m)
    n = {"variants": len(t), "snv": int((t["kind"] == "snv").sum()),
         "other": int((t["kind"] != "snv").sum())}
    if any(n[k] != EXPECTED[k] for k in n):
        raise SystemExit(f"the frozen matrix gives {n}, not {EXPECTED}")
    trial = trial_input(t, m.set_index("variant_id")["hgvs_c"])
    work.mkdir(parents=True, exist_ok=True)
    t.to_parquet(work / SCORE_INPUT.name, index=False)
    trial.to_parquet(work / TRIAL_INPUT.name, index=False)
    rec = {"frozen_matrix": str(frozen.relative_to(REPO)) if frozen.is_relative_to(REPO)
           else str(frozen), "frozen_matrix_sha256": sha256_file(frozen),
           "score_input": {"path": str((work / SCORE_INPUT.name).relative_to(REPO))
                           if work.is_relative_to(REPO) else str(work / SCORE_INPUT.name),
                           "sha256": sha256_file(work / SCORE_INPUT.name), **n,
                           "by_kind": {k: int((t["kind"] == k).sum()) for k in KINDS},
                           "order": "the SNVs, then the others; each in the matrix's order"},
           "trial_input": {"path": str((work / TRIAL_INPUT.name).relative_to(REPO))
                           if work.is_relative_to(REPO) else str(work / TRIAL_INPUT.name),
                           "sha256": sha256_file(work / TRIAL_INPUT.name), "seed": SEED,
                           "draw": "numpy.random.default_rng(seed): 10 SNVs without "
                                   "replacement; then one deletion, one insertion and one "
                                   "complex variant at +-1-2 bp; then 7 other non-SNVs",
                           "variants": trial[["variant_id", "gene", "kind", "region",
                                              "trial_role"]].to_dict("records")}}
    (work / INPUTS_RECORD.name).write_text(json.dumps(rec, indent=1) + "\n")
    return rec


# ==========================================================================
# reading the scorer's tables
# ==========================================================================
def read_scores(path: Path) -> pd.DataFrame:
    """A table score_official.py wrote (values with repr), read back to the bit."""
    t = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for c in COMPONENTS + [COMPOSITE, RELEASE_COMPOSITE]:
        if c in t.columns:
            t[c] = t[c].map(lambda x: float(x) if x != "" else np.nan)
    return t


def read_release(path: Path) -> pd.DataFrame:
    """The release's all_scores.tsv, read round-trip: pandas' default parser can be one
    unit in the last place away."""
    return pd.read_csv(path, sep="\t", dtype={"variant_id": str},
                       float_precision="round_trip")


def bitwise_equal(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Equal as float64, NaN equal to NaN."""
    x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    return (x == y) | (np.isnan(x) & np.isnan(y))


def trial_report(path: Path, trial_input: Path = TRIAL_INPUT) -> dict:
    """The trial's checks: every variant scored, no raw score below zero, the merged
    score finite, and a row from each scorer."""
    t = read_scores(path)
    kinds = pd.read_parquet(trial_input).set_index("variant_id")["kind"]
    out = {"variants": int(len(t)), "by_kind": t["variant_id"].map(kinds).value_counts().to_dict()}
    out["negative_raw"] = {c: int((pd.to_numeric(t[f"min_raw_{c}"], errors="coerce") < 0).sum())
                           for c in COMPONENTS}
    out["merged_not_finite"] = int((~np.isfinite(t[COMPOSITE].to_numpy(dtype=float))).sum())
    out["no_rows"] = {c: int((t[f"rows_{c}"].astype(int) == 0).sum()) for c in COMPONENTS}
    out["passed"] = (not any(out["negative_raw"].values()) and out["merged_not_finite"] == 0)
    return out


def segments(scored_utc: pd.Series, gap_minutes: float = GAP_MINUTES) -> list[dict]:
    """The run's segments, read from the variants' scoring times: a pause of more than
    gap_minutes between two consecutive times begins a new one (a run stopped and resumed
    from the cache). Each with its count and its first and last scoring time."""
    t = pd.Series(pd.to_datetime(scored_utc, utc=True)).sort_values().reset_index(drop=True)
    seg = (t.diff() > pd.Timedelta(minutes=gap_minutes)).cumsum()
    return [{"segment": int(k) + 1, "n": int(len(p)), "first_scored_utc": p.iloc[0].isoformat(),
             "last_scored_utc": p.iloc[-1].isoformat()} for k, p in t.groupby(seg)]


# ==========================================================================
# assembly
# ==========================================================================
def build_table(inp: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """One row per variant of the input, in the frozen matrix's order of genes and rows
    as the input holds them; the scorer's four values under this repository's names."""
    s = scores.set_index("variant_id")
    t = inp[["variant_id", "gene", "kind"]].copy()
    for src, col in COLUMNS.items():
        t[col] = t["variant_id"].map(s[src]) if src in s.columns else np.nan
    t["scored_utc"] = t["variant_id"].map(s["scored_utc"])
    return t


def release_concordance(t: pd.DataFrame, release: pd.DataFrame,
                        analysis_set: pd.DataFrame) -> list[dict]:
    """The release analysis set's variants, row by row, to the bit."""
    ids = analysis_set["variant_id"]
    r = release.set_index("variant_id")
    ours = t.set_index("variant_id")
    rows = []
    for src, col in COLUMNS.items():
        rel_col = RELEASE_COMPOSITE if src == COMPOSITE else src
        a = ours.reindex(ids)[col]
        b = r.reindex(ids)[rel_col]
        same = bitwise_equal(a, b)
        diff = np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
        rows.append({"comparison": "release", "column": col, "against": rel_col,
                     "scope": "release_analysis_set_snv", "gene": "all", "stratum": "all",
                     "n": int(len(ids)), "n_ours_missing": int(a.isna().sum()),
                     "n_release_missing": int(b.isna().sum()),
                     "n_identical": int(same.sum()), "n_differing": int((~same).sum()),
                     "max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0,
                     "spearman_rho": np.nan, "holds": bool(same.all())})
    return rows


def v5_concordance(t: pd.DataFrame, v5: pd.DataFrame, hgvs_c: pd.Series) -> list[dict]:
    """Spearman rho of the official merged score with the matrix's older column, overall
    and by gene x stratum, for every variant and for the SNVs. Reported only."""
    d = t[["variant_id", "gene", "kind", COMPOSITE]].merge(
        v5[["variant_id", "alphagenome"]], on="variant_id", how="left")
    d["stratum"] = d["variant_id"].map(hgvs_c).map(classify_region)
    rows = []
    for scope, part in (("all", d), ("snv", d[d["kind"] == "snv"])):
        cells = [("all", "all", part)]
        cells += [(g, "all", p) for g, p in part.groupby("gene", sort=True)]
        cells += [("all", s, part[part["stratum"] == s]) for s in STRATA]
        cells += [(g, s, p[p["stratum"] == s]) for g, p in part.groupby("gene", sort=True)
                  for s in STRATA]
        for gene, stratum, p in cells:
            both = p[[COMPOSITE, "alphagenome"]].dropna()
            rho = (float(spearmanr(both[COMPOSITE], both["alphagenome"])[0])
                   if len(both) >= 3 and both[COMPOSITE].nunique() > 1
                   and both["alphagenome"].nunique() > 1 else np.nan)
            rows.append({"comparison": "v5", "column": COMPOSITE, "against": "alphagenome",
                         "scope": scope, "gene": gene, "stratum": stratum, "n": int(len(p)),
                         "n_ours_missing": int(p[COMPOSITE].isna().sum()),
                         "n_release_missing": int(p["alphagenome"].isna().sum()),
                         "n_identical": np.nan, "n_differing": np.nan, "max_abs_diff": np.nan,
                         "spearman_rho": rho, "n_both": int(len(both)), "holds": np.nan})
    return rows


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def provenance(t: pd.DataFrame, conc: pd.DataFrame, out: Path = OUT,
               trial_out: Path = TRIAL_OUT) -> dict:
    ext = _read_json(out / "external_scores.tsv.provenance.json")
    start = _read_json(out / "check_start_scores.tsv.provenance.json")
    end = _read_json(out / "check_end_scores.tsv.provenance.json")
    trial = _read_json(trial_out / "external_scores.tsv.provenance.json")
    inputs = _read_json(INPUTS_RECORD)
    env = ext["environment"]
    scorer_sha = sha256_file(SCORER)
    segs = segments(t["scored_utc"])
    return {
        "written_by": "atlas.alphagenome_official assemble",
        "output": _rel(OUT_TABLE), "output_sha256": sha256_file(OUT_TABLE),
        "rows": int(len(t)),
        "columns": {col: ("the notebook's merged splicing score (alphagenome_splicing)"
                          if src == COMPOSITE else f"the {src} scorer's maximum absolute "
                          "raw score") for src, col in COLUMNS.items()},
        "definition": {"file": ext["definition"], "sha256": ext["definition_sha256"],
                       "code_run": ext["code_run"]},
        "scorer": {"file": _rel(SCORER), "sha256": scorer_sha,
                   "sha256_matches_A2": scorer_sha.startswith(SCORER_SHA256_PREFIX),
                   "hashed_when": "when this record is assembled"},
        "client": env.get("client"),
        "model_version_requested": env.get("model_version_requested"),
        "default_when_not_requested": env.get("default_when_not_requested"),
        "model_version_in_response": env.get("model_version_in_response"),
        "response_metadata_keys": ext.get("response_metadata_keys"),
        "input": inputs["score_input"] | {"frozen_matrix_sha256": inputs["frozen_matrix_sha256"]},
        "run": {k: ext.get(k) for k in ("start_utc", "end_utc", "seconds_scoring", "n_variants",
                                        "n_scored_this_run", "n_from_cache", "scored_utc_range",
                                        "calls", "calls_note", "retries", "failures",
                                        "workers", "n_by_gene", "summary")},
        "segments": {"rule": f"a pause of more than {GAP_MINUTES} minutes between two "
                             "consecutive scoring times begins a segment",
                     "segments": segs}
        | ({"calls_and_retries": "recorded by the scorer for the segment that completed the "
                                 "run only (run, above); a segment stopped before the end "
                                 "leaves no record"} if len(segs) > 1 else {}),
        "run_records": {m: _rel(out / f"{m}_scores.tsv.provenance.json")
                        for m in ("check_start", "external", "check_end")},
        "unscored": {col: int(t[col].isna().sum()) for col in COLUMNS.values()},
        "trial": {"input": inputs["trial_input"], "report": trial_report(
            trial_out / "external_scores.tsv"),
            "calls": trial.get("calls"), "retries": trial.get("retries"),
            "failures": trial.get("failures"), "seconds_scoring": trial.get("seconds_scoring")},
        "check_start": {k: v for k, v in start.items() if k.startswith("against")}
        | {"scored_utc_range": start.get("scored_utc_range")},
        "check_end": {k: v for k, v in end.items()
                      if k.startswith("against") or k == "after_external"}
        | {"scored_utc_range": end.get("scored_utc_range")},
        "concordance": {"file": _rel(OUT_CONC),
                        "release_holds": bool(conc.loc[conc["comparison"] == "release",
                                                       "holds"].all())},
        "release": {"tag": TAG, "commit": TAG_COMMIT, "doi": DOI,
                    "files": {rel: sha256_file(companion(rel)) for rel in FILES + (ANALYSIS_SET,)}},
        "pip_freeze": env.get("pip_freeze"), "python": env.get("python"),
        "requirements": env.get("requirements"),
        "requirements_sha256": env.get("requirements_sha256"),
        "licence": "AlphaGenome Terms of Service: non-commercial research use only; the "
                   "outputs are not used to train any other model",
    }


def assemble(out: Path = OUT) -> dict:
    inp = pd.read_parquet(SCORE_INPUT)
    scores = read_scores(out / "external_scores.tsv")
    t = build_table(inp, scores)
    m = pd.read_parquet(FROZEN, columns=["variant_id", "hgvs_c"]).set_index("variant_id")
    order = pd.read_parquet(FROZEN, columns=["variant_id"])["variant_id"]
    t = t.set_index("variant_id").loc[order].reset_index()   # the frozen matrix's order
    s = pd.read_parquet(companion(ANALYSIS_SET), columns=["variant_id", "chrom", "pos", "ref", "alt"])
    if len(s) != EXPECTED["release_set"] or not s["variant_id"].isin(t["variant_id"]).all():
        raise SystemExit("the release analysis set is not 8,853 variants of the matrix")
    release = read_release(companion(FILES[0]))
    rows = release_concordance(t, release, s) + v5_concordance(t, pd.read_parquet(V5),
                                                                m["hgvs_c"])
    conc = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    t.drop(columns=["kind"]).to_parquet(OUT_TABLE, index=False)
    conc.to_csv(OUT_CONC, sep="\t", index=False, float_format="%.6f")
    prov = provenance(t, conc, out)
    OUT_PROV.write_text(json.dumps(prov, indent=1, ensure_ascii=False, default=str) + "\n")
    held = prov["concordance"]["release_holds"]
    print(f"wrote {_rel(OUT_TABLE)} ({len(t)} rows), {_rel(OUT_CONC)};"
          f" release concordance {'holds' if held else 'DOES NOT hold'}")
    if not held:
        raise SystemExit("a release analysis-set variant differs: stopping")
    return prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["companion", "inputs", "assemble"])
    args = ap.parse_args(argv)
    if args.command == "companion":
        print(json.dumps(fetch_companion(), indent=1))
    elif args.command == "inputs":
        rec = write_inputs()
        print(json.dumps({k: (v if k != "trial_input" else {x: y for x, y in v.items()
                                                            if x != "variants"})
                          for k, v in rec.items()}, indent=1))
    else:
        assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())
