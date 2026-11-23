"""Scores for the external genes' intronic SNVs: the variants, their checks, the
lookup columns, and the delivered table.

The five external deposits of `atlas.external_reliability` (CTCF, SFPQ, RAD51D, XRCC2,
SBDS) carry 5,095 intronic SNVs. The companion benchmark's release (tag
``v3.1.0-submission``, DOI ``10.5281/zenodo.24243120``) maps them to GRCh38 and scored
the 4,522 at 3-50 bp. Here every one of the 5,095 is scored again, by the scorers under
`models/` (CONVENTIONS.md rule 2), and the release's scores serve only as a check.

Subcommands (run from the repository root, PYTHONPATH=src):

  companion   copy the release's files with ``git show`` from a checkout of the
              companion repository (``ATLAS_COMPANION_REPO``, default
              ~/work/variant-fm-benchmark), read-only (GIT_OPTIONAL_LOCKS=0), into
              data/raw/companion_v3.1.0/, each registered with its release path,
              SHA-256, tag, commit and DOI
  check       the four checks of the variant tables (rule 4), each row by row:
                1  every variant_urn is in the deposit A1 fetched
                   (data/raw/mavedb_external/), with the same score;
                2  atlas.evaluate.classify_region, on the c. name without its
                   transcript prefix, gives the table's stratum;
                3  the deposit's intronic SNVs are exactly the table's rows; its other
                   intronic variants are counted and must all be deletions;
                4  direction: in the deposit's own classification the abnormal class
                   scores lower, so functional_pathogenicity = -score
                   (atlas.mapping's rule) holds;
              then writes the scoring input, data/raw/external_v1/score_input.parquet
  lookups     CADD, GPN-MSA, phyloP, phastCons and gnomAD (global, popmax): the
              committed scorers under models/, each run from a byte-identical copy in
              models/<scorer>/cache/external_v1/, so that its run log does not
              overwrite the record of the core-gene run beside the scorer
  assemble    results/external_scores_v1.parquet, its provenance record and
              results/external_concordance_v1.tsv (the comparison with the release)

SpliceAI, Pangolin and AlphaGenome run in their own environments:
models/spliceai/score_external.py, models/pangolin/score_external.py and
models/alphagenome/score_official.py; `assemble` reads their outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.evaluate import classify_region
from atlas.external_reliability import EXTERNAL, hgvs_c, load_deposit
from atlas.manifest import register_file, sha256_file

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
RAW = REPO / "data" / "raw"
COMPANION_DIR = RAW / "companion_v3.1.0"
WORK = RAW / "external_v1"
SCORE_INPUT = WORK / "score_input.parquet"

TAG = "v3.1.0-submission"
TAG_COMMIT = "29bbbe56b52c766b050bb04ad62c3d37a4314972"
DOI = "10.5281/zenodo.24243120"

GENES = ("CTCF", "SFPQ", "RAD51D", "XRCC2", "SBDS")
URN = {spec["gene"]: urn for urn, spec in EXTERNAL.items() if spec["gene"] in GENES}
EXT_DIR = "phase1/data/evidence/external"
AG_DIR = "phase1/data/evidence/alphagenome_official"

# the instruction's table: rows, then pm12, s3_10, s11_50, deep; and the file's sha256
EXPECTED = {"CTCF": ((1535, 90, 360, 1010, 75), "e565e144697a535a"),
            "SFPQ": ((1121, 102, 396, 623, 0), "e52d293ccf76693a"),
            "RAD51D": ((1464, 108, 429, 867, 60), "7837a3e13ba15c81"),
            "XRCC2": ((468, 24, 96, 294, 54), "c12b4dcaacd7da7b"),
            "SBDS": ((507, 45, 189, 258, 15), "dbee75889e97a69e")}
STRATUM = {"pm12": "splice_1_2", "s3_10": "splice_3_10", "s11_50": "splice_11_50",
           "out_of_scope_deep": "splice_deep"}
INTRONIC = set(STRATUM.values())
SNV_C = re.compile(r"^c\.[-*]?\d+[+-]\d+[ACGT]>[ACGT]$")
DELETION_C = re.compile(r"^c\.\S+del[ACGT]*$")
COMPARISON_RUNS = ("spliceai", "pangolin", "cadd", "gpn_msa", "conservation")
RUN_LOGS = "scripts/stage_E18/run_log"
RELEASE_RUN_LOGS = ("run_spliceai_20261023T082635Z.json",
                    "build_pangolin_refs_20261023T044027Z.json")


class CheckFailed(SystemExit):
    """A check of the variant tables found a row it does not hold for."""


# ==========================================================================
# the release's files
# ==========================================================================
def companion_files() -> list[str]:
    """Release paths copied here: the variant tables, the score files they are
    compared with (each with its provenance and environment records), and what the
    AlphaGenome scorer needs (the notebook source, the external scores, the drift
    check's scores, and the analysis set that locates the drift variants)."""
    files = []
    for g in GENES:
        stem = f"{EXT_DIR}/{g.lower()}"
        files += [f"{stem}_variants.parquet", f"{stem}_variants.parquet.provenance.json"]
        for run in COMPARISON_RUNS:
            files += [f"{stem}_{run}.parquet{s}" for s in ("", ".provenance.json",
                                                            ".environment.json")]
    files += [f"{AG_DIR}/{n}" for n in (
        "NOTEBOOK_SOURCE.json", "external_scores.tsv", "external_scores.tsv.provenance.json",
        "drift_2b_end_scores.tsv", "drift_2b_end_scores.tsv.provenance.json",
        "ENVIRONMENT_2b.json")]
    files += ["phase1/data/evidence/analysis_set_v1.parquet",
              "scripts/stage_E17/requirements.txt", "scripts/stage_E17/score_official.py"]
    files += [f"{RUN_LOGS}/{n}" for n in RELEASE_RUN_LOGS]  # the references' hashes
    return files


def companion_repo() -> Path:
    return Path(os.environ.get("ATLAS_COMPANION_REPO",
                               str(Path.home() / "work" / "variant-fm-benchmark")))


def _git(*args: str) -> bytes:
    """git in the companion checkout, without optional locks: nothing is written there."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.run(["git", "-C", str(companion_repo()), *args], env=env,
                          capture_output=True, check=True).stdout


def fetch_companion(out: Path = COMPANION_DIR) -> dict:
    commit = _git("rev-parse", f"{TAG}^{{commit}}").decode().strip()
    if commit != TAG_COMMIT:
        raise SystemExit(f"{TAG} points to {commit}, not {TAG_COMMIT}")
    manifest = out / "MANIFEST.json"
    for rel in companion_files():
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_git("show", f"{TAG}:{rel}"))
        register_file(manifest, dest, release_path=rel, tag=TAG, commit=TAG_COMMIT, doi=DOI,
                      read_with=f"git show {TAG}:{rel}")
    bad = {g: sha256_file(out / f"{EXT_DIR}/{g.lower()}_variants.parquet")[:16]
           for g in GENES}
    bad = {g: s for g, s in bad.items() if s != EXPECTED[g][1]}
    if bad:
        raise SystemExit(f"variant tables differ from the instruction's: {bad}")
    return {"files": len(companion_files()), "tag": TAG, "commit": commit, "doi": DOI}


def companion(rel: str, base: Path = COMPANION_DIR) -> Path:
    path = base / rel
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m atlas.external_scores companion`")
    return path


def load_variant_tables(base: Path = COMPANION_DIR) -> pd.DataFrame:
    parts = []
    for g in GENES:
        v = pd.read_parquet(companion(f"{EXT_DIR}/{g.lower()}_variants.parquet", base))
        if set(v["gene"]) != {g}:
            raise CheckFailed(f"{g}'s table names genes {sorted(set(v['gene']))}")
        parts.append(v)
    return pd.concat(parts, ignore_index=True)


# ==========================================================================
# the four checks (rule 4)
# ==========================================================================
def counts(v: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    out = {}
    for g, d in v.groupby("gene"):
        s = d["stratum"].value_counts()
        out[g] = (len(d),) + tuple(int(s.get(k, 0)) for k in STRATUM)
    return out


def check_urns_and_scores(v: pd.DataFrame, deposit: pd.DataFrame) -> dict:
    """1: every variant_urn is in the deposit, with the same score (exact)."""
    dep = deposit.set_index("accession")["score"]
    present = v["variant_urn"].isin(dep.index)
    same = present & (v["variant_urn"].map(dep) == v["score"])
    return {"rows": int(len(v)), "missing": v.loc[~present, "variant_urn"].tolist(),
            "score_differs": v.loc[present & ~same, "variant_urn"].tolist()}


def check_strata(v: pd.DataFrame) -> dict:
    """2: classify_region on the bare c. name gives the table's stratum."""
    got = [classify_region(hgvs_c(h, t)) for h, t in zip(v["hgvs_nt"], v["transcript"])]
    want = v["stratum"].map(STRATUM)
    bad = v.loc[want.isna() | (want != pd.Series(got, index=v.index)), "variant_urn"]
    return {"rows": int(len(v)), "differs": bad.tolist()}


def check_completeness(v: pd.DataFrame, deposit: pd.DataFrame) -> dict:
    """3: the deposit's intronic SNVs are exactly the table's rows; its other
    intronic variants are all deletions."""
    intronic = deposit[deposit["region"].isin(INTRONIC)]
    snv = intronic["hgvs_c"].str.match(SNV_C)
    dep_snv, table = set(intronic.loc[snv, "accession"]), set(v["variant_urn"])
    other = intronic.loc[~snv, "hgvs_c"]
    return {"deposit_intronic_snv": len(dep_snv), "table_rows": len(table),
            "only_in_deposit": sorted(dep_snv - table), "only_in_table": sorted(table - dep_snv),
            "other_intronic": int(len(other)),
            "other_intronic_not_deletion": other[~other.str.match(DELETION_C)].tolist()}


def deposit_classes(meta: dict, deposit: pd.DataFrame) -> tuple[str, pd.Series]:
    """The deposit's own classification per variant: 'abnormal', 'normal' or other.
    The investigator-provided functional classes where the record has them (score
    ranges), else the deposit's functional_consequence column."""
    cal = [c for c in meta.get("scoreCalibrations") or []
           if c.get("investigatorProvided") and c.get("title", "").startswith("Investigator-provided")]
    if cal:
        def cls(x):
            for fc in cal[0]["functionalClassifications"]:
                lo, hi = fc["range"]
                above = lo is None or (x >= lo if fc["inclusiveLowerBound"] else x > lo)
                below = hi is None or (x <= hi if fc["inclusiveUpperBound"] else x < hi)
                if above and below:
                    return fc["functionalClassification"]
            return "unclassified"
        return f"scoreCalibrations: {cal[0]['title']}", deposit["score"].map(cls)
    if "functional_consequence" in deposit.columns:
        names = {"functionally_abnormal": "abnormal", "functionally_normal": "normal"}
        return ("scores.csv column functional_consequence",
                deposit["functional_consequence"].map(lambda c: names.get(c, str(c))))
    raise CheckFailed("the deposit carries no classification to check the direction with")


def check_direction(v: pd.DataFrame, deposit: pd.DataFrame, meta: dict) -> dict:
    """4: the abnormal class scores lower, on the whole deposit and on the table's
    rows, so functional_pathogenicity = -score puts abnormal variants higher."""
    source, cls = deposit_classes(meta, deposit)
    dep = deposit.assign(cls=cls.values)
    rows = dep[dep["accession"].isin(set(v["variant_urn"]))]
    out = {"classification": source}
    for name, d in (("deposit", dep), ("table_rows", rows)):
        med = d.groupby("cls")["score"].median()
        n = d["cls"].value_counts()
        out[name] = {"median_abnormal": float(med.get("abnormal", np.nan)),
                     "median_normal": float(med.get("normal", np.nan)),
                     "n_abnormal": int(n.get("abnormal", 0)), "n_normal": int(n.get("normal", 0))}
        out[name]["abnormal_lower"] = bool(out[name]["median_abnormal"] < out[name]["median_normal"])
    return out


def run_checks(base: Path = COMPANION_DIR, deposits: Path | None = None) -> tuple[pd.DataFrame, dict]:
    from atlas import external_reliability as ER
    from atlas.mapping import ORIENTATION_RULE
    v = load_variant_tables(base)
    report = {"counts": {g: list(c) for g, c in counts(v).items()}, "per_gene": {},
              "orientation_rule": ORIENTATION_RULE}
    failures = []
    if any(tuple(report["counts"].get(g, ())) != EXPECTED[g][0] for g in GENES):
        failures.append("row counts differ from the instruction's table")
    if "-score" not in ORIENTATION_RULE.replace(" ", ""):
        failures.append("atlas.mapping's orientation rule is not -score")
    for g in GENES:
        urn = URN[g]
        dep = load_deposit(urn, deposits or ER.RAW)
        meta = json.loads(((deposits or ER.RAW) / ("urn_mavedb_" + urn.split(":")[-1])
                           / "metadata.json").read_text())
        gv = v[v["gene"] == g]
        if set(gv["transcript"]) != {EXTERNAL[urn]["transcript"]}:
            failures.append(f"{g}: transcripts {sorted(set(gv['transcript']))}")
        r = {"urn": urn, "1_urn_and_score": check_urns_and_scores(gv, dep),
             "2_stratum": check_strata(gv), "3_completeness": check_completeness(gv, dep),
             "4_direction": check_direction(gv, dep, meta)}
        report["per_gene"][g] = r
        one, two, three, four = (r["1_urn_and_score"], r["2_stratum"], r["3_completeness"],
                                 r["4_direction"])
        if one["missing"] or one["score_differs"]:
            failures.append(f"{g}: check 1")
        if two["differs"]:
            failures.append(f"{g}: check 2")
        if three["only_in_deposit"] or three["only_in_table"] or three["other_intronic_not_deletion"]:
            failures.append(f"{g}: check 3")
        if not (four["deposit"]["abnormal_lower"] and four["table_rows"]["abnormal_lower"]):
            failures.append(f"{g}: check 4")
    report["total_rows"] = int(len(v))
    report["other_intronic_total"] = sum(r["3_completeness"]["other_intronic"]
                                         for r in report["per_gene"].values())
    report["failures"] = failures
    return v, report


def score_input(v: pd.DataFrame) -> pd.DataFrame:
    """What every scorer scores: one row per variant, keyed by the release's
    genomic variant_id (chrom-pos-ref-alt), which its score files use too."""
    cols = ["variant_id", "gene", "chrom", "pos", "ref", "alt", "variant_urn", "stratum"]
    out = v[cols].copy()
    out["chrom"], out["pos"] = out["chrom"].astype(str), out["pos"].astype(int)
    if out["variant_id"].duplicated().any():
        raise CheckFailed("a variant_id comes twice")
    want = out["chrom"] + "-" + out["pos"].astype(str) + "-" + out["ref"] + "-" + out["alt"]
    if not (want == out["variant_id"]).all():
        raise CheckFailed("variant_id is not chrom-pos-ref-alt on every row")
    if not ((out["ref"].str.len() == 1) & (out["alt"].str.len() == 1)).all():
        raise CheckFailed("a row is not an SNV")
    return out.reset_index(drop=True)


# ==========================================================================
# the lookup columns: the committed scorers, run from byte-identical copies
# ==========================================================================
MODELS = REPO / "models"
COPY_DIR = "cache/external_v1"
# column -> (scorer folder, its arguments, its output column)
LOOKUPS = {
    "cadd": ("cadd", (), "cadd"),
    "gpn_msa": ("gpn_msa", (), "gpn_msa"),
    "phylop100way": ("conservation", ("--track", "phyloP100way"), "phylop100way"),
    "phastcons100way": ("conservation", ("--track", "phastCons100way"), "phastcons100way"),
    "gnomad_af_global": ("gnomad", ("--metric", "global"), "gnomad_af_global"),
    "gnomad_af_popmax": ("gnomad", ("--metric", "popmax"), "gnomad_af_popmax"),
}
GPN_REVISION = "cf1718a9"      # the revision the core and the release's columns record


def lookup_dir(folder: str) -> Path:
    return MODELS / folder / COPY_DIR


def lookup_output(col: str) -> Path:
    return lookup_dir(LOOKUPS[col][0]) / f"{col}.parquet"


def _load_module(path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(path.parent.parent.parent.name + "_copy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def copy_scorers() -> dict[str, str]:
    """Copy each committed scorer (and GPN-MSA's pinned tabix index) into its
    cache/external_v1/; the copy must be byte for byte the committed file."""
    import shutil
    shas = {}
    for folder in sorted({f for f, _, _ in LOOKUPS.values()}):
        dst = lookup_dir(folder)
        dst.mkdir(parents=True, exist_ok=True)
        for name in ["score.py"] + (["gpn_scores.tsv.bgz.tbi"] if folder == "gpn_msa" else []):
            src = MODELS / folder / name
            shutil.copyfile(src, dst / name)
            a, b = sha256_file(src), sha256_file(dst / name)
            if a != b:
                raise SystemExit(f"the copy of models/{folder}/{name} differs")
            shas[f"models/{folder}/{name}"] = a
    return shas


def lookup_sources() -> dict:
    """The remote sources' versions as served now; GPN-MSA must be the pinned revision
    at the pinned size, or nothing is scored."""
    import requests
    from datetime import datetime, timezone
    out = {"read_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    gpn = _load_module(lookup_dir("gpn_msa") / "score.py")
    h = requests.head(gpn.RESOLVE_URL, allow_redirects=False, timeout=60).headers
    out["gpn_msa"] = {"url": gpn.RESOLVE_URL, "x_repo_commit": h.get("X-Repo-Commit"),
                      "x_linked_size": h.get("X-Linked-Size"),
                      "x_linked_etag": h.get("X-Linked-Etag")}
    if not str(h.get("X-Repo-Commit", "")).startswith(GPN_REVISION) or \
            int(h.get("X-Linked-Size") or -1) != gpn.TOTAL_SIZE:
        raise SystemExit(f"GPN-MSA is not revision {GPN_REVISION} at the pinned size: "
                         f"{out['gpn_msa']}")
    cadd = _load_module(lookup_dir("cadd") / "score.py")
    out["cadd"] = {"api_version": cadd.API_VERSION, "files": {}}
    for url in (cadd.MIRRORS["US"], cadd.MIRRORS["US"] + ".tbi"):
        h = requests.head(url, allow_redirects=True, timeout=60).headers
        out["cadd"]["files"][url] = {k: h.get(k) for k in ("Content-Length", "Last-Modified",
                                                           "ETag")}
    cons = _load_module(lookup_dir("conservation") / "score.py")
    first = pd.read_parquet(SCORE_INPUT).iloc[0]
    out["ucsc"] = {}
    for track in cons.TRACKS:
        r = requests.get(cons.API, params={"genome": "hg38", "track": track,
                                           "chrom": f"chr{first['chrom']}",
                                           "start": int(first["pos"]) - 1,
                                           "end": int(first["pos"])}, timeout=120)
        r.raise_for_status()
        j = r.json()
        out["ucsc"][track] = {k: j.get(k) for k in ("dataTime", "dataTimeStamp", "trackType",
                                                    "bigDataUrl", "genome")}
    gno = _load_module(lookup_dir("gnomad") / "score.py")
    out["gnomad"] = {"api": gno.API, "dataset": gno.DATASET}
    return out


def run_lookups(python: str = sys.executable) -> dict:
    from datetime import datetime, timezone
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    shas = copy_scorers()
    sources = lookup_sources()
    _write_json(WORK / "lookups.sources.json", sources)
    counts_ = {}
    for col, (folder, extra, _) in LOOKUPS.items():
        d = lookup_dir(folder)
        cmd = [python, str(d / "score.py"), "--input", str(SCORE_INPUT),
               "--output", str(lookup_output(col)), "--cache-dir", str(d / "cache"), *extra]
        print("$", " ".join(str(c).replace(str(REPO) + "/", "") for c in cmd), flush=True)
        subprocess.run(cmd, check=True)
        counts_[col] = int(len(pd.read_parquet(lookup_output(col))))
    freeze = subprocess.run([python, "-m", "pip", "freeze"], check=True, capture_output=True,
                            text=True).stdout.split("\n")
    ver = subprocess.run([python, "-c", "import platform; print(platform.python_version()); "
                          "print(platform.platform())"], check=True, capture_output=True,
                         text=True).stdout.split("\n")
    env = {"scorers": "the committed models/{cadd,gpn_msa,conservation,gnomad}/score.py, "
                      "run unchanged from byte-identical copies in models/<scorer>/" + COPY_DIR,
           "scorer_sha256": shas, "interpreter": python, "python": ver[0], "platform": ver[1],
           "pip_freeze": [x for x in freeze if x], "run_started_utc": started,
           "run_ended_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_scored": counts_, "input_sha256": sha256_file(SCORE_INPUT)}
    _write_json(WORK / "lookups.environment.json", env)
    return env


# ==========================================================================
# the delivered table, its provenance, and the comparison with the release
# ==========================================================================
OUT_TABLE = RESULTS / "external_scores_v1.parquet"
OUT_PROV = RESULTS / "external_scores_v1.provenance.json"
OUT_CONC = RESULTS / "external_concordance_v1.tsv"
SPLICEAI_OUT = MODELS / "spliceai" / COPY_DIR / "spliceai.parquet"
PANGOLIN_OUT = MODELS / "pangolin" / COPY_DIR / "pangolin.parquet"
AG_OUT = MODELS / "alphagenome" / "cache" / "official_v1"
AG_COMPONENTS = {"SPLICE_SITES": "alphagenome_splice_sites",
                 "SPLICE_SITE_USAGE": "alphagenome_splice_site_usage",
                 "SPLICE_JUNCTIONS": "alphagenome_splice_junctions"}
SCORE_COLUMNS = ["spliceai_ds", "pangolin_score", "cadd", "gpn_msa", "phylop100way",
                 "phastcons100way", "gnomad_af_global", "gnomad_af_popmax",
                 "alphagenome_official"]
ID_COLUMNS = ["gene", "urn", "variant_urn", "hgvs_nt", "hgvs_c", "variant_id", "chrom", "pos",
              "ref", "alt", "offset", "stratum", "companion_stratum", "score",
              "functional_pathogenicity", "standard_error"]
# this round's column -> (the release's file run, its column, the rule)
CONCORDANCE = {"cadd": ("cadd", "cadd", "equal"),
               "gpn_msa": ("gpn_msa", "gpn_msa", "equal"),
               "phylop100way": ("conservation", "phylop", "equal"),
               "phastcons100way": ("conservation", "phastcons", "equal"),
               "spliceai_ds": ("spliceai", "spliceai", "max_abs_diff <= 0.001"),
               "pangolin_score": ("pangolin", "pangolin", "max_abs_diff <= 0.001"),
               "alphagenome_official": ("alphagenome", "alphagenome_splicing", "report"),
               "alphagenome_splice_sites": ("alphagenome", "SPLICE_SITES", "report"),
               "alphagenome_splice_site_usage": ("alphagenome", "SPLICE_SITE_USAGE", "report"),
               "alphagenome_splice_junctions": ("alphagenome", "SPLICE_JUNCTIONS", "report")}
MAX_ABS = 0.001


def read_ag(path: Path) -> pd.DataFrame:
    """An AlphaGenome scores table, values read back exactly (they are written with repr)."""
    t = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for c in ["alphagenome_splicing", "alphagenome_official", *AG_COMPONENTS]:
        if c in t.columns:
            t[c] = t[c].map(lambda x: float(x) if x != "" else np.nan)
    return t


def build_table(v: pd.DataFrame, se: dict) -> pd.DataFrame:
    """The 5,095 rows with every column, the scores read from the scorers' outputs."""
    t = pd.DataFrame({
        "gene": v["gene"], "urn": v["urn"], "variant_urn": v["variant_urn"],
        "hgvs_nt": v["hgvs_nt"],
        "hgvs_c": [hgvs_c(h, tr) for h, tr in zip(v["hgvs_nt"], v["transcript"])],
        "variant_id": v["variant_id"], "chrom": v["chrom"].astype(str),
        "pos": v["pos"].astype(int), "ref": v["ref"], "alt": v["alt"],
        "offset": v["offset"].astype(int), "stratum": v["stratum"].map(STRATUM),
        "companion_stratum": v["stratum"], "score": v["score"],
        "functional_pathogenicity": -v["score"],
        "standard_error": v["variant_urn"].map(se)})
    for col, (_, _, out_col) in LOOKUPS.items():
        d = pd.read_parquet(lookup_output(col))[["variant_id", out_col]]
        t = t.merge(d.rename(columns={out_col: col}), on="variant_id", how="left")
    t = t.merge(pd.read_parquet(SPLICEAI_OUT)[["variant_id", "spliceai_ds"]],
                on="variant_id", how="left")
    t = t.merge(pd.read_parquet(PANGOLIN_OUT)[["variant_id", "pangolin_score"]],
                on="variant_id", how="left")
    ag = read_ag(AG_OUT / "external_scores.tsv")
    t = t.merge(ag[["variant_id", "alphagenome_official", *AG_COMPONENTS]]
                .rename(columns=AG_COMPONENTS), on="variant_id", how="left")
    if len(t) != len(v):
        raise SystemExit("a scorer's output holds a variant twice")
    return t[ID_COLUMNS + SCORE_COLUMNS + list(AG_COMPONENTS.values())]


def release_values(run: str, col: str, base: Path = COMPANION_DIR) -> pd.Series:
    if run == "alphagenome":
        t = read_ag(companion(f"{AG_DIR}/external_scores.tsv", base))
        t = t[t["gene"].isin(GENES)]
        return t.set_index("variant_id")[col]
    parts = [pd.read_parquet(companion(f"{EXT_DIR}/{g.lower()}_{run}.parquet", base))
             for g in GENES]
    return pd.concat(parts).set_index("variant_id")[col]


def concordance(t: pd.DataFrame, release=release_values) -> pd.DataFrame:
    """Per gene and column, on the 3-50 bp rows, the only ones the release scored."""
    from scipy.stats import spearmanr
    rows = []
    sub = t[t["stratum"].isin(["splice_3_10", "splice_11_50"])]
    for col, (run, rcol, rule) in CONCORDANCE.items():
        rel = release(run, rcol)
        for g in GENES + ("all",):
            d = sub if g == "all" else sub[sub["gene"] == g]
            ours = d.set_index("variant_id")[col].astype(float)
            theirs = rel.reindex(ours.index).astype(float)
            both = ours.notna() & theirs.notna()
            equal = (ours == theirs) | (ours.isna() & theirs.isna())
            diff = (ours[both] - theirs[both]).abs()
            rec = {"gene": g, "column": col, "release_column": rcol, "rule": rule,
                   "rows": int(len(d)), "both_scored": int(both.sum()),
                   "only_this_round": int((ours.notna() & theirs.isna()).sum()),
                   "only_release": int((ours.isna() & theirs.notna()).sum()),
                   "equal_rows": int(equal.sum()),
                   "max_abs_diff": float(diff.max()) if len(diff) else np.nan,
                   "spearman_rho": (float(spearmanr(ours[both], theirs[both]).statistic)
                                    if both.sum() >= 3 and rule == "report" else np.nan)}
            if rule == "equal":
                rec["holds"] = bool(rec["equal_rows"] == rec["rows"])
            elif rule.startswith("max_abs"):
                rec["holds"] = bool(rec["only_release"] == 0 and rec["only_this_round"] == 0
                                    and not rec["max_abs_diff"] > MAX_ABS)
            else:
                rec["holds"] = None
            rows.append(rec)
    return pd.DataFrame(rows)


def scored_counts(t: pd.DataFrame) -> dict:
    """gene -> stratum -> column -> variants scored."""
    cols = SCORE_COLUMNS + list(AG_COMPONENTS.values())
    out = {}
    for (g, s), d in t.groupby(["gene", "stratum"]):
        out.setdefault(g, {})[s] = {"rows": int(len(d))} | {c: int(d[c].notna().sum())
                                                              for c in cols}
    return out


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def provenance(t: pd.DataFrame, checks: dict) -> dict:
    """What produced each column, from the scorers' own records."""
    from datetime import datetime, timezone
    look = _read_json(WORK / "lookups.environment.json")
    src = _read_json(WORK / "lookups.sources.json")
    sa = _read_json(Path(str(SPLICEAI_OUT) + ".environment.json"))
    pg = _read_json(Path(str(PANGOLIN_OUT) + ".environment.json"))
    ag = _read_json(AG_OUT / "external_scores.tsv.provenance.json")
    refs = _read_json(REPO / "data" / "refs" / "grch38_external.build.json")
    lookup_logs = {}
    for col, (folder, _, out_col) in LOOKUPS.items():
        log = lookup_dir(folder) / ("run_log.json" if folder in ("cadd", "gpn_msa")
                                    else f"run_log_{out_col}.json")
        lookup_logs[col] = _read_json(log) if log.exists() else None
    lookup_env = {k: look[k] for k in ("interpreter", "python", "platform", "pip_freeze",
                                       "run_started_utc", "run_ended_utc")}
    columns = {
        "spliceai_ds": {"tool": sa["tool"], "scorer": sa["scorer"],
                        "scorer_sha256": sa["scorer_sha256"],
                        "driver": "models/spliceai/score_external.py",
                        "driver_sha256": sa["driver_sha256"],
                        "configuration": sa["configuration"], "reference": sa["reference"],
                        "direction": "larger = more splice-altering = more damaging; no flip",
                        "scored_utc_range": sa["scored_utc_range"],
                        "environment": {k: sa[k] for k in ("interpreter", "python", "platform",
                                                           "pip_freeze")}},
        "pangolin_score": {"tool": pg["tool"], "scorer": pg["scorer"],
                           "scorer_sha256": pg["scorer_sha256"],
                           "driver": "models/pangolin/score_external.py",
                           "driver_sha256": pg["driver_sha256"],
                           "configuration": pg["configuration"], "reference": pg["reference"],
                           "gtf_db": pg["gtf_db"],
                           "direction": "larger = more splice-altering = more damaging; no flip",
                           "scored_utc_range": pg["scored_utc_range"],
                           "environment": {k: pg["workers_records"][0][k] for k in (
                               "interpreter", "python", "platform", "pip_freeze")}},
        "alphagenome_official": {
            "tool": f"AlphaGenome, client {ag['environment']['client']['version']}, the "
                    "official notebook's merged splicing score (alphagenome_splicing)",
            "scorer": "models/alphagenome/score_official.py",
            # the scorer's run records name it (written_by) without its hash
            "scorer_sha256": sha256_file(MODELS / "alphagenome" / "score_official.py"),
            "scorer_sha256_when": "hashed when this record is assembled",
            "drift_check": {m: {k: v for k, v in _read_json(
                AG_OUT / f"{m}_scores.tsv.provenance.json").items()
                if k.startswith("against") or k == "after_external"}
                for m in ("check_start", "check_end")},
            "definition": ag["definition"], "definition_sha256": ag["definition_sha256"],
            "code_run": ag["code_run"],
            "model_version_requested": None,
            "default_when_not_requested": ag["environment"]["default_when_not_requested"],
            "components": AG_COMPONENTS,
            "direction": "larger = more damaging; no flip",
            "scored_utc_range": ag["scored_utc_range"], "calls": ag["calls"],
            "retries": ag["retries"], "failures": ag["failures"],
            "environment": {k: ag["environment"][k] for k in (
                "interpreter", "python", "platform", "pip_freeze", "requirements",
                "requirements_sha256")},
            "use": "research only; not used to train any other model"},
    }
    for col, (folder, extra, _) in LOOKUPS.items():
        columns[col] = {"scorer": f"models/{folder}/score.py",
                        "scorer_sha256": look["scorer_sha256"][f"models/{folder}/score.py"],
                        "run_from": f"models/{folder}/{COPY_DIR}/score.py (byte-identical copy)",
                        "arguments": list(extra), "run_log": lookup_logs[col],
                        "environment": lookup_env}
    columns["cadd"] |= {"tool": "CADD GRCh38-v1.7, precomputed whole-genome SNV file",
                        "source": src["cadd"], "direction": "PHRED; larger = more damaging"}
    columns["gpn_msa"] |= {"tool": "GPN-MSA hg38 precomputed scores, songlab/gpn-msa-hg38-scores",
                           "source": src["gpn_msa"], "direction": "gpn_msa = -raw; larger = more damaging"}
    for col, track in (("phylop100way", "phyloP100way"), ("phastcons100way", "phastCons100way")):
        columns[col] |= {"tool": f"UCSC {track}, hg38 track API", "source": src["ucsc"][track],
                         "direction": "larger = more conserved = more damaging"}
    for col in ("gnomad_af_global", "gnomad_af_popmax"):
        columns[col] |= {"tool": "gnomAD v4 (GraphQL API, dataset gnomad_r4)",
                         "source": src["gnomad"],
                         "direction": "-AF; larger = more damaging; a variant absent from "
                                      "gnomAD is not scored"}
    inputs = {"variant_tables": {g: {"release_path": f"{EXT_DIR}/{g.lower()}_variants.parquet",
                                     "sha256": sha256_file(companion(
                                         f"{EXT_DIR}/{g.lower()}_variants.parquet"))}
                                 for g in GENES},
              "release": {"tag": TAG, "commit": TAG_COMMIT, "doi": DOI},
              "deposits": {g: sha256_file(RAW / "mavedb_external" /
                                          ("urn_mavedb_" + URN[g].split(":")[-1]) / "scores.csv")
                           for g in GENES},
              "score_input": {"path": str(SCORE_INPUT.relative_to(REPO)),
                              "sha256": sha256_file(SCORE_INPUT)},
              "references": {k: refs["outputs"][k] for k in refs["outputs"]}}
    return {"file": str(OUT_TABLE.relative_to(REPO)), "rows": int(len(t)),
            "sha256": sha256_file(OUT_TABLE),
            "written_by": "src/atlas/external_scores.py",
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "direction": "functional_pathogenicity = -score (atlas.mapping); every score "
                         "column larger = more damaging",
            "columns": columns, "inputs": inputs,
            "scored_counts": scored_counts(t),
            "checks": {k: v for k, v in checks.items() if k != "per_gene"} |
                      {"per_gene_summary": {g: {
                          "1_missing": len(r["1_urn_and_score"]["missing"]),
                          "1_score_differs": len(r["1_urn_and_score"]["score_differs"]),
                          "2_stratum_differs": len(r["2_stratum"]["differs"]),
                          "3_deposit_intronic_snv": r["3_completeness"]["deposit_intronic_snv"],
                          "3_other_intronic_deletions": r["3_completeness"]["other_intronic"],
                          "4_direction": r["4_direction"]} for g, r in checks["per_gene"].items()}}}


def assemble() -> dict:
    v, checks = run_checks()
    if checks["failures"]:
        raise CheckFailed(f"checks failed: {checks['failures']}")
    se = {}
    for g in GENES:
        dep = load_deposit(URN[g])
        se.update(dict(zip(dep["accession"], dep["standard_error"])))
    t = build_table(v, se)
    conc = concordance(t)
    stops = conc[(conc["gene"] == "all") & (conc["holds"] == False)]  # noqa: E712
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(OUT_TABLE, index=False)
    conc.to_csv(OUT_CONC, sep="\t", index=False, float_format=lambda x: repr(float(x)))
    _write_json(OUT_PROV, provenance(t, checks))
    return {"rows": len(t), "stops": stops[["column", "rule"]].to_dict("records")}


# ==========================================================================
# command line
# ==========================================================================
def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["companion", "check", "lookups", "assemble"])
    args = ap.parse_args(argv)
    if args.command == "companion":
        print(json.dumps(fetch_companion(), indent=1))
        return 0
    if args.command == "lookups":
        env = run_lookups()
        print(f"lookups done: {env['n_scored']}")
        return 0
    if args.command == "assemble":
        res = assemble()
        print(f"wrote {OUT_TABLE.name} ({res['rows']} rows), {OUT_PROV.name}, {OUT_CONC.name}")
        if res["stops"]:
            print("comparison with the release fails:", res["stops"], file=sys.stderr)
            return 1
        return 0
    v, report = run_checks()
    _write_json(WORK / "checks.json", report)
    if report["failures"]:
        print("checks failed:", report["failures"], file=sys.stderr)
        return 1
    inp = score_input(v)
    SCORE_INPUT.parent.mkdir(parents=True, exist_ok=True)
    inp.to_parquet(SCORE_INPUT, index=False)
    print(f"four checks hold on {report['total_rows']:,} rows; wrote {SCORE_INPUT} "
          f"(sha256 {sha256_file(SCORE_INPUT)[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
