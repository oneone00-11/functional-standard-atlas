"""The survey's deposits read from their MaveDB records: whether a deposit is human, and
whether it belongs to the designed-stability platform.

The survey (atlas.mavedb_survey, results/mavedb_survey_v1.tsv) is not redone: its 2,803
deposits, column-name rules and classes stay as they are. Two of its labels are
corrected here.

Human (G13). v1's flag looked for "homo sapiens" or "human" in the target genes' text
of the search records; deposits registered by genomic accession were therefore all
filed as non-human, and a target named "SpCas9 (Human Codon Optimized)" as human. Here
each target is judged on its record, and a deposit is human if one of its targets is:

  taxonomy        a target registered by sequence: targetSequence.taxonomy.code is 9606
  assembly        a target registered by accession: targetAccession.assembly is GRCh37 or
                  GRCh38 (or hg19, hg38)
  accession       a target registered by accession without an assembly: the species of
                  the accession. Ensembl's ENST, ENSG and ENSP are human; a RefSeq
                  accession is looked up at NCBI (the replies cached)
  undetermined    none of these decides; counted as non-human, and listed
  record_unavailable  the record could not be fetched; v1's flag is kept

Platform (G9). v1's rule matched names (villin, pin1, HHH_rd...) and propagated a match
across the deposits that share a title stem among the 2,452 ceiling candidates only: it
missed three combined deposits whose two protease deposits had failed to fetch, and took
in five Domainome deposits named after villin or pin1. Here, on all 2,803 deposits:

  seed            the score table carries a column score_95CI_low (the survey's header),
                  or the title contains "trypsin digestion" or "chymotrypsin digestion"
  set             a seed's whole experiment set (the digits after "urn:mavedb:") is the
                  platform

Subcommands (PYTHONPATH=src):
  fetch       every survey deposit's score-set record into data/raw/mavedb_records_v1/
              (untracked), with the time each was fetched and the URNs that failed
  accessions  the RefSeq accessions without an assembly, looked up at NCBI (cached)
  survey      results/mavedb_survey_v2.tsv and results/mavedb_metadata_v2.tsv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from atlas import paths
from atlas.mavedb_survey import API, PAUSE, TIMEOUT

REPO = paths.REPO
RESULTS = paths.RESULTS
CACHE = REPO / "data" / "raw" / "mavedb_records_v1"
SURVEY_V1 = RESULTS / "mavedb_survey_v1.tsv"
METADATA_V1 = RESULTS / "mavedb_metadata_v1.tsv"
USER_AGENT = "atlas-mavedb-records/1.0"


def _safe(urn: str) -> str:
    return urn.replace(":", "_")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get(url: str, session, attempts: int = 4, params: dict | None = None):
    for k in range(attempts):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT,
                            headers={"User-Agent": USER_AGENT})
            if r.status_code == 404:
                r.raise_for_status()
            r.raise_for_status()
            return r
        except Exception:                                       # noqa: BLE001
            if k == attempts - 1:
                raise
            time.sleep(5 * 2 ** k)


def fetch_records(urns: list[str], cache: Path = CACHE, force: bool = False) -> dict:
    """Each URN's record, cached as records/<urn>.json; the log records the time each was
    fetched, the failures their error. A second run fetches only what is missing."""
    import requests
    session = requests.Session()
    (cache / "records").mkdir(parents=True, exist_ok=True)
    log_path = cache / "fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"records": {}, "failed": {}}
    for i, urn in enumerate(urns, 1):
        dest = cache / "records" / f"{_safe(urn)}.json"
        if not force and dest.exists() and urn in log["records"]:
            continue
        try:
            rec = _get(f"{API}/score-sets/{urn}", session).json()
        except Exception as e:                                  # noqa: BLE001
            log["failed"][urn] = {"utc": _now(), "error": f"{type(e).__name__}: {e}"[:200]}
            log_path.write_text(json.dumps(log, indent=1) + "\n")
            continue
        dest.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        log["records"][urn] = {"fetched_utc": _now()}
        log["failed"].pop(urn, None)
        if i % 50 == 0:
            log_path.write_text(json.dumps(log, indent=1) + "\n")
            print(f"  {i}/{len(urns)} ({len(log['failed'])} failed so far)", flush=True)
        time.sleep(PAUSE)
    log_path.write_text(json.dumps(log, indent=1) + "\n")
    return log


# ==========================================================================
# the human flag
# ==========================================================================
HUMAN_TAXID = 9606
HUMAN_ASSEMBLIES = {"grch37", "grch38", "hg19", "hg38"}
ENSEMBL = re.compile(r"^ENS([A-Z]*?)([TGP])\d")       # ENST... human; ENSMUST... mouse
REFSEQ = re.compile(r"^(?:[ANXYW][CGMRTWZP])_\d")
NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
BASES = ("taxonomy", "assembly", "accession", "undetermined", "record_unavailable")


def record(urn: str, cache: Path = CACHE) -> dict | None:
    p = cache / "records" / f"{_safe(urn)}.json"
    return json.loads(p.read_text()) if p.exists() else None


def refseq_accessions(records: dict[str, dict]) -> list[str]:
    """The RefSeq accessions of targets registered by accession without an assembly."""
    out = set()
    for rec in records.values():
        for t in rec.get("targetGenes") or []:
            acc = t.get("targetAccession") or {}
            a = str(acc.get("accession") or "")
            if acc and not acc.get("assembly") and REFSEQ.match(a):
                out.add(a)
    return sorted(out)


def fetch_ncbi(accessions: list[str], cache: Path = CACHE) -> dict:
    """Each accession's NCBI summary (nuccore, or protein for NP_/XP_/YP_/WP_), cached."""
    import requests
    session = requests.Session()
    d = cache / "ncbi"
    d.mkdir(parents=True, exist_ok=True)
    log_path = d / "fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for a in accessions:
        if (d / f"{a}.json").exists() and a in log:
            continue
        db = "protein" if a[:2] in ("NP", "XP", "YP", "WP") else "nuccore"
        try:
            r = _get(NCBI, session, params={"db": db, "id": a, "retmode": "json"})
            (d / f"{a}.json").write_text(r.text)
            log[a] = {"fetched_utc": _now(), "db": db}
        except Exception as e:                                  # noqa: BLE001
            log[a] = {"fetched_utc": _now(), "db": db, "error": f"{type(e).__name__}: {e}"[:200]}
        log_path.write_text(json.dumps(log, indent=1) + "\n")
        time.sleep(0.4)                                         # under NCBI's 3 a second
    return log


def ncbi_taxid(accession: str, cache: Path = CACHE) -> int | None:
    p = cache / "ncbi" / f"{accession}.json"
    if not p.exists():
        return None
    res = json.loads(p.read_text()).get("result") or {}
    for uid in res.get("uids") or []:
        tax = res.get(uid, {}).get("taxid")
        if tax:
            return int(tax)
    return None


def target_human(t: dict, cache: Path = CACHE) -> tuple[bool | None, str]:
    """(human or not, or None if undecided; the basis) for one target."""
    seq, acc = t.get("targetSequence"), t.get("targetAccession")
    if seq:
        code = (seq.get("taxonomy") or {}).get("code")
        return (int(code) == HUMAN_TAXID, "taxonomy") if code is not None else (None, "undetermined")
    if acc:
        asm = str(acc.get("assembly") or "").strip()
        if asm:
            return asm.lower() in HUMAN_ASSEMBLIES, "assembly"
        a = str(acc.get("accession") or "")
        m = ENSEMBL.match(a)
        if m:
            return m.group(1) == "", "accession"
        if REFSEQ.match(a):
            tax = ncbi_taxid(a, cache)
            return (tax == HUMAN_TAXID, "accession") if tax is not None else (None, "undetermined")
    return None, "undetermined"


def deposit_human(rec: dict | None, v1_flag: bool, cache: Path = CACHE) -> tuple[bool, str]:
    """Human if one target is; else non-human, on the first target's basis when every
    target is decided, and 'undetermined' otherwise. No record: v1's flag."""
    if rec is None:
        return bool(v1_flag), "record_unavailable"
    judged = [target_human(t, cache) for t in rec.get("targetGenes") or []]
    for h, basis in judged:
        if h is True:
            return True, basis
    if judged and all(h is False for h, _ in judged):
        return False, judged[0][1]
    return False, "undetermined"


# ==========================================================================
# the platform
# ==========================================================================
SEED_COLUMN = "score_95CI_low"
SEED_TITLE = re.compile(r"(?:chymo)?trypsin digestion", re.I)
SET_RE = re.compile(r"^urn:mavedb:(\d+)-")


def experiment_set(urn: str) -> str:
    m = SET_RE.match(urn)
    if not m:
        raise ValueError(f"{urn!r} is not a score-set URN")
    return m.group(1)


def platform_table(survey: pd.DataFrame, titles: pd.Series, columns: pd.Series) -> pd.DataFrame:
    """urn -> platform, platform_basis: a seed's experiment set is the platform."""
    t = pd.DataFrame({"urn": survey["urn"], "title": titles.fillna("").astype(str),
                      "columns": columns.fillna("").astype(str)})
    t["set"] = t["urn"].map(experiment_set)
    by_col = t["columns"].map(lambda c: SEED_COLUMN in c.split(";"))
    by_title = t["title"].map(lambda x: bool(SEED_TITLE.search(x)))
    t["seed"] = by_col | by_title
    sets = set(t.loc[t["seed"], "set"])
    t["platform"] = t["set"].map(lambda k: "designed_stability" if k in sets else "other")
    t["platform_basis"] = ""
    t.loc[t["platform"] == "designed_stability", "platform_basis"] = "experiment set"
    t.loc[by_col & ~by_title, "platform_basis"] = f"seed: {SEED_COLUMN}"
    t.loc[by_title & ~by_col, "platform_basis"] = "seed: title"
    t.loc[by_title & by_col, "platform_basis"] = f"seed: {SEED_COLUMN} and title"
    return t[["urn", "set", "platform", "platform_basis"]]


# ==========================================================================
# the survey and the metadata, v2
# ==========================================================================
def annotations(survey: pd.DataFrame, cache: Path = CACHE) -> pd.DataFrame:
    """Per deposit: human and human_basis (the record), platform and platform_basis. The
    platform reads the record's full title and score columns where the survey has none
    (the survey truncated titles to 120 characters and holds no header for a table it
    failed to fetch)."""
    recs = {u: record(u, cache) for u in survey["urn"]}
    rows, titles, columns = [], [], []
    for u, v1 in zip(survey["urn"], survey["human"]):
        rec = recs[u]
        h, basis = deposit_human(rec, bool(v1), cache)
        rows.append({"urn": u, "human": h, "human_basis": basis})
        titles.append((rec or {}).get("title") or "")
        columns.append(";".join(((rec or {}).get("datasetColumns") or {}).get("scoreColumns")
                                or []))
    out = pd.DataFrame(rows)
    sv_cols = survey["all_columns"].fillna("").astype(str)
    cols = pd.Series([a if a else b for a, b in zip(sv_cols, columns)], index=survey.index)
    sv_titles = survey["title"].fillna("").astype(str)
    tit = pd.Series([r if r else s for r, s in zip(titles, sv_titles)], index=survey.index)
    plat = platform_table(survey, tit, cols)
    return out.merge(plat, on="urn", how="left", validate="one_to_one")


def survey_v2(survey: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    a = ann.set_index("urn")
    t = survey.copy()
    t["human"] = t["urn"].map(a["human"]).astype(bool)
    for c in ("human_basis", "platform", "platform_basis"):
        t[c] = t["urn"].map(a[c])
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "accessions", "survey"])
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)
    survey = pd.read_csv(SURVEY_V1, sep="\t")
    if args.command == "fetch":
        log = fetch_records(survey["urn"].tolist())
        print(f"{len(log['records'])} records cached; {len(log['failed'])} failed")
        return 0
    if args.command == "accessions":
        recs = {u: r for u in survey["urn"] if (r := record(u)) is not None}
        acc = refseq_accessions(recs)
        log = fetch_ncbi(acc)
        bad = [a for a in acc if "error" in log.get(a, {})]
        print(f"{len(acc)} RefSeq accessions without an assembly; {len(bad)} failed: {bad}")
        return 0
    ann = annotations(survey)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sv2 = survey_v2(survey, ann)
    md2 = survey_v2(pd.read_csv(METADATA_V1, sep="\t"), ann)
    sv2.to_csv(out / paths.name("mavedb_survey"), sep="\t", index=False)
    md2.to_csv(out / paths.name("mavedb_metadata"), sep="\t", index=False)
    print(f"{paths.name('mavedb_survey')}: {len(sv2)} rows, human {int(sv2['human'].sum())} "
          f"(v1 {int(survey['human'].sum())}); basis {sv2['human_basis'].value_counts().to_dict()}")
    plat = sv2[sv2["platform"] == "designed_stability"]
    print(f"platform: {plat['urn'].map(experiment_set).nunique()} experiment sets, "
          f"{len(plat)} deposits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
