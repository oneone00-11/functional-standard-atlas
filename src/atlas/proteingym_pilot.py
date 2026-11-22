"""Trial: how much of the spread in predictor performance across experiments does
measurement reliability explain, and how far are the best models from the ceiling?

ProteinGym scores every zero-shot model on every substitution experiment (the
Spearman rho per experiment). The atlas has a deposit-level attenuation ceiling for
every MaveDB deposit that permits one (`results/mavedb_ceilings_v1.tsv`, read-only).
This module pairs ProteinGym experiments with MaveDB deposits and puts the two
together.

This is a trial. Every output goes to `results/_scratch/k10_pilot/` (CONVENTIONS.md
rule 6), which `atlas.package_release` never ships. So do the downloaded inputs, in
`results/_scratch/k10_pilot/inputs/`, each registered with its URL, version and
SHA-256 in that directory's MANIFEST.json. If this becomes an analysis, its outputs
move to `results/`.

Inputs
------
* ProteinGym `reference_files/DMS_substitutions.csv` and
  `benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv`
  at commit ``PG_COMMIT`` (the main branch on 2026-03-25). The reference file is
  byte-identical to the one in the latest release, PG_v1.3. The Spearman table on
  main carries the zero-shot models added after that release. The release's own
  table is fetched too, for the record.
* The v1.3 DMS archive (`DMS_ProteinGym_substitutions.zip`), for the data check.
* The MaveDB record of every deposit in the ceiling table outside the protease-
  resistance stability platform (stratum ``designed_stability``), and the score table
  of every candidate (see Pairing). That platform's interval is a curve-fitting CI on
  dG, not a reproducibility interval, and the atlas never pools its ceilings
  (`atlas.mavedb_ceilings`); ProteinGym's experiments from the same platform
  therefore find no deposit here.
* UniProt accessions and gene names for the ProteinGym target entry names.
* Read-only, for the accounting: `results/mavedb_survey_v1.tsv` (every deposit's
  columns) and `results/mavedb_metadata_v1.tsv` (every deposit's target names).

Pairing (`candidate_table`, then `pair`)
----------------------------------------
ProteinGym's reference file names no MaveDB deposit. A deposit is a candidate for a
ProteinGym experiment when
  protein  the UniProt accession, or else a gene symbol, of the ProteinGym target
           matches one of the deposit's target genes, and
  paper    a publication of the deposit or of its experiment, primary or else
           secondary, has ProteinGym's title (after normalisation; a MaveDB title cut
           short matches as a prefix of at least TITLE_PREFIX characters), or its
           first author's surname and a year within one; or the deposit records no
           publication at all, and then only the data can tie it.
A candidate that MaveDB's own record marks as superseded by another candidate gives
way to it. The last number of a URN is not a version: urn:mavedb:00000069-a-1 and
-a-2 are two sorts of one experiment.

Every candidate is then checked against ProteinGym's own scores on the single
mutants they share (`data_check`). It reproduces them when |Spearman rho| >=
DATA_MATCH and it covers at least DATA_COVERAGE of ProteinGym's single mutants. The
status of each experiment in `pairs.tsv`:
  paired          exactly one candidate reproduces the scores; or several do and the
                  phenotype words of the ProteinGym record are shared with one
                  deposit's title and description more than with any other; or none
                  does, exactly one candidate could not be compared (its deposit names
                  no protein-level single change) and its paper matches, and that pair
                  is flagged "data not checked"
  ambiguous       several reproduce the scores and the phenotype words do not decide,
                  or several uncheckable candidates with a matching paper remain
  not_reproduced  candidates exist, none reproduces the scores
  no_match        no candidate (a deposit without a ceiling is not searched)
  deposit_claimed_twice  a deposit paired with two experiments: both are left unpaired
Each pair records its evidence; `candidates.tsv` holds every candidate's check.

Tiers (`tier_of`)
-----------------
The ceiling is the deposit-level value in the ceiling table; only status ``ok`` is used.
  strict  replicates whose mean reproduces the score, an interval (SE from a CI pair),
          or a standard error that reproduces the deposit's own 95% interval on every
          row (checked here, as `atlas.assay_aux` does)
  loose   the strict deposits plus those whose error column is used without that
          check: a standard deviation used as a standard error, a standard error with
          no interval to check against or one that fails the check, a variance
Ceilings below ``CORRECTION_BOUND`` (0.45) are reported and not used as divisors.

Analyses (`analyse`, once per tier)
-----------------------------------
Models are ranked by mean rho over the tier's experiments with ceiling >= 0.45. The
best model is rank 1 and the top ten are ranks 1-10.
  1  Spearman correlation, across experiments, of the ceiling with the best model's
     rho and with the median rho of the top ten (all experiments with a ceiling, and
     those >= 0.45); also with the per-experiment maximum over all models
  2  realisation = rho / ceiling for the best model and each of the top ten: the
     distribution, and the share of experiments at >= 0.9 and >= 0.8
  3  Kendall tau between the ranking by mean rho and the ranking by mean realisation,
     with each ranking's top ten

Accounting (`accounting`)
-------------------------
Where the ProteinGym experiments go: the stability platform; the pairing statuses;
for the experiments without a candidate, what MaveDB holds under the gene's names;
and two bounds by gene, on the loose tier (the gene has a deposit with an ok ceiling)
and on the strict tier (the gene has such a deposit with replicates, an interval, or
a standard error with an interval to check it against). Gene-level counts match
target names only and are an approximation.

Usage (PYTHONPATH=src):
    python -m atlas.proteingym_pilot --fetch      # download the inputs, then run
    python -m atlas.proteingym_pilot              # pair, check, analyse
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import re
import sys
import time
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import kendalltau, spearmanr

from atlas.manifest import load_manifest, register_file

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SCRATCH = RESULTS / "_scratch" / "k10_pilot"
INPUTS = SCRATCH / "inputs"
CEILINGS = RESULTS / "mavedb_ceilings_v1.tsv"
SURVEY = RESULTS / "mavedb_survey_v1.tsv"
METADATA = RESULTS / "mavedb_metadata_v1.tsv"

PG_REPO = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym"
PG_COMMIT = "144fe22b07dfaeec2b366f2346203a9838a55b4c"      # main, 2026-03-25
PG_RELEASE = "PG_v1.3"
PG_REFERENCE = "reference_files/DMS_substitutions.csv"
PG_SPEARMAN = ("benchmarks/DMS_zero_shot/substitutions/Spearman/"
               "DMS_substitutions_Spearman_DMS_level.csv")
PG_ARCHIVE = ("https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
              "DMS_ProteinGym_substitutions.zip")
PG_FILES = {
    "pg_reference.csv": (f"{PG_REPO}/{PG_COMMIT}/{PG_REFERENCE}", PG_COMMIT),
    "pg_spearman.csv": (f"{PG_REPO}/{PG_COMMIT}/{PG_SPEARMAN}", PG_COMMIT),
    "pg_spearman_release.csv": (f"{PG_REPO}/{PG_RELEASE}/{PG_SPEARMAN}", PG_RELEASE),
    "pg_dms_substitutions_v1.3.zip": (PG_ARCHIVE, "ProteinGym_v1.3"),
}
PG_META_COLUMNS = ("DMS ID", "Number of Mutants", "Selection Type", "UniProt ID",
                   "MSA_Neff_L_category", "Taxon")
PLATFORM_STUDY = ("tsuboyama", 2023)   # ProteinGym's experiments from the stability platform

MAVEDB = "https://api.mavedb.org/api/v1"
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
TIMEOUT = 120
PAUSE = 0.3

CORRECTION_BOUND = 0.45
TOP = 10
TITLE_MATCH = 0.90
TITLE_PREFIX = 40        # characters: a MaveDB title cut short still matches as a prefix
DATA_MATCH = 0.99        # |rho| on the shared single mutants: the same data
DATA_COVERAGE = 0.90     # share of ProteinGym's single mutants the deposit must cover
MIN_SINGLES = 10         # fewer protein-level single changes: the data cannot be compared
CI_Z = 1.96
CI_ATOL = 1e-3


# ==========================================================================
# fetching (network); everything below the fetch section reads INPUTS only
# ==========================================================================
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


RETRIES = 3
GIVE_UP_AFTER = 5        # consecutive failed deposits: the service is down, stop


def _get(url: str, retries: int = RETRIES, **kw) -> requests.Response:
    """GET with a retry on server errors and timeouts (30 s, then 60 s, apart)."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=TIMEOUT, **kw)
            if r.status_code < 500:
                r.raise_for_status()
                return r
            err: Exception = requests.HTTPError(f"{r.status_code} for {url}")
        except (requests.Timeout, requests.ConnectionError) as exc:
            err = exc
        if attempt < retries - 1:
            time.sleep(30 * (attempt + 1))
    raise err


def _store(rel: str, content: bytes, **meta) -> Path:
    path = INPUTS / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    register_file(INPUTS / "MANIFEST.json", path, fetched_utc=_now(), **meta)
    return path


def fetch_proteingym() -> None:
    """The ProteinGym files, skipping those already stored (a rerun keeps the first copy)."""
    for name, (url, version) in PG_FILES.items():
        if (INPUTS / name).exists():
            continue
        _store(name, _get(url).content, url=url, source="ProteinGym", version=version)
        time.sleep(PAUSE)


def candidate_urns() -> list[str]:
    """Every deposit of the ceiling table outside the protease-resistance stability
    platform, whose ceilings the atlas never pools (see the module docstring)."""
    c = pd.read_csv(CEILINGS, sep="\t")
    return sorted(c.loc[c["stratum"] != "designed_stability", "urn"].unique())


def _urn_file(urn: str, kind: str) -> str:
    stem = urn.replace(":", "_")
    return f"mavedb/{stem}.json" if kind == "record" else f"mavedb_scores/{stem}.csv"


def _fetch_each(urns: list[str], kind: str) -> list[str]:
    """Fetch one file per deposit, skipping those already stored. Stops after
    GIVE_UP_AFTER consecutive failures rather than walking the whole list."""
    failed, run = [], 0
    for i, urn in enumerate(urns, 1):
        rel = _urn_file(urn, kind)
        if (INPUTS / rel).exists():
            continue
        url = f"{MAVEDB}/score-sets/{urn}" + ("/scores" if kind == "scores" else "")
        try:
            _store(rel, _get(url).content, url=url, source="MaveDB", urn=urn)
            run = 0
        except requests.RequestException as exc:
            failed.append(f"{urn}: {exc}")
            run += 1
            if run >= GIVE_UP_AFTER:
                raise RuntimeError(f"MaveDB failed {run} times in a row; last: {exc}")
        if i % 100 == 0:
            print(f"  MaveDB {kind} {i}/{len(urns)}", file=sys.stderr, flush=True)
        time.sleep(PAUSE)
    return failed


def fetch_mavedb_records(urns: list[str]) -> list[str]:
    return _fetch_each(urns, "record")


def fetch_mavedb_scores(urns: list[str]) -> list[str]:
    return _fetch_each(urns, "scores")


def fetch_uniprot(entry_names: list[str], batch: int = 50) -> None:
    """Accession, entry name and gene names for each ProteinGym entry name. Batches
    already stored are skipped."""
    names = sorted(set(entry_names))
    for k in range(0, len(names), batch):
        chunk = names[k:k + batch]
        if (INPUTS / f"uniprot/entries_{k // batch:02d}.tsv").exists():
            continue
        query = " OR ".join(f"(id:{n})" for n in chunk)
        params = {"query": query, "format": "tsv", "size": 500,
                  "fields": "accession,id,gene_primary,gene_synonym"}
        r = _get(UNIPROT, params=params)
        _store(f"uniprot/entries_{k // batch:02d}.tsv", r.content, url=r.url,
               source="UniProt", entries=len(chunk))
        time.sleep(PAUSE)


# ==========================================================================
# reading the inputs
# ==========================================================================
def load_reference(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or INPUTS / "pg_reference.csv")


def load_spearman(path: Path | None = None) -> pd.DataFrame:
    sp = pd.read_csv(path or INPUTS / "pg_spearman.csv")
    return sp.set_index("DMS ID")


def model_columns(sp: pd.DataFrame) -> list[str]:
    cols = [c for c in sp.columns if c not in PG_META_COLUMNS]
    bad = [c for c in cols if not pd.api.types.is_numeric_dtype(sp[c])]
    if bad:
        raise ValueError(f"non-numeric columns taken for models: {bad}")
    outside = [c for c in cols if (sp[c].abs() > 1).any()]
    if outside:
        raise ValueError(f"columns outside [-1, 1] cannot be Spearman rho: {outside}")
    return cols


def load_uniprot() -> pd.DataFrame:
    frames = [pd.read_csv(p, sep="\t") for p in sorted((INPUTS / "uniprot").glob("*.tsv"))]
    u = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["Entry", "Entry Name", "Gene Names (primary)", "Gene Names (synonym)"])
    return u.rename(columns={"Entry": "accession", "Entry Name": "entry_name",
                             "Gene Names (primary)": "gene_primary",
                             "Gene Names (synonym)": "gene_synonym"})


def _pubs(rec: dict) -> list[dict]:
    """Publications of a score set and of its experiment, primary then secondary."""
    out = []
    exp = rec.get("experiment") or {}
    for kind in ("primary", "secondary"):
        key = f"{kind}PublicationIdentifiers"
        for p in (rec.get(key) or []) + (exp.get(key) or []):
            authors = p.get("authors") or []
            first = next((a.get("name") for a in authors if a.get("primary")),
                         authors[0].get("name") if authors else "")
            out.append({"kind": kind, "first_author": first or "",
                        "year": p.get("publicationYear"), "title": p.get("title") or "",
                        "doi": p.get("doi") or "",
                        "id": f"{p.get('dbName')}:{p.get('identifier')}"})
    return out


def _targets(rec: dict) -> tuple[set[str], set[str]]:
    """(UniProt accessions, gene names) of a score set's target genes."""
    acc, genes = set(), set()
    for t in rec.get("targetGenes") or []:
        for g in (t.get("name"), t.get("mappedHgncName")):
            if g:
                genes.add(str(g).upper())
        if t.get("uniprotIdFromMappedMetadata"):
            acc.add(t["uniprotIdFromMappedMetadata"])
        for e in t.get("externalIdentifiers") or []:
            ident = e.get("identifier") or {}
            if str(ident.get("dbName", "")).lower() == "uniprot" and ident.get("identifier"):
                acc.add(ident["identifier"])
    return acc, genes


def load_deposits() -> pd.DataFrame:
    """One row per fetched MaveDB record, with what pairing needs."""
    rows = []
    for p in sorted((INPUTS / "mavedb").glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        acc, genes = _targets(rec)
        rows.append({
            "urn": rec["urn"], "title": rec.get("title") or "",
            "description": rec.get("shortDescription") or "",
            "experiment": (rec.get("experiment") or {}).get("urn", ""),
            "accessions": sorted(acc), "genes": sorted(genes), "pubs": _pubs(rec),
            "superseded_by": _urn_of(rec.get("supersedingScoreSet")),
            "supersedes": _urn_of(rec.get("supersededScoreSet")),
        })
    return pd.DataFrame(rows)


def _urn_of(link) -> str | None:
    """MaveDB writes a supersede link as a short score-set record, or as its URN."""
    return link.get("urn") if isinstance(link, dict) else link


# ==========================================================================
# pairing: candidates by protein and paper
# ==========================================================================
def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


_INITIALS = re.compile(r"(?:[A-Z]\.?-?){1,4}")


def surname(name: str) -> str:
    """'Findlay, Gregory M', 'Findlay GM' and 'Findlay' -> 'findlay';
    'van Loggerenberg, Warren' -> 'vanloggerenberg'.

    MaveDB writes authors as 'Surname, Given' or as 'Surname Initials'; ProteinGym
    gives the surname alone.
    """
    n = _ascii(name).strip()
    if "," in n:
        n = n.split(",", 1)[0]
    else:
        toks = n.split()
        while len(toks) > 1 and _INITIALS.fullmatch(toks[-1]):
            toks.pop()
        n = " ".join(toks)
    return re.sub(r"[^a-z]", "", n.lower())


def norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", _ascii(t).lower())).strip()


def title_match(a: str, b: str) -> bool:
    """The same title after normalisation (similarity >= TITLE_MATCH), or one cut short:
    a prefix of the other at least TITLE_PREFIX characters long."""
    a, b = norm_title(a), norm_title(b)
    if not (a and b):
        return False
    short, full = sorted((a, b), key=len)
    if len(short) >= TITLE_PREFIX and full.startswith(short):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= TITLE_MATCH


def paper_match(pg_author: str, pg_year, pg_title: str, pubs: list[dict]) -> str:
    """Which evidence ties these publications to the ProteinGym record ('' if none)."""
    want = surname(pg_author)
    for p in pubs:
        if title_match(pg_title, p["title"]):
            return "title"
    for p in pubs:
        try:
            near = abs(int(p["year"]) - int(pg_year)) <= 1
        except (TypeError, ValueError):
            near = False
        if want and surname(p["first_author"]) == want and near:
            return "first author + year"
    return ""


def paper_evidence(pg_row, pubs: list[dict]) -> str:
    """How a deposit's publications tie it to a ProteinGym record: the primary ones
    first, then the secondary ones; 'none recorded' when the deposit records no
    publication at all; '' when its publications are of other papers."""
    if not pubs:
        return "none recorded"
    for kind in ("primary", "secondary"):
        m = paper_match(pg_row["first_author"], pg_row["year"], pg_row["title"],
                        [p for p in pubs if p.get("kind", "primary") == kind])
        if m:
            return m if kind == "primary" else f"{m} (secondary publication)"
    return ""


def protein_match(pg_acc: set[str], pg_genes: set[str], acc: list[str],
                  genes: list[str]) -> str:
    if pg_acc & set(acc):
        return "UniProt accession"
    if {g.upper() for g in pg_genes} & set(genes):
        return "gene"
    return ""


def pg_proteins(uniprot_id: str, up: pd.DataFrame) -> tuple[set[str], set[str]]:
    """(accessions, gene names) of a ProteinGym target from its UniProt entry name.
    The entry name's mnemonic stands in as a gene only when UniProt has no record."""
    acc, genes = set(), set()
    if len(up) and uniprot_id in up.index:
        hit = up.loc[[uniprot_id]]
        acc = set(hit["accession"].dropna())
        for col in ("gene_primary", "gene_synonym"):
            for v in hit[col].dropna():
                genes |= {g.upper() for g in re.split(r"[ ;]+", str(v)) if g}
    if not acc:
        genes.add(str(uniprot_id).split("_")[0].upper())
    return acc, genes


_STOP = {"score", "scores", "the", "and", "of", "to", "in", "for", "a", "an", "on",
         "with", "by", "dms", "deep", "mutational", "scan", "scanning", "assay",
         "variant", "variants", "effect", "effects", "fitness", "data", "set", "nan"}


def phenotype_words(*texts: str) -> set[str]:
    words = set()
    for t in texts:
        words |= {w for w in re.split(r"[^a-z0-9]+", _ascii(str(t)).lower())
                  if len(w) > 2 and w not in _STOP}
    return words


def dms_suffix(dms_id: str) -> str:
    """What follows the year in a ProteinGym DMS_id: 'P53_HUMAN_Giacomelli_2018_WT_Nutlin'
    -> 'WT_Nutlin'; '' when nothing does."""
    m = re.search(r"_(?:19|20)\d{2}(?:_(.*))?$", str(dms_id))
    return (m.group(1) or "") if m else ""


def _pg_words(r) -> set[str]:
    return phenotype_words(r.get("raw_DMS_phenotype_name", ""), r.get("selection_assay", ""),
                           r.get("selection_type", ""), dms_suffix(r["DMS_id"]))


def supersede_map(deposits: pd.DataFrame) -> dict[str, str]:
    """old URN -> the URN that supersedes it, from either side of MaveDB's links."""
    out = {}
    for _, d in deposits.iterrows():
        new, old = _link(d.get("superseded_by")), _link(d.get("supersedes"))
        if new:
            out[d["urn"]] = new
        if old:
            out[old] = d["urn"]
    return out


def _link(x) -> str | None:
    """A URN, or None; pandas turns a missing link into NaN, which is truthy."""
    return x if isinstance(x, str) and x else None


def current_versions(urns: list[str], superseded_by: dict[str, str]) -> list[str]:
    """Drop a deposit when one that supersedes it (directly or down the chain) is a
    candidate too. Nothing is inferred from the URN's numbers."""
    cands = set(urns)

    def replaced(u: str) -> bool:
        seen, nxt = {u}, superseded_by.get(u)
        while nxt and nxt not in seen:
            if nxt in cands:
                return True
            seen.add(nxt)
            nxt = superseded_by.get(nxt)
        return False
    return sorted(u for u in cands if not replaced(u))


CANDIDATE_COLUMNS = ["DMS_id", "urn", "protein", "paper", "phenotype_shared"]


def candidate_table(pg: pd.DataFrame, deposits: pd.DataFrame,
                    uniprot: pd.DataFrame) -> pd.DataFrame:
    """One row per (ProteinGym experiment, deposit) that passes the protein and paper
    tests, after superseded versions give way."""
    up = uniprot.set_index("entry_name") if len(uniprot) else uniprot
    links = supersede_map(deposits)
    deps = deposits.to_dict("records")
    by_urn = {d["urn"]: d for d in deps}
    rows = []
    for _, r in pg.iterrows():
        acc, genes = pg_proteins(r["UniProt_ID"], up)
        found = {}
        for d in deps:
            prot = protein_match(acc, genes, d["accessions"], d["genes"])
            paper = paper_evidence(r, d["pubs"]) if prot else ""
            if paper:
                found[d["urn"]] = (prot, paper)
        words = _pg_words(r)
        for u in current_versions(list(found), links):
            d = by_urn[u]
            rows.append({"DMS_id": r["DMS_id"], "urn": u, "protein": found[u][0],
                         "paper": found[u][1],
                         "phenotype_shared": len(words & phenotype_words(d["title"],
                                                                         d["description"]))})
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


# ==========================================================================
# pairing: the data decide
# ==========================================================================
CHECK_COLUMNS = ["check", "abs_rho", "rho", "coverage", "shared_variants",
                 "pg_single_mutants", "deposit_single_mutants", "offset",
                 "wild_type_agreement"]


def reproduces(c: dict) -> bool:
    """Does a checked candidate reproduce ProteinGym's scores?"""
    rho, cov = c.get("abs_rho"), c.get("coverage")
    return (c.get("check") == "checked" and rho is not None and np.isfinite(rho)
            and rho >= DATA_MATCH and cov is not None and cov >= DATA_COVERAGE)


def _pair_basis(c, note: str = "") -> str:
    data = (f"data: |rho| {c['abs_rho']:.4f} on {int(c['shared_variants'])} of "
            f"{int(c['pg_single_mutants'])} single mutants" if c["check"] == "checked" else "")
    return "; ".join(p for p in (f"protein: {c['protein']}", f"paper: {c['paper']}", data, note)
                     if p)


def _best_attempt(mine: pd.DataFrame) -> str:
    chk = mine[mine["check"] == "checked"]
    if not len(chk):
        return "no candidate's data could be compared: " + "; ".join(sorted(set(mine["check"])))
    b = chk.sort_values("abs_rho", ascending=False, na_position="last").iloc[0]
    return f"closest {b['urn']}: |rho| {b['abs_rho']:.4f}, coverage {b['coverage']:.2f}"


def pair(pg: pd.DataFrame, cands: pd.DataFrame, check) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decide each ProteinGym experiment from its candidates (rules in the module
    docstring). `check(dms_id, urn)` returns `data_check`'s dict with 'check' =
    'checked' and 'coverage', or {'check': <why the data cannot be compared>}.
    Returns (one row per experiment, one row per candidate with its check)."""
    rows = []
    for c in cands.to_dict("records"):
        res = check(c["DMS_id"], c["urn"])
        rows.append({**c, **{k: res.get(k) for k in CHECK_COLUMNS}})
    cc = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS + CHECK_COLUMNS)
    cc["reproduces"] = [reproduces(r) for r in rows]
    out = []
    for dms in pg["DMS_id"]:
        mine = cc[cc["DMS_id"] == dms]
        rec = {"DMS_id": dms, "urn": "", "status": "", "basis": "",
               "n_candidates": int(len(mine)), "candidates": ";".join(mine["urn"])}
        hits = mine[mine["reproduces"]]
        blind = mine[(mine["check"] != "checked") & (mine["paper"] != "none recorded")]
        if not len(mine):
            rec["status"] = "no_match"
        elif len(hits) == 1:
            rec.update(urn=hits.iloc[0]["urn"], status="paired",
                       basis=_pair_basis(hits.iloc[0]))
        elif len(hits) > 1:
            ranked = hits.sort_values(["phenotype_shared", "urn"], ascending=[False, True])
            top, nxt = (int(x) for x in ranked["phenotype_shared"].iloc[:2])
            if top > 0 and top > nxt:
                rec.update(urn=ranked.iloc[0]["urn"], status="paired",
                           basis=_pair_basis(ranked.iloc[0], f"phenotype words ({top} shared,"
                                                             f" next {nxt})"))
            else:
                rec.update(status="ambiguous",
                           basis=f"{len(hits)} candidates reproduce the scores")
        elif len(blind) == 1:
            rec.update(urn=blind.iloc[0]["urn"], status="paired",
                       basis=_pair_basis(blind.iloc[0],
                                         f"data not checked ({blind.iloc[0]['check']})"))
        elif len(blind) > 1:
            rec.update(status="ambiguous",
                       basis=f"{len(blind)} candidates whose data cannot be compared")
        else:
            rec.update(status="not_reproduced", basis=_best_attempt(mine))
        out.append(rec)
    res = pd.DataFrame(out)
    dup = res[res["status"] == "paired"]["urn"].value_counts()
    mask = res["urn"].isin(set(dup[dup > 1].index)) & (res["status"] == "paired")
    res.loc[mask, "basis"] = "deposit also paired with another experiment: " + res.loc[mask, "urn"]
    res.loc[mask, "status"] = "deposit_claimed_twice"
    res.loc[mask, "urn"] = ""
    return res, cc


# ==========================================================================
# the data check (ProteinGym score vs deposit score on shared variants)
# ==========================================================================
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
       "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
       "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
       "Tyr": "Y", "Val": "V"}
_P_SINGLE = re.compile(r"^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$")


def protein_change(hgvs_pro: str) -> tuple[str, int, str] | None:
    """'p.Ala23Pro' -> ('A', 23, 'P'); None for anything but a single missense change."""
    m = _P_SINGLE.match(str(hgvs_pro).strip())
    if not m or m.group(1) not in AA3 or m.group(3) not in AA3:
        return None
    wt, pos, mt = AA3[m.group(1)], int(m.group(2)), AA3[m.group(3)]
    return (wt, pos, mt) if wt != mt else None


def deposit_single_mutants(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean deposited score per single missense change (codon-level rows are averaged)."""
    if "hgvs_pro" not in scores.columns:
        return pd.DataFrame(columns=["wt", "pos", "mt", "score"])
    ch = scores["hgvs_pro"].map(protein_change)
    d = scores.assign(change=ch).dropna(subset=["change"])
    d = d.assign(wt=d["change"].str[0], pos=d["change"].str[1], mt=d["change"].str[2],
                 score=pd.to_numeric(d["score"], errors="coerce")).dropna(subset=["score"])
    return d.groupby(["wt", "pos", "mt"], as_index=False)["score"].mean()


def best_offset(dep: pd.DataFrame, target_seq: str, candidates=(0,)) -> tuple[int, float]:
    """The numbering shift that makes the deposit's wild-type residues agree with
    ProteinGym's target sequence, and the share of positions that then agree."""
    best = (0, -1.0)
    for s in candidates:
        idx = dep["pos"] + s - 1
        ok = idx.between(0, len(target_seq) - 1)
        if not ok.any():
            continue
        agree = float((dep.loc[ok, "wt"].values ==
                       np.array([target_seq[i] for i in idx[ok]])).mean())
        if agree > best[1]:
            best = (s, agree)
    return best


def data_check(pg_dms: pd.DataFrame, dep: pd.DataFrame, target_seq: str,
               offset_candidates=(0,)) -> dict:
    """|Spearman rho| between ProteinGym's DMS_score and the deposited score."""
    singles = pg_dms[~pg_dms["mutant"].astype(str).str.contains(":")]
    m = singles["mutant"].astype(str).str.extract(r"^([A-Z])(\d+)([A-Z])$")
    pg = pd.DataFrame({"wt": m[0], "pos": pd.to_numeric(m[1]), "mt": m[2],
                       "pg_score": singles["DMS_score"]}).dropna()
    shift, agree = best_offset(dep, target_seq, offset_candidates)
    if agree < 0.95:                 # numbering from another construct: try every shift
        span = len(target_seq) + int(dep["pos"].max())
        every = sorted(range(-span, span + 1), key=abs)
        shift, agree = best_offset(dep, target_seq, every)
    d = dep.assign(pos=dep["pos"] + shift)
    j = pg.merge(d, on=["wt", "pos", "mt"], how="inner")
    rho = float(spearmanr(j["pg_score"], j["score"]).statistic) if len(j) >= 3 else float("nan")
    return {"shared_variants": int(len(j)), "pg_single_mutants": int(len(pg)),
            "deposit_single_mutants": int(len(dep)), "offset": int(shift),
            "wild_type_agreement": agree, "rho": rho,
            "abs_rho": abs(rho) if np.isfinite(rho) else float("nan")}


class DataCheck:
    """`check(dms_id, urn)` for `pair`: ProteinGym's DMS file against the deposit's
    scores. Each file is read once."""

    def __init__(self, ref: pd.DataFrame, archive: Path | None = None,
                 scores=None) -> None:
        self.meta = ref.set_index("DMS_id")
        self.zip = zipfile.ZipFile(archive or INPUTS / "pg_dms_substitutions_v1.3.zip")
        self.names = {Path(n).name: n for n in self.zip.namelist() if n.endswith(".csv")}
        self.scores = scores or load_scores
        self._pg: dict[str, pd.DataFrame] = {}
        self._dep: dict[str, pd.DataFrame] = {}

    def __call__(self, dms: str, urn: str) -> dict:
        if urn not in self._dep:
            try:
                self._dep[urn] = deposit_single_mutants(self.scores(urn))
            except FileNotFoundError:
                return {"check": "scores not fetched"}
        dep = self._dep[urn]
        if len(dep) < MIN_SINGLES:
            return {"check": "no protein-level single changes"}
        m = self.meta.loc[dms]
        if dms not in self._pg:
            self._pg[dms] = pd.read_csv(io.BytesIO(self.zip.read(self.names[m["DMS_filename"]])))
        off = m["raw_mut_offset"]
        shifts = (0,) if pd.isna(off) else (0, int(off), -int(off))
        res = data_check(self._pg[dms], dep, m["target_seq"], shifts)
        n = res["pg_single_mutants"]
        return {"check": "checked", **res, "coverage": res["shared_variants"] / n if n else 0.0}


# ==========================================================================
# tiers
# ==========================================================================
def _ci_pair(columns: list[str]) -> tuple[str, str] | None:
    lo = [c for c in columns if re.search(r"ci", c, re.I) and re.search(r"low(er)?|lo($|_)", c, re.I)]
    hi = [c for c in columns if re.search(r"ci", c, re.I) and re.search(r"high(er)?|upper|hi($|_)", c, re.I)]
    for a in lo:
        stem = re.sub(r"low(er)?|lo($|_)", "", a, flags=re.I)
        for b in hi:
            if re.sub(r"high(er)?|upper|hi($|_)", "", b, flags=re.I) == stem:
                return a, b
    return None


def check_deposit_se(scores: pd.DataFrame, se_col: str) -> str:
    """'validated' if the SE reproduces the deposit's own 95% interval on every row,
    'ci_mismatch' if it does not, 'no_interval' if there is no interval to check."""
    pair_ = _ci_pair([c for c in scores.columns if c != se_col])
    if pair_ is None:
        return "no_interval"
    s = pd.to_numeric(scores["score"], errors="coerce")
    se = pd.to_numeric(scores[se_col], errors="coerce")
    lo = pd.to_numeric(scores[pair_[0]], errors="coerce")
    hi = pd.to_numeric(scores[pair_[1]], errors="coerce")
    up, dn = ((hi - s) / se).dropna(), ((s - lo) / se).dropna()
    up, dn = up[np.isfinite(up)], dn[np.isfinite(dn)]
    if len(up) == 0 or len(dn) == 0:
        return "no_interval"
    ok = np.allclose(up, CI_Z, atol=CI_ATOL) and np.allclose(dn, CI_Z, atol=CI_ATOL)
    return "validated" if ok else "ci_mismatch"


def tier_of(method: str, se_check: str = "") -> str:
    """'strict' or 'loose' for a deposit whose ceiling has status ok."""
    if method in ("replicates", "ci"):
        return "strict"
    if method == "se":
        return "strict" if se_check == "validated" else "loose"
    if method in ("sd_as_se", "variance"):
        return "loose"
    raise ValueError(f"unknown ceiling method {method!r}")


# ==========================================================================
# analyses
# ==========================================================================
def rank_models(rho: pd.DataFrame) -> pd.Series:
    """Mean rho per model over the experiments given, best first (ties by name)."""
    m = rho.mean(axis=0)
    return m.sort_index().sort_values(ascending=False, kind="stable")


def realisation(rho: pd.DataFrame, ceiling: pd.Series) -> pd.DataFrame:
    """rho / ceiling, only where the ceiling is at least CORRECTION_BOUND."""
    keep = ceiling >= CORRECTION_BOUND
    return rho.loc[keep].div(ceiling[keep], axis=0)


def _dist(x: pd.Series) -> dict:
    x = x.dropna()
    if not len(x):
        return {"n": 0}
    q = x.quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
    return {"n": int(len(x)), "min": q[0], "q25": q[1], "median": q[2], "q75": q[3],
            "max": q[4], "share_ge_0.9": float((x >= 0.9).mean()),
            "share_ge_0.8": float((x >= 0.8).mean()), "n_above_1": int((x > 1).sum())}


def _spearman(a: pd.Series, b: pd.Series) -> dict:
    j = pd.concat([a, b], axis=1).dropna()
    if len(j) < 3:
        return {"n": int(len(j)), "rho": None, "p": None}
    res = spearmanr(j.iloc[:, 0], j.iloc[:, 1])
    return {"n": int(len(j)), "rho": float(res.statistic), "p": float(res.pvalue)}


def analyse(rho: pd.DataFrame, ceiling: pd.Series) -> dict:
    """The three analyses for one tier. `rho`: experiments x models; `ceiling`: per
    experiment, same index."""
    ceiling = ceiling.reindex(rho.index)
    usable = ceiling >= CORRECTION_BOUND
    if not usable.any():
        return {"note": f"no experiment with a ceiling >= {CORRECTION_BOUND}",
                "n_experiments_with_ceiling": int(ceiling.notna().sum())}
    # means over different subsets of experiments are not comparable
    complete = rho.columns[rho.loc[usable].notna().all()]
    dropped = sorted(set(rho.columns) - set(complete))
    rho = rho[complete]
    ranked = rank_models(rho.loc[usable])
    best, top = ranked.index[0], list(ranked.index[:TOP])
    rho_best = rho[best]
    top_median = rho[top].median(axis=1)
    per_exp_max = rho.max(axis=1)

    one = {"best_model": best, "top_models": top,
           "n_experiments_with_ceiling": int(ceiling.notna().sum()),
           "n_experiments_ge_bound": int(usable.sum()),
           "ceiling_vs_best_model": {"all": _spearman(ceiling, rho_best),
                                     "ge_bound": _spearman(ceiling[usable], rho_best[usable])},
           "ceiling_vs_top_median": {"all": _spearman(ceiling, top_median),
                                     "ge_bound": _spearman(ceiling[usable], top_median[usable])},
           "ceiling_vs_per_experiment_max": {
               "all": _spearman(ceiling, per_exp_max),
               "ge_bound": _spearman(ceiling[usable], per_exp_max[usable])}}

    real = realisation(rho, ceiling)
    two = {m: _dist(real[m]) for m in top}

    by_rho = rank_models(rho.loc[usable])
    by_real = real.mean(axis=0).sort_index().sort_values(ascending=False, kind="stable")
    common = by_rho.index.intersection(by_real.index)
    tau = kendalltau(by_rho[common], by_real[common])
    three = {"kendall_tau": float(tau.statistic), "p": float(tau.pvalue),
             "n_models": int(len(common)),
             "top_by_mean_rho": [{"model": m, "mean_rho": float(by_rho[m])}
                                 for m in by_rho.index[:TOP]],
             "top_by_mean_realisation": [{"model": m, "mean_realisation": float(by_real[m])}
                                         for m in by_real.index[:TOP]]}
    return {"models_dropped_for_missing_rho": dropped,
            "1_ceiling_vs_rho": one, "2_realisation": two, "3_rankings": three,
            "below_bound": [{"DMS_id": i, "ceiling": float(c)}
                            for i, c in ceiling[~usable & ceiling.notna()].items()]}


# ==========================================================================
# accounting: where the ProteinGym experiments go
# ==========================================================================
def _names(s) -> set[str]:
    return {g.strip().upper() for g in str(s).split(";") if g.strip()} if isinstance(s, str) else set()


def strict_capable(ok: pd.DataFrame, survey: pd.DataFrame) -> pd.Series:
    """Could each deposit (rows of the ceiling table with an ok ceiling) enter the
    strict tier? Replicates and intervals can; a standard error can when the deposit
    also carries an interval to check it against (the survey's column list)."""
    cols = ok["urn"].map(survey.set_index("urn")["all_columns"]).fillna("")
    err = ok["error_column"].fillna("")
    has_ci = pd.Series([_ci_pair([c for c in str(cs).split(";") if c and c != e]) is not None
                        for cs, e in zip(cols, err)], index=ok.index)
    return ok["method"].isin(["replicates", "ci"]) | ((ok["method"] == "se") & has_ci)


def accounting(ref: pd.DataFrame, pairs: pd.DataFrame, uniprot: pd.DataFrame,
               ceilings: pd.DataFrame, survey: pd.DataFrame,
               metadata: pd.DataFrame) -> dict:
    """Counts of the ProteinGym experiments by where they go (module docstring)."""
    up = uniprot.set_index("entry_name") if len(uniprot) else uniprot
    status = pairs.set_index("DMS_id")["status"]
    platform = ((ref["first_author"].astype(str).str.strip().str.lower() == PLATFORM_STUDY[0])
                & (ref["year"] == PLATFORM_STUDY[1]))
    names = metadata.set_index("urn")["target_genes"].map(_names)
    ok = ceilings[(ceilings["status"] == "ok") & (ceilings["stratum"] != "designed_stability")]
    ok = ok.assign(strict_capable=strict_capable(ok, survey).values,
                   names=ok["urn"].map(names).map(lambda s: s if isinstance(s, set) else set()))
    holds = {"no deposit under the gene's names": 0,
             "deposits of the gene, none with an ok ceiling": 0,
             "an ok-ceiling deposit of the gene, from other papers only": 0}
    bound_loose, bound_strict = [], []
    for _, r in ref[~platform].iterrows():
        _, genes = pg_proteins(r["UniProt_ID"], up)
        any_dep = bool(names.map(lambda s: bool(s & genes)).any())
        mine = ok[ok["names"].map(lambda s: bool(s & genes))]
        if len(mine):
            bound_loose.append(r["DMS_id"])
        if mine["strict_capable"].any():
            bound_strict.append(r["DMS_id"])
        if status.get(r["DMS_id"]) == "no_match":
            key = ("an ok-ceiling deposit of the gene, from other papers only" if len(mine)
                   else "deposits of the gene, none with an ok ceiling" if any_dep
                   else "no deposit under the gene's names")
            holds[key] += 1
    return {
        "n_experiments": int(len(ref)),
        "stability_platform": {"n": int(platform.sum()),
                               "by_status": status[ref.loc[platform, "DMS_id"]]
                               .value_counts().to_dict()},
        "outside_platform": {"n": int((~platform).sum()),
                             "by_status": status[ref.loc[~platform, "DMS_id"]]
                             .value_counts().to_dict(),
                             "no_match_by_what_mavedb_holds": holds},
        "bound_by_gene": {"loose_tier": len(bound_loose), "strict_tier": len(bound_strict),
                          "strict_tier_experiments": sorted(bound_strict)},
        "strict_capable_deposits": {"n": int(ok["strict_capable"].sum()),
                                    "by_method": ok.loc[ok["strict_capable"], "method"]
                                    .value_counts().to_dict()},
        "note": "gene-level counts match target names only (results/mavedb_metadata_v1.tsv)",
    }


# ==========================================================================
# putting it together
# ==========================================================================
def load_scores(urn: str) -> pd.DataFrame:
    path = INPUTS / _urn_file(urn, "scores")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run with --fetch")
    return pd.read_csv(path, low_memory=False)


def paired_with_ceilings(pairs: pd.DataFrame, checked: pd.DataFrame) -> pd.DataFrame:
    """The paired experiments, each with its data check and its deposit's row of the
    ceiling table."""
    c = pd.read_csv(CEILINGS, sep="\t")
    keep = ["urn", "method", "error_column", "status", "ceiling", "reliability_raw",
            "n_variants", "stratum"]
    p = pairs[pairs["status"] == "paired"][["DMS_id", "urn", "basis"]]
    p = p.merge(checked[["DMS_id", "urn", "paper", "check", "abs_rho", "coverage",
                         "shared_variants", "pg_single_mutants"]],
                on=["DMS_id", "urn"], how="left")
    return p.merge(c[keep].rename(columns={"status": "ceiling_status",
                                           "n_variants": "ceiling_n"}),
                   on="urn", how="left")


def fetch() -> None:
    """Download every input not yet stored. Network; safe to rerun after a failure."""
    INPUTS.mkdir(parents=True, exist_ok=True)
    fetch_proteingym()
    ref = load_reference()
    fetch_uniprot(ref["UniProt_ID"].tolist())
    _all_or_raise(fetch_mavedb_records(candidate_urns()), "record")
    # the data check reads the scores of every candidate
    cands = candidate_table(ref, load_deposits(), load_uniprot())
    _all_or_raise(fetch_mavedb_scores(sorted(set(cands["urn"]))), "score table")


def _all_or_raise(failed: list[str], kind: str) -> None:
    if failed:
        raise RuntimeError(f"{len(failed)} MaveDB {kind} files not fetched; rerun with "
                           f"--fetch (stored files are kept). First: {failed[0]}")


def missing_records() -> list[str]:
    """Deposits of the ceiling table whose MaveDB record is not stored."""
    return [u for u in candidate_urns() if not (INPUTS / _urn_file(u, "record")).exists()]


def missing_scores(urns: list[str]) -> list[str]:
    return [u for u in urns if not (INPUTS / _urn_file(u, "scores")).exists()]


def assign_tiers(paired: pd.DataFrame) -> pd.DataFrame:
    p = paired.copy()
    p["se_check"] = ""
    for i, r in p[(p["ceiling_status"] == "ok") & (p["method"] == "se")].iterrows():
        p.at[i, "se_check"] = check_deposit_se(load_scores(r["urn"]), r["error_column"])
    p["tier"] = [tier_of(m, s) if st == "ok" else ""
                 for m, s, st in zip(p["method"], p["se_check"], p["ceiling_status"])]
    return p


def run() -> dict:
    missing = missing_records()
    if missing:      # pairing on part of the records would silently lose pairs
        raise FileNotFoundError(f"{len(missing)} MaveDB records missing (first {missing[0]}); "
                                f"rerun with --fetch")
    ref = load_reference()
    uniprot = load_uniprot()
    cands = candidate_table(ref, load_deposits(), uniprot)
    absent = missing_scores(sorted(set(cands["urn"])))
    if absent:       # an unchecked candidate would decide a pair by default
        raise FileNotFoundError(f"{len(absent)} score tables missing (first {absent[0]}); "
                                f"rerun with --fetch")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    sp = load_spearman()
    models = model_columns(sp)
    pairs, checked = pair(ref, cands, DataCheck(ref))
    paired = assign_tiers(paired_with_ceilings(pairs, checked))
    manifest = load_manifest(INPUTS / "MANIFEST.json")["files"]

    out = {"inputs": {k: {kk: v.get(kk) for kk in ("url", "version", "sha256", "bytes",
                                                    "fetched_utc")}
                      for k, v in manifest.items() if k.startswith("pg_")},
           "n_inputs": {"mavedb_records": sum(k.startswith("mavedb/") for k in manifest),
                        "mavedb_score_tables": sum(k.startswith("mavedb_scores/")
                                                   for k in manifest),
                        "uniprot_batches": sum(k.startswith("uniprot/") for k in manifest)},
           "thresholds": {"DATA_MATCH": DATA_MATCH, "DATA_COVERAGE": DATA_COVERAGE,
                          "TITLE_MATCH": TITLE_MATCH, "TITLE_PREFIX": TITLE_PREFIX,
                          "MIN_SINGLES": MIN_SINGLES, "CORRECTION_BOUND": CORRECTION_BOUND},
           "n_proteingym_experiments": int(len(ref)), "n_models": len(models),
           "n_candidates": int(len(checked)),
           "pairing": pairs["status"].value_counts().to_dict(),
           "accounting": accounting(ref, pairs, uniprot, pd.read_csv(CEILINGS, sep="\t"),
                                    pd.read_csv(SURVEY, sep="\t"),
                                    pd.read_csv(METADATA, sep="\t")),
           "paired_by_ceiling_status": paired["ceiling_status"].fillna("not in table")
                                       .value_counts().to_dict(),
           "paired_by_tier_and_method": paired[paired["tier"] != ""]
                                        .groupby(["tier", "method", "se_check"]).size()
                                        .rename("n").reset_index().to_dict("records"),
           "tiers": {}}
    for tier, allowed in (("strict", {"strict"}), ("loose", {"strict", "loose"})):
        t = paired[paired["tier"].isin(allowed)].set_index("DMS_id")
        rho = sp.loc[t.index, models]
        out["tiers"][tier] = {"n_experiments": int(len(t)),
                              **analyse(rho, t["ceiling"].astype(float))}
        t.join(rho.max(axis=1).rename("max_rho")).to_csv(
            SCRATCH / f"experiments_{tier}.tsv", sep="\t")
    pairs.to_csv(SCRATCH / "pairs.tsv", sep="\t", index=False)
    checked.to_csv(SCRATCH / "candidates.tsv", sep="\t", index=False)
    paired.to_csv(SCRATCH / "paired_with_ceilings.tsv", sep="\t", index=False)
    (SCRATCH / "summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="download the inputs first")
    args = ap.parse_args(argv)
    if args.fetch:
        fetch()
        print("[pilot] every input fetched", flush=True)
    out = run()
    print(f"[pilot] pairing: {out['pairing']}")
    for tier, t in out["tiers"].items():
        one = t.get("1_ceiling_vs_rho")
        if one is None:
            print(f"[pilot] {tier}: {t['n_experiments']} experiments; {t.get('note')}")
            continue
        print(f"[pilot] {tier}: {t['n_experiments']} experiments "
              f"({one['n_experiments_ge_bound']} with ceiling >= {CORRECTION_BOUND}); "
              f"best model {one['best_model']}; ceiling vs best rho "
              f"{one['ceiling_vs_best_model']['all']}; "
              f"Kendall tau {t['3_rankings']['kendall_tau']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
