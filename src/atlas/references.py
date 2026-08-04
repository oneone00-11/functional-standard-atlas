"""Complete the manuscript reference list from Crossref.

Volume, page and DOI fields are fetched from Crossref by title search rather
than written from memory (AGENTS.md rule 4). A match is accepted only when the
returned title is a near-exact match for the queried one; everything else is
flagged for manual completion rather than guessed at.

Usage (PYTHONPATH=src):  python -m atlas.references --out results/
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UA = "functional-standard-atlas/1.0 (mailto:cliffzhang@u.nus.edu)"
MATCH_THRESHOLD = 0.90

# (key, authors, title, fallback venue). Titles are the search key; nothing else
# is trusted until Crossref confirms it.
# Optional DOI overrides, for records where title search is ambiguous. Each DOI
# comes from a fetched source (the MaveDB deposit metadata or the Crossref record
# for the preprint), never from memory.
DOI_OVERRIDE = {
    "findlay": "10.1038/s41586-018-0461-z",   # MaveDB deposit metadata
    "brca2": "10.1038/s41586-024-08388-8",    # MaveDB deposit metadata
    "evo2": "10.1101/2025.02.18.638918",      # bioRxiv preprint, exact title match
}

REFS: list[tuple[str, str, str, str]] = [
    ("cadd", "Kircher M, Witten DM, Jain P, O'Roak BJ, Cooper GM, Shendure J",
     "A general framework for estimating the relative pathogenicity of human genetic variants", "Nat Genet. 2014"),
    ("phylop", "Pollard KS, Hubisz MJ, Rosenbloom KR, Siepel A",
     "Detection of nonneutral substitution rates on mammalian phylogenies", "Genome Res. 2010"),
    ("phastcons", "Siepel A, Bejerano G, Pedersen JS, Hinrichs AS, Hou M, Rosenbloom K, et al",
     "Evolutionarily conserved elements in vertebrate, insect, worm, and yeast genomes", "Genome Res. 2005"),
    ("spliceai", "Jaganathan K, Kyriazopoulou Panagiotopoulou S, McRae JF, Darbandi SF, Knowles D, Li YI, et al",
     "Predicting Splicing from Primary Sequence with Deep Learning", "Cell. 2019"),
    ("pangolin", "Zeng T, Li YI",
     "Predicting RNA splicing from DNA sequence using Pangolin", "Genome Biol. 2022"),
    ("alphamissense", "Cheng J, Novati G, Pan J, Bycroft C, Žemgulytė A, Applebaum T, et al",
     "Accurate proteome-wide missense variant effect prediction with AlphaMissense", "Science. 2023"),
    ("gpnmsa", "Benegas G, Albors C, Aw AJ, Ye C, Song YS",
     "A DNA language model based on multispecies alignment predicts the effects of genome-wide variants", "Nat Biotechnol. 2025"),
    ("nt", "Dalla-Torre H, Gonzalez L, Mendoza-Revilla J, Lopez Carranza N, Grzywaczewski AH, Oteri F, et al",
     "Nucleotide Transformer: building and evaluating robust foundation models for human genomics", "Nat Methods. 2025"),
    ("evo2", "Brixi G, Durrant MG, Ku J, Poli M, Brockman G, Chang D, et al",
     "Genome modeling and design across all domains of life with Evo 2", "bioRxiv. 2025"),
    ("alphagenome", "Avsec Ž, Latysheva N, Cheng J, Novati G, Taylor KR, Ward T, et al",
     "AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model", "bioRxiv. 2025"),
    ("findlay", "Findlay GM, Daza RM, Martin B, Zhang MD, Leith AP, Gasperini M, et al",
     "Accurate classification of BRCA1 variants with saturation genome editing", "Nature. 2018"),
    ("mavedb", "Esposito D, Weile J, Shendure J, Starita LM, Papenfuss AT, Roth FP, et al",
     "MaveDB: an open-source platform to distribute and interpret data from multiplexed assays of variant effect", "Genome Biol. 2019"),
    ("bap1", "Waters AJ, et al",
     "Saturation genome editing of BAP1 functionally classifies somatic and germline variants", "Nat Genet. 2024"),
    ("rad51c", "Olvera-León R, et al",
     "High-resolution functional mapping of RAD51C by saturation genome editing", "Cell. 2024"),
    ("vhl", "Buckley M, et al",
     "Saturation genome editing maps the functional spectrum of pathogenic VHL alleles", "Nat Genet. 2024"),
    ("brca2", "Huang H, et al",
     "Functional evaluation and clinical classification of BRCA2 variants", "Nature. 2025"),
    ("palb2", "Boonen RAM, Knaup SC, Menafra R, Braspenning ME, Rother MB, Kleiblova P, et al",
     "Site-saturation functional screens identify PALB2 missense variants associated with increased breast cancer risk", "Nat Commun. 2026"),
    ("proteingym", "Notin P, Kollasch AW, Ritter D, van Niekerk L, Paul S, Spinner H, et al",
     "ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design", "NeurIPS. 2023"),
    ("mutalyzer", "Lefter M, Vis JK, Vermaat M, den Dunnen JT, Taschner PEM, Laros JFJ",
     "Mutalyzer 2: next generation HGVS nomenclature checker", "Bioinformatics. 2021"),
    ("gnomad", "Genome Aggregation Database Consortium",
     "gnomAD v4.1, queried via the gnomAD GraphQL API (https://gnomad.broadinstitute.org)",
     "Broad Institute. Accessed 2026. [primary citation to be confirmed at submission]"),
    ("dersimonian", "DerSimonian R, Laird N",
     "Meta-analysis in clinical trials", "Control Clin Trials. 1986"),
    ("spearman", "Spearman C", "Correlation calculated from faulty data", "Br J Psychol. 1910"),
    ("steiger", "Steiger JH", "Tests for comparing elements of a correlation matrix",
     "Psychol Bull. 1980"),
    ("hanley", "Hanley JA, McNeil BJ",
     "The meaning and use of the area under a receiver operating characteristic (ROC) curve",
     "Radiology. 1982"),
    ("tavtigian", "Tavtigian SV, Greenblatt MS, Harrison SM, Nussbaum RL, Prabhu SA, Boucher KM, et al",
     "Modeling the ACMG/AMP variant classification guidelines as a Bayesian classification framework",
     "Genet Med. 2018"),
    ("pejaver", "Pejaver V, Byrne AB, Feng B-J, Pagel KA, Mooney SD, Karchin R, et al",
     "Calibration of computational tools for missense variant pathogenicity classification and ClinGen recommendations for PP3/BP4 criteria",
     "Am J Hum Genet. 2022"),
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def crossref(title: str, rows: int = 5) -> list[dict]:
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": rows})
    req = urllib.request.Request(f"https://api.crossref.org/works?{q}",
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as fh:
        return json.loads(fh.read().decode())["message"]["items"]


def by_doi(doi: str) -> dict:
    req = urllib.request.Request(
        f"https://api.crossref.org/works/{urllib.parse.quote(doi)}",
        headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as fh:
        return json.loads(fh.read().decode())["message"]


def _pack(best: dict, score: float) -> dict:
    issued = best.get("issued", {}).get("date-parts", [[None]])[0]
    return {
        "match_score": round(score, 3),
        "doi": best.get("DOI"),
        "journal": (best.get("container-title") or [None])[0],
        "volume": best.get("volume"),
        "page": best.get("page"),
        "year": issued[0] if issued else None,
        "type": best.get("type"),
    }


def resolve(title: str, doi: str | None = None) -> dict | None:
    if doi:
        try:
            return _pack(by_doi(doi), 1.0)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    try:
        items = crossref(title)
    except Exception as exc:  # network/rate-limit: report, never invent
        return {"error": f"{type(exc).__name__}: {exc}"}
    best, score = None, 0.0
    for it in items:
        cand = (it.get("title") or [""])[0]
        s = difflib.SequenceMatcher(None, _norm(title), _norm(cand)).ratio()
        if s > score:
            best, score = it, s
    if best is None or score < MATCH_THRESHOLD:
        return {"match_score": round(score, 3),
                "best_title": (best.get("title") or [""])[0] if best else None}
    return _pack(best, score)


def format_ref(authors: str, title: str, meta: dict | None, fallback: str) -> str:
    if not meta or "doi" not in meta:
        return f"{authors}. {title}. {fallback}. [bibliographic details to be completed]"
    venue = meta.get("journal")
    if venue:
        bits = [f"{authors}. {title}. {venue}."]
    else:
        # no container title: a preprint or proceedings. Strip any year from the
        # fallback so it is not printed twice.
        bits = [f"{authors}. {title}. {fallback.split('.')[0].strip()}."]
    if meta.get("year"):
        bits.append(f" {meta['year']}")
    if meta.get("volume"):
        bits.append(f";{meta['volume']}")
    if meta.get("page"):
        bits.append(f":{meta['page']}")
    bits.append(".")
    if meta.get("doi"):
        bits.append(f" doi:{meta['doi']}.")
    return "".join(bits)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args(argv)

    resolved, formatted = {}, []
    for key, authors, title, fallback in REFS:
        meta = resolve(title, DOI_OVERRIDE.get(key))
        resolved[key] = {"authors": authors, "title": title, "fallback": fallback,
                         "crossref": meta}
        formatted.append(format_ref(authors, title, meta, fallback))
        flag = ("OK " if meta and meta.get("doi") else "MANUAL")
        print(f"[{flag}] {key:<14} score={meta.get('match_score') if meta else '-'} "
              f"{(meta or {}).get('journal') or ''} "
              f"{(meta or {}).get('volume') or ''}:{(meta or {}).get('page') or ''}")
        time.sleep(args.sleep)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "references_v1.json").write_text(json.dumps(resolved, indent=2) + "\n")
    (out / "references_v1.txt").write_text("\n".join(
        f"{i}. {r}" for i, r in enumerate(formatted, 1)) + "\n")
    n_ok = sum(1 for v in resolved.values()
               if v["crossref"] and v["crossref"].get("doi"))
    print(f"\nresolved {n_ok}/{len(REFS)} from Crossref; the rest are flagged in the output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
