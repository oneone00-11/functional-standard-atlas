"""The coding SNVs of the five external deposits: their consequence, and for the
synonymous ones GRCh38 coordinates and SpliceAI, so that the external genes have
neutral variants as the core genes do (atlas.error_models).

The deposits of CTCF, SFPQ, RAD51D, XRCC2 and SBDS (atlas.external_reliability) name
their variants in c. on one transcript each and carry no ``hgvs_pro``; their coding
variants are not in the score matrix. Here:

  consequence   each deposit transcript's Mutalyzer model (NM_006565.4, NM_005066.3,
                NM_002878.4, NM_005431.2, ENST00000246868.7), cached in
                data/raw/reference/mutalyzer/ with its own fetch log (the directory's
                MANIFEST.json is not touched); every coding SNV (c.N X>Y) classified by
                atlas.consequence.classify_variant on the model's spliced transcript,
                which checks the reference base first (``ref_mismatch``). A RefSeq
                model is read with atlas.consequence.load_transcript; Mutalyzer's model
                of an Ensembl transcript is a genomic record, whose exons are spliced
                here. As a check, the consequence is derived a second way from the
                deposit alone: the reference base of every c. position from the SNV
                names, codons read in c. numbering (codons the deposit does not cover
                in full are ``incomplete_codon``).
  coordinates   atlas.mapping's TranscriptMap and map_variant (the atlas's c. to GRCh38
                arithmetic, the reference base checked against GRCh38): the Mutalyzer
                exons in transcript coordinates, anchored by the exons of the MANE
                Select Ensembl transcript (data/raw/reference/mane/, MANE v1.5) in
                Ensembl release 112 (data/refs/grch38_external.gtf), on the
                release-112 sequence the external scorers use
                (data/refs/grch38_external.fa). Every exon length must agree, and the
                spliced genomic sequence must be the model's transcript. The 5,095
                intronic SNVs of results/external_scores_v2.parquet are mapped the
                same way and must land on the table's coordinates.
  SpliceAI      the synonymous SNVs only, by A2's scorer, environment and parameters
                (models/spliceai/score_external.py, which runs score_fullprec.py
                unchanged; one process); first 100 of the 5,095 intronic SNVs, drawn
                with a fixed seed, are scored again and must equal the table's values
                bit for bit.

Output: results/external_coding_consequence_v1.parquet, one row per coding SNV of the
five deposits (gene, urn, variant_urn, hgvs_c, the score, its SE,
functional_pathogenicity = -score, the consequence by transcript and by the deposit's
own codons; for the synonymous SNVs variant_id, chrom, pos, ref, alt and spliceai_ds),
and its provenance record.

Subcommands (PYTHONPATH=src, from the repository root):
  fetch      the five Mutalyzer models (network)
  prepare    consequence, coordinates and the checks; writes the scoring inputs
             data/raw/external_coding_v1/{score_input,rescore_check_input}.parquet
  assemble   reads the two SpliceAI runs (models/spliceai/cache/external_coding_v1/),
             checks the 100 against the table and writes the output
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from atlas import paths
from atlas.consequence import classify_variant, load_transcript
from atlas.evaluate import classify_region
from atlas.external_reliability import EXTERNAL, load_deposit
from atlas.mapping import Exon, MappingFailure, RegionSeq, TranscriptMap, map_variant, revcomp

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
MUTALYZER = REPO / "data" / "raw" / "reference" / "mutalyzer"
MUTALYZER_API = "https://mutalyzer.nl/api/reference_model/?reference_id={}"
FETCH_LOG = MUTALYZER / "fetch_log_external_coding_v1.json"
MANE = REPO / "data" / "raw" / "reference" / "mane" / "MANE.GRCh38.v1.5.summary.txt.gz"
GTF = REPO / "data" / "refs" / "grch38_external.gtf"
FASTA = REPO / "data" / "refs" / "grch38_external.fa"
WORK = REPO / "data" / "raw" / "external_coding_v1"
SCORE_INPUT = WORK / "score_input.parquet"
CHECK_INPUT = WORK / "rescore_check_input.parquet"
SPLICEAI = REPO / "models" / "spliceai" / "cache" / "external_coding_v1"
SPLICEAI_OUT = SPLICEAI / "spliceai.parquet"
CHECK_OUT = SPLICEAI / "rescore_check.parquet"
SCORER_SHA256 = "a5056203b0e941ed7b9b53b1914c769cc42bf2a5b681e994ec6f5e2e7280b970"
A2_ENV = REPO / "models" / "spliceai" / "cache" / "external_v1" / "spliceai.parquet.environment.json"

GENES = ("CTCF", "SFPQ", "RAD51D", "XRCC2", "SBDS")
URN = {spec["gene"]: urn for urn, spec in EXTERNAL.items() if spec["gene"] in GENES}
TRANSCRIPT = {spec["gene"]: spec["transcript"] for spec in EXTERNAL.values()
              if spec["gene"] in GENES}
CODING_SNV = re.compile(r"^c\.(\d+)([ACGT])>([ACGT])$")
INTRONIC_SNV = re.compile(r"^c\.[-*]?\d+[+-]\d+[ACGT]>[ACGT]$")
PAD = 5000
CHECK_SEED = 20261126
CHECK_N = 100
USER_AGENT = "atlas-external-coding/1.0"

# the instruction's counts (Cowork, from the deposits' own codons): coding SNVs,
# synonymous, missense, nonsense
EXPECTED = {"CTCF": (4692, 943, 3498, 221), "SFPQ": (3609, 763, 2656, 182),
            "RAD51D": (2655, 678, 1874, 86), "XRCC2": (2312, 481, 1693, 121),
            "SBDS": (2141, 437, 1580, 107)}
EXPECTED_INCOMPLETE = {"CTCF": 30}

OUT_COLUMNS = ["gene", "urn", "variant_urn", "hgvs_c", "region", "score", "standard_error",
               "functional_pathogenicity", "consequence", "consequence_deposit_codons",
               "variant_id", "chrom", "pos", "ref", "alt", "spliceai_ds"]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tilde(text: str) -> str:
    """A path with the home directory written as ~."""
    return str(text).replace(str(HOME), "~")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# ==========================================================================
# fetch
# ==========================================================================
def fetch_models(out: Path = MUTALYZER, log: Path = FETCH_LOG) -> dict:
    """Each transcript's Mutalyzer model, once; the log records what was fetched when."""
    rec = json.loads(log.read_text()) if log.exists() else {"models": {}}
    for gene in GENES:
        acc = TRANSCRIPT[gene]
        dest = out / f"reference_model_{acc}.json"
        url = MUTALYZER_API.format(acc)
        if dest.exists():
            rec["models"].setdefault(acc, {"gene": gene, "file": dest.name,
                                           "note": "already cached; not fetched again"})
            continue
        r = requests.get(url, timeout=120, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        json.loads(r.content)                         # a model, not an error page
        dest.write_bytes(r.content)
        rec["models"][acc] = {"gene": gene, "file": dest.name, "url": url,
                              "fetched_utc": utc(), "bytes": len(r.content),
                              "sha256": hashlib.sha256(r.content).hexdigest()}
        print(f"[fetch] {acc}: {len(r.content)} bytes", flush=True)
    log.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


# ==========================================================================
# the transcript models
# ==========================================================================
def _mrna(model: dict, accession: str) -> dict:
    found = []

    def walk(feats):
        for f in feats:
            if f.get("type") == "mRNA" and str(f.get("id", "")).split(".")[0] == \
                    accession.split(".")[0]:
                found.append(f)
            walk(f.get("features") or [])

    walk(model["annotations"].get("features") or [])
    if len(found) != 1:
        raise RuntimeError(f"{accession}: {len(found)} mRNA features match")
    return found[0]


def _pos(loc: dict) -> tuple[int, int]:
    return loc["start"]["position"], loc["end"]["position"]


def mutalyzer_transcript(accession: str) -> dict:
    """The spliced transcript of a cached Mutalyzer model: sequence, CDS offsets (0-based,
    half-open, transcript coordinates) and exons in transcript coordinates."""
    path = MUTALYZER / f"reference_model_{accession}.json"
    model = json.loads(path.read_text())
    mrna = _mrna(model, accession)
    exons = sorted((f for f in mrna["features"] if f["type"] == "exon"),
                   key=lambda f: f["location"]["start"]["position"])
    cds = next(f for f in mrna["features"] if f["type"] == "CDS")
    if accession.startswith(("NM_", "NR_")):
        seq, cs, ce = load_transcript(accession)       # atlas.consequence's reader
        t_exons = [_pos(e["location"]) for e in exons]
        kind = "RefSeq transcript record"
    else:
        # a genomic record: positions are 0-based on the chromosome; the record's
        # sequence covers its own location on the transcript's strand (for a minus-
        # strand transcript it runs from the record's end down; the comparison with
        # the genome in transcript_map checks this)
        r0, r1 = _pos(model["annotations"]["location"])
        gseq = model["sequence"]["seq"]
        if len(gseq) != r1 - r0:
            raise RuntimeError(f"{accession}: record sequence is not its location")
        strand = mrna["location"].get("strand") or exons[0]["location"].get("strand")
        order = exons if strand == 1 else exons[::-1]
        pieces, t_exons, t = [], [], 0
        c0, c1 = _pos(cds["location"])
        cs = ce = None
        for e in order:
            a, b = _pos(e["location"])
            pieces.append(gseq[a - r0:b - r0] if strand == 1 else gseq[r1 - b:r1 - a])
            # CDS ends, in transcript coordinates
            if strand == 1:
                if a <= c0 < b:
                    cs = t + (c0 - a)
                if a < c1 <= b:
                    ce = t + (c1 - a)
            else:
                if a < c1 <= b:
                    cs = t + (b - c1)
                if a <= c0 < b:
                    ce = t + (b - c0)
            t_exons.append((t, t + (b - a)))
            t += b - a
        seq = "".join(pieces).upper()
        kind = f"genomic record {r0}-{r1}, strand {strand}"
        if cs is None or ce is None:
            raise RuntimeError(f"{accession}: CDS ends not in an exon")
    return {"accession": accession, "file": path.name, "kind": kind, "seq": seq,
            "cds_start": cs, "cds_end": ce, "t_exons": t_exons,
            "sha256": sha256_file(path)}


def mane_partner(gene: str) -> dict:
    with gzip.open(MANE, "rt") as f:
        head = f.readline().lstrip("#").rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(head, line.rstrip("\n").split("\t")))
            if row["symbol"] == gene and row["MANE_status"] == "MANE Select":
                return row
    raise RuntimeError(f"{gene}: no MANE Select record")


def gtf_exons(transcript: str, gtf: Path = GTF) -> dict:
    """The exons of one Ensembl transcript (stable id with version) in the GTF."""
    stable, version = transcript.split(".")
    key = f'transcript_id "{stable}"'
    rows = []
    with open(gtf) as f:
        for line in f:
            if key in line:
                c = line.split("\t")
                if c[2] != "exon":
                    continue
                m = re.search(r'transcript_version "(\d+)"', c[8])
                rows.append({"chrom": c[0], "start": int(c[3]), "end": int(c[4]),
                             "strand": c[6], "version": m.group(1) if m else None})
    if not rows:
        raise RuntimeError(f"{transcript}: not in {gtf.name}")
    versions = {r["version"] for r in rows}
    if versions != {version}:
        raise RuntimeError(f"{transcript}: the GTF has version(s) {sorted(versions)}")
    strand = {r["strand"] for r in rows}
    if len(strand) != 1:
        raise RuntimeError(f"{transcript}: exons on both strands")
    s = strand.pop()
    rows.sort(key=lambda r: r["start"], reverse=s == "-")
    return {"chrom": rows[0]["chrom"], "strand": 1 if s == "+" else -1, "exons": rows}


class Fasta:
    """Reads a region of an indexed FASTA (the .fai beside it)."""

    def __init__(self, path: Path = FASTA):
        self.path = path
        self.index = {}
        for line in Path(str(path) + ".fai").read_text().splitlines():
            name, length, offset, lb, lw = line.split("\t")[:5]
            self.index[name] = (int(length), int(offset), int(lb), int(lw))

    def region(self, chrom: str, start: int, end: int) -> str:
        """1-based inclusive."""
        length, offset, lb, lw = self.index[chrom]
        start, end = max(1, start), min(length, end)
        a = offset + (start - 1) // lb * lw + (start - 1) % lb
        b = offset + (end - 1) // lb * lw + (end - 1) % lb
        with open(self.path, "rb") as f:
            f.seek(a)
            raw = f.read(b - a + 1)
        return raw.replace(b"\n", b"").decode().upper()


def transcript_map(gene: str, fasta: Fasta) -> tuple[TranscriptMap, RegionSeq, dict, dict]:
    """atlas.mapping's map for one deposit transcript, with its checks."""
    acc = TRANSCRIPT[gene]
    tx = mutalyzer_transcript(acc)
    mane = mane_partner(gene)
    ens = acc if acc.startswith("ENST") else mane["Ensembl_nuc"]
    if acc.startswith("NM_") and mane["RefSeq_nuc"] != acc:
        raise RuntimeError(f"{gene}: {acc} is not MANE Select ({mane['RefSeq_nuc']})")
    if acc.startswith("ENST") and mane["Ensembl_nuc"] != acc:
        raise RuntimeError(f"{gene}: {acc} is not MANE Select ({mane['Ensembl_nuc']})")
    g = gtf_exons(ens)
    if g["strand"] != (1 if mane["chr_strand"] == "+" else -1):
        raise RuntimeError(f"{gene}: strand differs between MANE and the GTF")
    if len(g["exons"]) != len(tx["t_exons"]):
        raise RuntimeError(f"{gene}: {len(tx['t_exons'])} exons in the model, "
                           f"{len(g['exons'])} in the GTF")
    exons = []
    for (ts, te), e in zip(tx["t_exons"], g["exons"]):
        if te - ts != e["end"] - e["start"] + 1:
            raise RuntimeError(f"{gene}: an exon's length differs between the model and "
                               f"the GTF ({te - ts} vs {e['end'] - e['start'] + 1})")
        exons.append(Exon(t_start=ts, t_end=te, g_start=e["start"], g_end=e["end"]))
    tmap = TranscriptMap(exons, tx["cds_start"], tx["cds_end"], g["strand"], g["chrom"])
    lo = min(e["start"] for e in g["exons"]) - PAD
    hi = max(e["end"] for e in g["exons"]) + PAD
    rseq = RegionSeq(lo, fasta.region(g["chrom"], lo, hi))
    spliced = "".join(rseq.span(e.g_start, e.g_end) if g["strand"] == 1
                      else revcomp(rseq.span(e.g_start, e.g_end)) for e in exons)
    differ = [i for i, (a, b) in enumerate(zip(spliced, tx["seq"])) if a != b]
    check = {"model": tx["file"], "model_kind": tx["kind"], "model_sha256": tx["sha256"],
             "ensembl_anchor": ens, "mane": {k: mane[k] for k in (
                 "RefSeq_nuc", "Ensembl_nuc", "GRCh38_chr", "chr_strand")},
             "chrom": g["chrom"], "strand": g["strand"], "n_exons": len(exons),
             "cds_t_offsets": [tx["cds_start"], tx["cds_end"]],
             "transcript_length": len(tx["seq"]),
             "spliced_genome_equals_model": len(spliced) == len(tx["seq"]) and not differ,
             "spliced_genome_bases_differing": len(differ),
             "cds_bases_differing": sum(tx["cds_start"] <= i < tx["cds_end"] for i in differ)}
    return tmap, rseq, check, tx


# ==========================================================================
# the consequence, two ways
# ==========================================================================
CODE = {a + b + c: aa for (a, b, c), aa in zip(
    itertools.product("TCAG", repeat=3),
    "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")}


def deposit_codons(hgvs: pd.Series) -> tuple[pd.Series, int]:
    """The consequence from the deposit alone (the reference base of each c. position
    from its SNV names, codons in c. numbering), and the positions named with two
    different reference bases."""
    m = hgvs.str.extract(CODING_SNV)
    m.columns = ["pos", "ref", "alt"]
    ok = m["pos"].notna()
    pos = m.loc[ok, "pos"].astype(int)
    ref_sets = m.loc[ok].assign(pos=pos).groupby("pos")["ref"].agg(set)
    conflicts = int((ref_sets.map(len) > 1).sum())
    ref = ref_sets.map(lambda s: next(iter(s)))
    out = pd.Series(np.nan, index=hgvs.index, dtype=object)
    for i, p, r, a in zip(pos.index, pos, m.loc[ok, "ref"], m.loc[ok, "alt"]):
        start = 3 * ((p - 1) // 3) + 1
        cod = [ref.get(start + j) for j in range(3)]
        if None in cod:
            out[i] = "incomplete_codon"
            continue
        alt = cod.copy()
        alt[p - start] = a
        ra, aa = CODE["".join(cod)], CODE["".join(alt)]
        out[i] = ("synonymous" if ra == aa else "nonsense" if aa == "*" else
                  "stop_lost" if ra == "*" else
                  "start_lost" if (start == 1 and ra == "M") else "missense")
    return out, conflicts


def prepare() -> tuple[pd.DataFrame, dict]:
    fasta = Fasta()
    ext = pd.read_parquet(paths.path("external_scores"),
                          columns=["variant_id", "gene", "variant_urn", "hgvs_c", "chrom",
                                   "pos", "ref", "alt", "spliceai_ds"])
    parts, record = [], {"genes": {}}
    for gene in GENES:
        urn = URN[gene]
        tmap, rseq, check, tx = transcript_map(gene, fasta)
        dep = load_deposit(urn).rename(columns={"accession": "variant_urn"})
        snv_c = dep["hgvs_c"].str.match(CODING_SNV.pattern)
        coding = dep[snv_c].copy()
        coding["consequence"] = [classify_variant(h, tx["seq"], tx["cds_start"],
                                                  tx["cds_end"]) for h in coding["hgvs_c"]]
        coding["consequence_deposit_codons"], conflicts = deposit_codons(coding["hgvs_c"])
        # every coding SNV mapped to GRCh38 (its reference base checked there)
        mapped, failures = [], {}
        for h in coding["hgvs_c"]:
            try:
                mapped.append(map_variant(tmap, rseq, h))
            except MappingFailure as exc:
                failures[exc.category] = failures.get(exc.category, 0) + 1
                mapped.append((np.nan, None, None))
        coding["pos"] = [p for p, _, _ in mapped]
        coding["ref"] = [r for _, r, _ in mapped]
        coding["alt"] = [a for _, _, a in mapped]
        coding["chrom"] = tmap.chrom
        # the intronic SNVs of external_scores_v2, mapped the same way
        e = ext[ext["gene"] == gene]
        got = []
        for h in e["hgvs_c"]:
            try:
                got.append(map_variant(tmap, rseq, h))
            except MappingFailure as exc:
                got.append((None, exc.category, None))
        same = [(str(c) == tmap.chrom and p == int(pp) and r == rr and a == aa)
                for (p, r, a), c, pp, rr, aa in zip(got, e["chrom"], e["pos"], e["ref"],
                                                    e["alt"])]
        comp = coding["consequence_deposit_codons"]
        complete = comp.ne("incomplete_codon")
        agree = coding.loc[complete, "consequence"].eq(comp[complete])
        counts = coding["consequence"].value_counts().to_dict()
        rec = {"urn": urn, "transcript": TRANSCRIPT[gene], **check,
               "coding_snvs": int(len(coding)),
               "consequence": {k: int(v) for k, v in sorted(counts.items())},
               "ref_mismatch": int(counts.get("ref_mismatch", 0)),
               "deposit_reference_conflicts": conflicts,
               "deposit_codons": {k: int(v) for k, v in comp.value_counts().sort_index().items()},
               "complete_codon_rows": int(complete.sum()),
               "complete_codon_disagreements": int((~agree).sum()),
               "incomplete_codon": coding.loc[~complete, ["variant_urn", "hgvs_c",
                                                          "consequence"]].to_dict("records"),
               "genomic_mapping_failures": failures,
               "intronic_snvs_in_external_scores": int(len(e)),
               "intronic_snvs_mapped_to_the_table": int(sum(same))}
        exp = EXPECTED[gene]
        rec["expected_from_instruction"] = {"coding_snvs": exp[0], "synonymous": exp[1],
                                            "missense": exp[2], "nonsense": exp[3],
                                            "incomplete_codon": EXPECTED_INCOMPLETE.get(gene, 0)}
        record["genes"][gene] = rec
        parts.append(coding.assign(gene=gene, urn=urn))
    table = pd.concat(parts, ignore_index=True)
    table["region"] = table["hgvs_c"].map(classify_region)
    table["score"] = pd.to_numeric(table["score"], errors="coerce")
    table["standard_error"] = pd.to_numeric(table["standard_error"], errors="coerce")
    table["functional_pathogenicity"] = -table["score"]
    syn = table["consequence"].eq("synonymous")
    table["pos"] = table["pos"].astype("Int64")
    table["variant_id"] = np.where(
        syn, table["chrom"] + "-" + table["pos"].astype(str) + "-" + table["ref"].astype(str)
        + "-" + table["alt"].astype(str), None)
    for c in ("chrom", "pos", "ref", "alt"):
        table[c] = table[c].where(syn)
    record["failures"] = failures_of(record)
    return table, record


def failures_of(record: dict) -> list[str]:
    out = []
    for gene, r in record["genes"].items():
        exp = r["expected_from_instruction"]
        cons = r["consequence"]
        dc = r["deposit_codons"]
        if r["coding_snvs"] != exp["coding_snvs"]:
            out.append(f"{gene}: {r['coding_snvs']} coding SNVs, the instruction "
                       f"{exp['coding_snvs']}")
        for k in ("synonymous", "missense", "nonsense"):
            if dc.get(k, 0) != exp[k]:
                out.append(f"{gene}: the deposit's codons give {dc.get(k, 0)} {k}, the "
                           f"instruction {exp[k]}")
        if dc.get("incomplete_codon", 0) != exp["incomplete_codon"]:
            out.append(f"{gene}: {dc.get('incomplete_codon', 0)} on incomplete codons, the "
                       f"instruction {exp['incomplete_codon']}")
        if r["complete_codon_disagreements"]:
            out.append(f"{gene}: {r['complete_codon_disagreements']} consequences differ "
                       "on complete codons")
        if r["ref_mismatch"]:
            out.append(f"{gene}: ref_mismatch {r['ref_mismatch']}")
        if r["genomic_mapping_failures"]:
            out.append(f"{gene}: mapping failures {r['genomic_mapping_failures']}")
        if not r["spliced_genome_equals_model"]:
            out.append(f"{gene}: the spliced genome differs from the model "
                       f"({r['spliced_genome_bases_differing']} bases)")
        if r["intronic_snvs_mapped_to_the_table"] != r["intronic_snvs_in_external_scores"]:
            out.append(f"{gene}: {r['intronic_snvs_mapped_to_the_table']} of "
                       f"{r['intronic_snvs_in_external_scores']} intronic SNVs land on "
                       "the table's coordinates")
        if cons.get("synonymous", 0) != dc.get("synonymous", 0) + sum(
                1 for x in r["incomplete_codon"] if x["consequence"] == "synonymous"):
            out.append(f"{gene}: the synonymous count by transcript is not the deposit's "
                       "plus the incomplete codons'")
    return out


def score_inputs(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The synonymous SNVs to score, and the 100 intronic SNVs scored again."""
    cols = ["variant_id", "gene", "chrom", "pos", "ref", "alt", "variant_urn"]
    syn = table[table["consequence"].eq("synonymous")][cols].copy()
    syn["pos"] = syn["pos"].astype(int)
    syn["stratum"] = "coding"
    if syn["variant_id"].duplicated().any():
        raise RuntimeError("a synonymous variant_id comes twice")
    a2 = pd.read_parquet(REPO / "data" / "raw" / "external_v1" / "score_input.parquet")
    pick = np.sort(np.random.default_rng(CHECK_SEED).choice(len(a2), CHECK_N, replace=False))
    return syn.reset_index(drop=True), a2.iloc[pick].reset_index(drop=True)


# ==========================================================================
# assemble
# ==========================================================================
def _env(path: Path) -> dict:
    """The run's environment record, the home directory written as ~ (the record is
    rewritten so; nothing else in it changes)."""
    p = Path(str(path) + ".environment.json")
    text = p.read_text()
    if str(HOME) in text:
        p.write_text(tilde(text))
    return json.loads(p.read_text())


def assemble(table: pd.DataFrame, record: dict) -> tuple[pd.DataFrame, dict]:
    ext = pd.read_parquet(paths.path("external_scores"), columns=["variant_id", "spliceai_ds"])
    chk = pd.read_parquet(CHECK_OUT).merge(ext, on="variant_id", suffixes=("", "_table"))
    bits = (chk["spliceai_ds"].to_numpy(np.float64).view(np.uint64)
            == chk["spliceai_ds_table"].to_numpy(np.float64).view(np.uint64))
    record["rescore_check"] = {
        "drawn_from": "data/raw/external_v1/score_input.parquet (the 5,095 intronic SNVs)",
        "seed": CHECK_SEED, "n": int(len(chk)), "bitwise_equal": int(bits.sum()),
        "max_abs_diff": float(np.abs(chk["spliceai_ds"] - chk["spliceai_ds_table"]).max()),
        "environment": {k: v for k, v in _env(CHECK_OUT).items() if k != "pip_freeze"}}
    if bits.sum() != len(chk) or len(chk) != CHECK_N:
        record["failures"].append(f"SpliceAI scored again: {bits.sum()} of {len(chk)} equal")
    sp = pd.read_parquet(SPLICEAI_OUT)[["variant_id", "spliceai_ds"]]
    env = _env(SPLICEAI_OUT)
    out = table.merge(sp, on="variant_id", how="left")
    syn = out["consequence"].eq("synonymous")
    record["spliceai"] = {"scored": int(out.loc[syn, "spliceai_ds"].notna().sum()),
                          "synonymous": int(syn.sum()),
                          "nan": int(out.loc[syn, "spliceai_ds"].isna().sum()),
                          "environment": {k: v for k, v in env.items() if k != "pip_freeze"},
                          "pip_freeze_lines": len(env.get("pip_freeze", []))}
    if record["spliceai"]["scored"] != record["spliceai"]["synonymous"]:
        record["failures"].append("a synonymous SNV has no SpliceAI value")
    # A2's scorer, driver and environment (models/spliceai/cache/external_v1/)
    a2 = json.loads((A2_ENV).read_text())
    same = {"scorer_sha256": env["scorer_sha256"] == a2["scorer_sha256"] == SCORER_SHA256,
            "driver_sha256": env["driver_sha256"] == a2["driver_sha256"],
            "configuration": env["configuration"] == a2["configuration"],
            "pip_freeze": env.get("pip_freeze") == a2.get("pip_freeze"),
            "reference_sha256": env["reference"]["sha256"] == a2["reference"]["sha256"]}
    record["same_as_a2_run"] = same
    if not all(same.values()):
        record["failures"].append(f"the SpliceAI run differs from A2's: {same}")
    return out[OUT_COLUMNS], record


def provenance(record: dict, out_path: Path) -> dict:
    fetch = json.loads(FETCH_LOG.read_text())
    return {"written_by": "atlas.external_coding", "output": out_path.name,
            "output_sha256": sha256_file(out_path),
            "inputs": {"deposits": {g: URN[g] for g in GENES},
                       "external_scores": paths.name("external_scores"),
                       "mane": tilde(MANE.relative_to(REPO)), "gtf": "data/refs/grch38_external.gtf",
                       "fasta": "data/refs/grch38_external.fa",
                       "fasta_sha256": record.get("fasta_sha256")},
            "mutalyzer_fetch": fetch, **record}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "prepare", "assemble"])
    ap.add_argument("--out", default=str(paths.RESULTS))
    args = ap.parse_args(argv)
    if args.step == "fetch":
        print(json.dumps(fetch_models(), indent=1))
        return 0
    table, record = prepare()
    if args.step == "prepare":
        syn, check = score_inputs(table)
        WORK.mkdir(parents=True, exist_ok=True)
        syn.to_parquet(SCORE_INPUT, index=False)
        check.to_parquet(CHECK_INPUT, index=False)
        (WORK / "prepare_record.json").write_text(json.dumps(record, indent=1, default=str)
                                                  + "\n")
        print(json.dumps({g: {k: r[k] for k in ("coding_snvs", "consequence",
                                                "deposit_codons",
                                                "complete_codon_disagreements",
                                                "intronic_snvs_mapped_to_the_table",
                                                "spliced_genome_equals_model")}
                          for g, r in record["genes"].items()}, indent=1))
        print(f"[prepare] {len(syn)} synonymous SNVs to score, {len(check)} to score again; "
              f"failures: {record['failures']}")
        return 1 if record["failures"] else 0
    record["fasta_sha256"] = sha256_file(FASTA)
    out_table, record = assemble(table, record)
    if record["failures"]:
        print("failures:", record["failures"], file=sys.stderr)
        return 1
    out = Path(args.out)
    path = out / paths.name("external_coding_consequence")
    out_table.to_parquet(path, index=False)
    prov = provenance(record, path)
    (out / paths.name("external_coding_consequence_provenance")).write_text(
        tilde(json.dumps(prov, indent=1, default=str)) + "\n")
    print(f"[assemble] wrote {path.name}: {len(out_table)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
