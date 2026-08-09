"""Map MaveDB c. HGVS variants to GRCh38 genomic coordinates and freeze the matrix.

Provenance rules (project conventions): every external identifier is fetched live from the
source API by this script and hash-registered; nothing is written from memory.

Sources used:
  - MANE Select transcripts : NCBI MANE summary table (URL + version recorded).
  - Transcript exon models  : Mutalyzer3 ``/api/reference_model`` (per assay
                              accession, exact version). Mutalyzer is the
                              authority for c. -> transcript-position semantics.
  - Genomic exon anchors    : Ensembl REST ``/lookup/id/{id}?expand=1`` (GRCh38).
                              Mutalyzer exon lengths must match Ensembl exon
                              lengths element-wise or the dataset is aborted.
  - Reference sequence      : Ensembl REST ``/sequence/region/human/...`` (GRCh38,
                              + strand), one slice per gene locus.
  - Live cross-validation   : Ensembl VEP ``/vep/human/hgvs`` (an independent
                              mapper) on a seeded random sample per dataset;
                              dbSNP rsIDs and ClinVar significance come from
                              VEP's colocated variants.

Orientation rule (project conventions "larger = more damaging"):
  functional_pathogenicity = -score   (SGE: higher raw score = fitter/normal)

Coordinate conventions in the frozen matrix:
  - ``chrom`` is the GRCh38 chromosome name (e.g. "17"), ``pos`` is 1-based.
  - SNVs: ``pos`` is the substituted base.
  - Indels are left-anchored VCF-style: ``pos`` is the base immediately before
    the event, ``ref`` = anchor + deleted, ``alt`` = anchor + inserted.
    Alleles are always on the genomic + strand.

Usage (PYTHONPATH=src):

    python -m atlas.mapping [--config config/assays.yaml] [--sample-size 10]
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import yaml

from atlas.manifest import register_file

# --------------------------------------------------------------------------
# Constants (endpoints, not identifiers)
# --------------------------------------------------------------------------

MANE_SUMMARY_URL = (
    "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/"
    "MANE.GRCh38.v1.5.summary.txt.gz"
)
MUTALYZER_API = "https://mutalyzer.nl/api"
ENSEMBL_REST = "https://rest.ensembl.org"

RAW_MAVEDB = Path("data/raw/mavedb")
REF_CACHE = Path("data/raw/reference")
FROZEN_DIR = Path("data/frozen")
RESULTS_DIR = Path("results")

FROZEN_VERSION = "v1"
ORIENTATION_RULE = "functional_pathogenicity = -score (SGE: higher raw score = more normal function)"

REGION_PAD = 5000  # bp of flanking sequence fetched around each gene locus
HTTP_TIMEOUT = 120

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

COLUMNS = [
    "variant_id",
    "gene",
    "urn",
    "chrom",
    "pos",
    "ref",
    "alt",
    "transcript",
    "hgvs_c",
    "functional_pathogenicity",
    "mapping_status",
]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def http_get(url: str, *, retries: int = 4, headers: dict | None = None) -> requests.Response:
    """GET with polite retry/backoff; raises on permanent failure."""
    hdrs = {"User-Agent": "functional-standard-atlas/0.1 (academic benchmark)"}
    if headers:
        hdrs.update(headers)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT, headers=hdrs)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:  # noqa: PERF203
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}: {last_exc}")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Cached, manifest-registered source downloads (data/raw/reference/)
# --------------------------------------------------------------------------


def cache_fetch(url: str, rel_path: str, *, binary: bool = False, **meta) -> Path:
    """Fetch ``url`` once into data/raw/reference/<rel_path> and register it.

    Files under data/raw are immutable: an existing cached file is reused
    as-is (its hash is already in the manifest).
    """
    dest = REF_CACHE / rel_path
    manifest_path = REF_CACHE / "MANIFEST.json"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = http_get(url)
        dest.write_bytes(r.content)
        register_file(manifest_path, dest, source_url=url, fetched_at=utcnow(), **meta)
    return dest


def fetch_mane_summary() -> Path:
    return cache_fetch(
        MANE_SUMMARY_URL,
        "mane/MANE.GRCh38.v1.5.summary.txt.gz",
        source="NCBI MANE project",
        description="MANE summary table, GRCh38",
    )


def fetch_mutalyzer_model(accession: str) -> dict:
    dest = cache_fetch(
        f"{MUTALYZER_API}/reference_model/?reference_id={accession}",
        f"mutalyzer/reference_model_{accession}.json",
        source="Mutalyzer3 /api/reference_model",
        accession=accession,
    )
    return json.loads(dest.read_text(encoding="utf-8"))


def fetch_ensembl_lookup(stable_id: str) -> dict:
    dest = cache_fetch(
        f"{ENSEMBL_REST}/lookup/id/{stable_id}?expand=1;content-type=application/json",
        f"ensembl/lookup_{stable_id}.json",
        source="Ensembl REST /lookup/id (GRCh38)",
        accession=stable_id,
    )
    return json.loads(dest.read_text(encoding="utf-8"))


def fetch_ensembl_region(chrom: str, start: int, end: int) -> tuple[int, int, str]:
    """Return (start, end, +strand sequence) for chrom:start..end (1-based)."""
    dest = cache_fetch(
        f"{ENSEMBL_REST}/sequence/region/human/{chrom}:{start}..{end}:1?content-type=text/plain",
        f"ensembl/seq_GRCh38_chr{chrom}_{start}_{end}.fa",
        source="Ensembl REST /sequence/region (GRCh38, + strand)",
        chrom=str(chrom),
    )
    seq = dest.read_text(encoding="utf-8").strip().upper()
    return start, end, seq


# --------------------------------------------------------------------------
# MANE summary
# --------------------------------------------------------------------------


@dataclass
class ManeRecord:
    symbol: str
    refseq_nuc: str
    ensembl_nuc: str
    grch38_chr_accession: str  # e.g. NC_000017.11
    chr_name: str  # e.g. "17"
    strand: str  # '+' or '-'
    gene_start: int
    gene_end: int


def load_mane_records(genes: list[str]) -> dict[str, ManeRecord]:
    path = fetch_mane_summary()
    records: dict[str, ManeRecord] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            if row["symbol"] in genes and row["MANE_status"] == "MANE Select":
                nc = row["GRCh38_chr"]
                chr_name = nc.split(".")[0].replace("NC_0000", "").lstrip("0")
                records[row["symbol"]] = ManeRecord(
                    symbol=row["symbol"],
                    refseq_nuc=row["RefSeq_nuc"],
                    ensembl_nuc=row["Ensembl_nuc"],
                    grch38_chr_accession=nc,
                    chr_name=chr_name,
                    strand=row["chr_strand"],
                    gene_start=int(row["chr_start"]),
                    gene_end=int(row["chr_end"]),
                )
    missing = set(genes) - set(records)
    if missing:
        raise RuntimeError(f"No MANE Select record for: {sorted(missing)}")
    return records


# --------------------------------------------------------------------------
# Transcript model: Mutalyzer (transcript coords) + Ensembl (genomic anchor)
# --------------------------------------------------------------------------


@dataclass
class Exon:
    t_start: int  # 0-based, half-open transcript coords (Mutalyzer model)
    t_end: int
    g_start: int  # 1-based, inclusive genomic coords (Ensembl)
    g_end: int


class TranscriptMap:
    """c. position -> GRCh38 position, from a Mutalyzer exon model anchored
    to the genome by Ensembl exon coordinates."""

    def __init__(
        self,
        exons: list[Exon],  # in transcript order
        cds_start: int,  # 0-based transcript offset of c.1
        cds_end: int,  # 0-based transcript offset just past the stop codon
        strand_sign: int,  # +1 / -1 on the genome
        chrom: str,
    ):
        self.exons = exons
        self.cds_start = cds_start
        self.cds_end = cds_end
        self.strand_sign = strand_sign
        self.chrom = chrom

    def _t_offset(self, tok: "CPosition") -> int:
        if tok.star:
            return self.cds_end + tok.num - 1
        if tok.five_prime:
            return self.cds_start - tok.num
        return self.cds_start + tok.num - 1

    def g_position(self, tok: "CPosition") -> int:
        """Genomic 1-based coordinate of the base described by ``tok``."""
        t = self._t_offset(tok)
        exon = next((e for e in self.exons if e.t_start <= t < e.t_end), None)
        if exon is None:
            raise MappingFailure("out_of_transcript", f"transcript offset {t} not exonic")
        if self.strand_sign == 1:
            g_base = exon.g_start + (t - exon.t_start)
        else:
            g_base = exon.g_end - (t - exon.t_start)
        if tok.offset:
            g_base += self.strand_sign * tok.offset
        return g_base


def build_transcript_map(
    assay_accession: str,
    mane: ManeRecord,
) -> tuple[TranscriptMap, dict]:
    """Fetch Mutalyzer + Ensembl models, cross-check them, return the map
    plus a provenance record.

    The Mutalyzer model is fetched for a RefSeq accession: the assay's own
    accession when it is an NM_*, otherwise the MANE Select RefSeq partner of
    the assay's Ensembl transcript (MANE Select guarantees identical exon
    structure and c. numbering between the RefSeq and Ensembl pair; the
    exon-length cross-check below enforces it for the exact versions in use).
    Mutalyzer's model of an ENST accession is a genomic gene-region record
    whose coordinates are not spliced transcript coordinates, so it cannot be
    used for c. arithmetic.
    """
    if assay_accession.startswith(("NM_", "NR_")):
        model_accession = assay_accession
        ensembl_id = mane.ensembl_nuc.split(".")[0]
    else:
        model_accession = mane.refseq_nuc
        ensembl_id = assay_accession.split(".")[0]
    model = fetch_mutalyzer_model(model_accession)
    ann = model["annotations"]

    # Pick the mRNA feature matching the model accession (RefSeq records
    # carry a single transcript nested under a gene feature).
    wanted = model_accession.split(".")[0]
    candidates: list[dict] = []

    def _collect(feats: list[dict]) -> None:
        for f in feats:
            if f.get("type") == "mRNA":
                candidates.append(f)
            _collect(f.get("features") or [])

    _collect(ann.get("features") or [])
    matching = [f for f in candidates if str(f.get("id", "")).split(".")[0] == wanted]
    if len(matching) != 1:
        raise RuntimeError(
            f"{model_accession}: expected 1 matching mRNA in Mutalyzer model, "
            f"found {len(matching)} of {len(candidates)}"
        )
    mrna = matching[0]
    cds_feat = next(f for f in mrna["features"] if f["type"] == "CDS")
    mut_exons = sorted(
        (f for f in mrna["features"] if f["type"] == "exon"),
        key=lambda f: f["location"]["start"]["position"],
    )
    cds_start = cds_feat["location"]["start"]["position"]
    cds_end = cds_feat["location"]["end"]["position"]

    # Genomic anchor: the assay's own Ensembl transcript when it is one,
    # otherwise the MANE Select Ensembl partner of the assay RefSeq.
    lookup = fetch_ensembl_lookup(ensembl_id)
    ens_exons_raw = lookup.get("Exon") or []
    ens_strand = int(lookup["strand"])
    strand_sign = 1 if mane.strand == "+" else -1
    if ens_strand != strand_sign:
        raise RuntimeError(
            f"{assay_accession}: strand mismatch MANE ({mane.strand}) vs Ensembl ({ens_strand})"
        )
    ens_exons = sorted(ens_exons_raw, key=lambda e: e["start"], reverse=strand_sign == -1)

    if len(mut_exons) != len(ens_exons):
        raise RuntimeError(
            f"{assay_accession}: exon count mismatch "
            f"Mutalyzer={len(mut_exons)} Ensembl={len(ens_exons)}"
        )

    # Length check per exon (in transcript order). Internal exons must match
    # exactly. A terminal (first/last) exon may differ when the assay's
    # transcript version revises only the UTR end (e.g. NM_007294.3 vs .4):
    # the shared splice boundary is kept fixed and the free UTR end is
    # extended/shortened to the Mutalyzer length. Anything else aborts.
    exons: list[Exon] = []
    adjustments: list[str] = []
    last = len(mut_exons) - 1
    for i, (me, ee) in enumerate(zip(mut_exons, ens_exons)):
        t_s = me["location"]["start"]["position"]
        t_e = me["location"]["end"]["position"]
        m_len = t_e - t_s
        e_len = ee["end"] - ee["start"] + 1
        g_start, g_end = ee["start"], ee["end"]
        if m_len != e_len:
            if i == 0:  # 5'-terminal exon: 3' (donor) boundary is shared
                if strand_sign == 1:
                    g_start = g_end - m_len + 1
                else:
                    g_end = g_start + m_len - 1
            elif i == last:  # 3'-terminal exon: 5' (acceptor) boundary is shared
                if strand_sign == 1:
                    g_end = g_start + m_len - 1
                else:
                    g_start = g_end - m_len + 1
            else:
                raise RuntimeError(
                    f"{assay_accession}: internal exon {i + 1} length mismatch "
                    f"Mutalyzer={m_len} Ensembl={e_len} (accession version drift?)"
                )
            adjustments.append(
                f"terminal exon {i + 1}: Mutalyzer length {m_len} != Ensembl {e_len}; "
                f"shared splice boundary kept, free UTR end adjusted"
            )
        exons.append(Exon(t_start=t_s, t_end=t_e, g_start=g_start, g_end=g_end))
    tmap = TranscriptMap(exons, cds_start, cds_end, strand_sign, mane.chr_name)
    provenance = {
        "assay_accession": assay_accession,
        "mutalyzer_model_accession": model_accession,
        "mutalyzer_model": f"{MUTALYZER_API}/reference_model/?reference_id={model_accession}",
        "ensembl_anchor": ensembl_id,
        "ensembl_lookup": f"{ENSEMBL_REST}/lookup/id/{ensembl_id}?expand=1",
        "n_exons": len(exons),
        "cds_t_offsets": [cds_start, cds_end],
        "exon_anchor_adjustments": adjustments,
    }
    return tmap, provenance


# --------------------------------------------------------------------------
# HGVS c. parsing
# --------------------------------------------------------------------------


class MappingFailure(Exception):
    def __init__(self, category: str, detail: str):
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


_POS_RE = re.compile(r"^(?P<star>\*)?(?P<five>-)?(?P<num>\d+)(?P<off>[+-]\d+)?$")


@dataclass
class CPosition:
    star: bool  # 3' UTR (*N)
    five_prime: bool  # 5' UTR (-N)
    num: int
    offset: int  # intronic offset (+/-K), 0 if exonic

    def describe(self) -> str:
        s = "*" if self.star else ""
        s += "-" if self.five_prime else ""
        s += str(self.num)
        if self.offset:
            s += f"{self.offset:+d}"
        return s


def parse_c_position(text: str) -> CPosition:
    m = _POS_RE.match(text)
    if not m:
        raise MappingFailure("parse_error", f"bad c. position {text!r}")
    return CPosition(
        star=bool(m.group("star")),
        five_prime=bool(m.group("five")),
        num=int(m.group("num")),
        offset=int(m.group("off") or 0),
    )


@dataclass
class ParsedVariant:
    kind: str  # 'snv' | 'del' | 'dup' | 'ins' | 'delins'
    start: CPosition
    end: CPosition | None
    ref: str = ""  # stated deleted/ref sequence (transcript orientation)
    alt: str = ""  # inserted/alt sequence (transcript orientation)


def parse_c_hgvs(hgvs_c: str) -> ParsedVariant:
    if not hgvs_c.startswith("c."):
        raise MappingFailure("parse_error", f"not a c. description: {hgvs_c!r}")
    body = hgvs_c[2:]
    for op in ("delins", "del", "dup", "ins"):
        if op in body:
            pos_part, seq = body.split(op, 1)
            kind = "delins" if op == "delins" else op
            break
    else:
        m = re.match(r"^(?P<pos>[^>]+)(?P<ref>[ACGT])>(?P<alt>[ACGT])$", body)
        if not m:
            raise MappingFailure("unsupported_hgvs", f"cannot parse {hgvs_c!r}")
        return ParsedVariant(
            kind="snv",
            start=parse_c_position(m.group("pos")),
            end=None,
            ref=m.group("ref"),
            alt=m.group("alt"),
        )
    if "_" in pos_part:
        a, b = pos_part.split("_", 1)
        start, end = parse_c_position(a), parse_c_position(b)
    else:
        start, end = parse_c_position(pos_part), None
    if kind == "ins" and end is None:
        raise MappingFailure("parse_error", f"ins without position range: {hgvs_c!r}")
    if kind in ("del", "dup"):
        return ParsedVariant(kind=kind, start=start, end=end, ref=seq.upper())
    return ParsedVariant(kind=kind, start=start, end=end, alt=seq.upper())


# --------------------------------------------------------------------------
# Genomic mapping of one parsed variant
# --------------------------------------------------------------------------


class RegionSeq:
    def __init__(self, start: int, seq: str):
        self.start = start  # 1-based genomic position of seq[0]
        self.seq = seq

    def base(self, g: int) -> str:
        i = g - self.start
        if not 0 <= i < len(self.seq):
            raise MappingFailure("region_out_of_bounds", f"position {g} outside fetched slice")
        return self.seq[i]

    def span(self, a: int, b: int) -> str:
        """1-based inclusive [a, b], a <= b."""
        if a < self.start or b >= self.start + len(self.seq):
            raise MappingFailure("region_out_of_bounds", f"{a}..{b} outside fetched slice")
        return self.seq[a - self.start : b - self.start + 1]


def _genomic_oriented(seq_transcript: str, strand_sign: int) -> str:
    return seq_transcript if strand_sign == 1 else revcomp(seq_transcript)


def map_variant(tmap: TranscriptMap, rseq: RegionSeq, hgvs_c: str) -> tuple[int, str, str]:
    """Map one c. HGVS to (pos, ref, alt) on GRCh38 + strand (VCF-style)."""
    pv = parse_c_hgvs(hgvs_c)
    s = tmap.strand_sign

    if pv.kind == "snv":
        g = tmap.g_position(pv.start)
        expected = _genomic_oriented(pv.ref, s)
        actual = rseq.base(g)
        if actual != expected:
            raise MappingFailure(
                "ref_mismatch",
                f"{hgvs_c}: GRCh38 has {actual} at {tmap.chrom}:{g}, HGVS expects {expected}",
            )
        return g, actual, _genomic_oriented(pv.alt, s)

    g1 = tmap.g_position(pv.start)
    g2 = tmap.g_position(pv.end) if pv.end is not None else g1
    g_lo, g_hi = min(g1, g2), max(g1, g2)

    if pv.kind == "del":
        deleted = rseq.span(g_lo, g_hi)
        if pv.ref and deleted != _genomic_oriented(pv.ref, s):
            raise MappingFailure("ref_mismatch", f"{hgvs_c}: deleted seq mismatch")
        anchor = rseq.base(g_lo - 1)
        return g_lo - 1, anchor + deleted, anchor

    if pv.kind == "delins":
        deleted = rseq.span(g_lo, g_hi)
        anchor = rseq.base(g_lo - 1)
        inserted = _genomic_oriented(pv.alt, s)
        return g_lo - 1, anchor + deleted, anchor + inserted

    if pv.kind == "dup":
        dup_seq = rseq.span(g_lo, g_hi)
        if pv.ref and dup_seq != _genomic_oriented(pv.ref, s):
            raise MappingFailure("ref_mismatch", f"{hgvs_c}: duplicated seq mismatch")
        if s == 1:
            anchor_g = g_hi  # copy inserted 3' of the segment (transcript sense)
        else:
            anchor_g = g_lo - 1  # on - strand, transcript 3' = lower genomic coord
        anchor = rseq.base(anchor_g)
        return anchor_g, anchor, anchor + dup_seq

    if pv.kind == "ins":
        if g_hi - g_lo != 1:
            raise MappingFailure(
                "unsupported_hgvs", f"{hgvs_c}: ins positions not adjacent on genome"
            )
        inserted = _genomic_oriented(pv.alt, s)
        anchor = rseq.base(g_lo)
        return g_lo, anchor, anchor + inserted

    raise MappingFailure("unsupported_hgvs", hgvs_c)


# --------------------------------------------------------------------------
# Live cross-validation: Ensembl VEP (independent mapper) + dbSNP / ClinVar
# --------------------------------------------------------------------------


def vcf_to_minimal(pos: int, ref: str, alt: str) -> tuple[int, str, str]:
    """Minimal (0-based position, deleted, inserted) of our left-anchored
    VCF-style alleles."""
    if len(ref) == 1 and len(alt) == 1:
        return pos - 1, ref, alt
    return pos, ref[1:], alt[1:]


def ensembl_vep_hgvs(accession: str, hgvs_c: str) -> dict | None:
    """Map a c. HGVS with Ensembl VEP (independent of our Mutalyzer-based
    mapping). Returns the first VEP entry, or None on failure."""
    url = f"{ENSEMBL_REST}/vep/human/hgvs/{quote(f'{accession}:{hgvs_c}', safe='')}?content-type=application/json"
    try:
        data = http_get(url).json()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, list) or not data or "start" not in data[0]:
        return None
    return data[0]


def vep_to_minimal(entry: dict, strand_sign: int) -> tuple[int, str, str] | None:
    """Convert a VEP hgvs entry to minimal (0-based position, deleted,
    inserted) on the genomic + strand, or None if unparseable.

    VEP echoes alleles in the orientation of the input (transcript) HGVS, so
    for minus-strand genes they must be reverse-complemented.
    """
    try:
        start, end = int(entry["start"]), int(entry["end"])
        ref, alt = entry["allele_string"].split("/")[:2]
    except (KeyError, ValueError):
        return None
    ref = "" if ref == "-" else ref
    alt = "" if alt == "-" else alt
    if strand_sign == -1:
        ref, alt = revcomp(ref), revcomp(alt)
    if start <= end:  # substitution / deletion / delins over [start, end]
        return start - 1, ref, alt
    return end, ref, alt  # insertion between end and start (start = end + 1)


def validate_sample(
    df: pd.DataFrame,
    gene: str,
    mane: ManeRecord,
    assay_accession: str,
    n: int,
    seed: int,
) -> list[dict]:
    """Cross-check a seeded random sample of mapped variants against live
    Ensembl VEP mapping, with dbSNP (rsID) and ClinVar (clin_sig) evidence
    from VEP colocated variants."""
    ok = df[df["mapping_status"] == "ok"]
    snvs = ok[ok["ref"].str.len() == 1]
    indels = ok[ok["ref"].str.len() > 1]
    rng = random.Random(seed)
    sample_idx = []
    n_snv = min(len(snvs), max(1, int(round(n * 0.7))))
    n_indel = min(len(indels), n - n_snv)
    sample_idx += rng.sample(list(snvs.index), n_snv) if n_snv else []
    sample_idx += rng.sample(list(indels.index), n_indel) if n_indel else []
    results = []
    for idx in sample_idx:
        row = df.loc[idx]
        rec = {
            "variant_id": row["variant_id"],
            "hgvs_c": row["hgvs_c"],
            "ours": {"chrom": row["chrom"], "pos": int(row["pos"]), "ref": row["ref"], "alt": row["alt"]},
        }
        entry = ensembl_vep_hgvs(assay_accession, row["hgvs_c"])
        query_acc = assay_accession
        if entry is None and assay_accession != mane.refseq_nuc:
            # VEP only knows current RefSeq versions; an older assay
            # accession (e.g. NM_007294.3) falls back to the MANE Select
            # version (c. numbering is CDS-anchored, hence identical).
            entry = ensembl_vep_hgvs(mane.refseq_nuc, row["hgvs_c"])
            query_acc = mane.refseq_nuc
        time.sleep(0.2)  # be polite to Ensembl REST
        strand_sign = 1 if mane.strand == "+" else -1
        vep_min = vep_to_minimal(entry, strand_sign) if entry else None
        if vep_min is None:
            rec["coordinate_match"] = None
            rec["vep"] = "no GRCh38 mapping returned by VEP"
        else:
            rec["vep_query_accession"] = query_acc
            ours_min = vcf_to_minimal(int(row["pos"]), row["ref"], row["alt"])
            rec["vep_minimal"] = {
                "seq_region_name": entry.get("seq_region_name"),
                "position0": vep_min[0],
                "deleted": vep_min[1],
                "inserted": vep_min[2],
            }
            rec["coordinate_match"] = bool(
                entry.get("seq_region_name") == row["chrom"]
                and vep_min[0] == ours_min[0]
                and vep_min[1] == ours_min[1]
                and vep_min[2] == ours_min[2]
            )
            colocated = entry.get("colocated_variants") or []
            rec["dbsnp_rsids"] = sorted({c["id"] for c in colocated if c.get("id", "").startswith("rs")})
            clin = sorted({
                sig
                for c in colocated
                for sig in (c.get("clin_sig") or [])
            })
            rec["clinvar_significance"] = clin
        results.append(rec)
    return results


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def load_registry(config_path: str | Path) -> dict[str, dict]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return {a["urn"]: a for a in cfg.get("assays", []) if a.get("urn")}


def process_dataset(
    urn_dir: Path,
    urn: str,
    gene: str,
    mane: ManeRecord,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    meta = json.loads((urn_dir / "metadata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(urn_dir / "scores.csv", dtype=str, keep_default_na=False)
    df = df[["accession", "hgvs_nt", "score"]].rename(columns={"accession": "variant_id"})

    parts = df["hgvs_nt"].str.split(":", n=1, expand=True)
    assay_accessions = parts[0].unique()
    if len(assay_accessions) != 1:
        raise RuntimeError(f"{urn}: multiple assay accessions in hgvs_nt: {assay_accessions}")
    assay_acc = assay_accessions[0]
    df["hgvs_c"] = parts[1]

    tmap, prov = build_transcript_map(assay_acc, mane)
    g_lo = min(e.g_start for e in tmap.exons) - REGION_PAD
    g_hi = max(e.g_end for e in tmap.exons) + REGION_PAD
    r_start, _, seq = fetch_ensembl_region(mane.chr_name, g_lo, g_hi)
    rseq = RegionSeq(r_start, seq)

    records = []
    failure_counts: dict[str, int] = {}
    failure_examples: dict[str, str] = {}
    for _, row in df.iterrows():
        status = "ok"
        chrom, pos, ref, alt = mane.chr_name, None, None, None
        try:
            pos, ref, alt = map_variant(tmap, rseq, row["hgvs_c"])
        except MappingFailure as exc:
            status = exc.category
            failure_counts[status] = failure_counts.get(status, 0) + 1
            failure_examples.setdefault(status, f"{row['hgvs_c']}: {exc.detail}")
        records.append(
            {
                "variant_id": row["variant_id"],
                "gene": gene,
                "urn": urn,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "transcript": mane.refseq_nuc,
                "hgvs_c": row["hgvs_c"],
                "functional_pathogenicity": -float(row["score"]),
                "mapping_status": status,
            }
        )
    out = pd.DataFrame(records, columns=COLUMNS)
    n_ok = int((out["mapping_status"] == "ok").sum())

    validation = validate_sample(out, gene, mane, assay_acc, sample_size, seed) if n_ok else []

    report = {
        "urn": urn,
        "gene": gene,
        "title": meta.get("title", ""),
        "rows": len(out),
        "mapped_ok": n_ok,
        "success_rate": n_ok / len(out) if len(out) else 0.0,
        "failure_counts": failure_counts,
        "failure_examples": failure_examples,
        "mane_select_refseq": mane.refseq_nuc,
        "mane_select_ensembl": mane.ensembl_nuc,
        "grch38_chr_accession": mane.grch38_chr_accession,
        "strand": mane.strand,
        "provenance": prov,
        "validation": validation,
    }
    return out, report


def render_report(reports: list[dict], summary: dict) -> str:
    lines = [
        "# Mapping report — frozen matrix v1",
        "",
        f"Generated: {summary['generated_at']} by `python -m atlas.mapping`",
        "",
        "## Provenance (all fetched live by script, hash-registered under `data/raw/reference/`)",
        "",
        f"- MANE Select transcripts: {MANE_SUMMARY_URL}",
        f"- Transcript exon models: Mutalyzer3 `{MUTALYZER_API}/reference_model/` (per assay accession)",
        f"- Genomic exon anchors + reference sequence: Ensembl REST `{ENSEMBL_REST}` (GRCh38)",
        f"- Cross-validation: Ensembl VEP `{ENSEMBL_REST}/vep/human/hgvs` "
        "(independent mapper; dbSNP/ClinVar via colocated variants)",
        "",
        f"Orientation rule: `{ORIENTATION_RULE}`",
        "",
        "Coordinate conventions: GRCh38; `pos` 1-based; indels left-anchored VCF-style",
        "(ref = anchor + deleted, alt = anchor + inserted); alleles on the + strand.",
        "",
        "## Per-dataset results",
        "",
        "| Gene | URN | Rows | Mapped | Success rate | Failure categories |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        fc = ", ".join(f"{k}: {v}" for k, v in sorted(r["failure_counts"].items())) or "—"
        lines.append(
            f"| {r['gene']} | {r['urn']} | {r['rows']} | {r['mapped_ok']} | "
            f"{r['success_rate']:.2%} | {fc} |"
        )
    lines += [
        "",
        f"**Total: {summary['total_rows']} rows, {summary['total_mapped']} mapped "
        f"({summary['total_mapped'] / summary['total_rows']:.2%}).**",
        "",
        "## MANE Select transcripts (live from NCBI MANE summary)",
        "",
        "| Gene | Assay accession | MANE Select RefSeq | MANE Select Ensembl | GRCh38 accession | Strand |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r['gene']} | {r['provenance']['assay_accession']} | {r['mane_select_refseq']} | "
            f"{r['mane_select_ensembl']} | {r['grch38_chr_accession']} | {r['strand']} |"
        )
    lines += ["", "## Live cross-validation (seeded random samples)", ""]
    for r in reports:
        val = r["validation"]
        checked = [v for v in val if v.get("coordinate_match") is not None]
        n_match = sum(1 for v in checked if v["coordinate_match"])
        n_rs = sum(1 for v in val if v.get("dbsnp_rsids"))
        n_cv = sum(1 for v in val if v.get("clinvar_significance"))
        lines.append(f"### {r['gene']} — {r['urn']}")
        lines.append("")
        lines.append(
            f"- Sampled {len(val)} mapped variants; Ensembl VEP returned a GRCh38 "
            f"mapping for {len(checked)}; coordinate agreement {n_match}/{len(checked)}."
        )
        lines.append(
            f"- dbSNP: {n_rs}/{len(val)} sample variants colocate with an rsID; "
            f"ClinVar: {n_cv}/{len(val)} colocate with a ClinVar record."
        )
        mismatches = [v for v in checked if not v["coordinate_match"]]
        if mismatches:
            lines.append("- MISMATCHES (need review):")
            for v in mismatches:
                lines.append(f"  - {v['variant_id']} {v['hgvs_c']}: ours={v['ours']} vep={v['vep_minimal']}")
        if r["failure_examples"]:
            lines.append("- Failure examples:")
            for cat, ex in sorted(r["failure_examples"].items()):
                lines.append(f"  - `{cat}`: {ex}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/assays.yaml")
    ap.add_argument("--raw", default=str(RAW_MAVEDB))
    ap.add_argument("--sample-size", type=int, default=10, help="validation sample per dataset")
    ap.add_argument("--seed", type=int, default=20260724)
    args = ap.parse_args(argv)

    registry = load_registry(args.config)
    raw_dir = Path(args.raw)
    urn_dirs = {p.name: p for p in raw_dir.iterdir() if p.is_dir()}

    jobs = []
    for urn, assay in registry.items():
        d = urn_dirs.get(urn.replace(":", "_"))
        if d is None:
            print(f"SKIP {urn}: not fetched under {raw_dir}", file=sys.stderr)
            continue
        jobs.append((d, urn, assay["gene"]))
    if not jobs:
        print("No datasets to process.", file=sys.stderr)
        return 1

    mane_records = load_mane_records([g for _, _, g in jobs])

    frames, reports = [], []
    for d, urn, gene in jobs:
        print(f"Mapping {urn} ({gene}) ...")
        df, report = process_dataset(
            d, urn, gene, mane_records[gene], args.sample_size, args.seed
        )
        frames.append(df)
        reports.append(report)
        print(
            f"  {report['mapped_ok']}/{report['rows']} mapped "
            f"({report['success_rate']:.2%}); failures: {report['failure_counts'] or '{}'}"
        )

    matrix = pd.concat(frames, ignore_index=True)[COLUMNS]
    if matrix["variant_id"].duplicated().any():
        dupes = matrix[matrix["variant_id"].duplicated()]["variant_id"].head().tolist()
        raise RuntimeError(f"duplicate variant_id values: {dupes}")

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = FROZEN_DIR / f"frozen-matrix-{FROZEN_VERSION}.parquet"
    matrix.to_parquet(parquet_path, index=False)

    summary = {
        "version": FROZEN_VERSION,
        "generated_at": utcnow(),
        "orientation_rule": ORIENTATION_RULE,
        "sources": {
            "mane_summary": MANE_SUMMARY_URL,
            "mutalyzer": f"{MUTALYZER_API}/reference_model/",
            "ensembl_rest": ENSEMBL_REST,
            "cross_validation": f"{ENSEMBL_REST}/vep/human/hgvs",
        },
        "total_rows": int(len(matrix)),
        "total_mapped": int((matrix["mapping_status"] == "ok").sum()),
        "datasets": [
            {
                "urn": r["urn"],
                "gene": r["gene"],
                "rows": r["rows"],
                "mapped_ok": r["mapped_ok"],
                "success_rate": r["success_rate"],
                "failure_counts": r["failure_counts"],
            }
            for r in reports
        ],
    }
    register_file(
        FROZEN_DIR / "MANIFEST.json",
        parquet_path,
        version=FROZEN_VERSION,
        rows=summary["total_rows"],
        orientation_rule=ORIENTATION_RULE,
        sources=summary["sources"],
    )
    prov_path = FROZEN_DIR / f"frozen-matrix-{FROZEN_VERSION}.provenance.json"
    prov_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    register_file(FROZEN_DIR / "MANIFEST.json", prov_path, version=FROZEN_VERSION)

    report_path = RESULTS_DIR / f"mapping_report_{FROZEN_VERSION}.md"
    report_path.write_text(render_report(reports, summary), encoding="utf-8")
    summary_path = RESULTS_DIR / f"mapping_summary_{FROZEN_VERSION}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nFrozen matrix: {parquet_path} ({summary['total_rows']} rows)")
    print(f"Report: {report_path}")
    print("\nPer-dataset mapping success rates:")
    for r in reports:
        print(f"  {r['gene']:<7} {r['urn']:<28} {r['success_rate']:>7.2%}  ({r['mapped_ok']}/{r['rows']})")
    print(f"  {'TOTAL':<7} {'':<28} {summary['total_mapped'] / summary['total_rows']:>7.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
