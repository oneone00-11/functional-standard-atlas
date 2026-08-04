"""Protein consequence class for every atlas variant, derived locally.

None of the seven MaveDB deposits populates ``hgvs_pro`` (all NA), so the
consequence of a coding variant has to be re-derived. It is derived here from
the same Mutalyzer transcript models the atlas already used for genomic mapping
(cached under ``data/raw/reference/mutalyzer/``): the model carries the spliced
transcript sequence and the CDS offsets, which is everything needed to place a
``c.`` position in its codon and translate.

Deriving it locally rather than calling an annotation service keeps the
consequence column reproducible offline and consistent by construction with the
``c.`` numbering the region classifier already uses.

Every coding SNV is checked against the transcript sequence before it is
classified: if the reference base implied by the HGVS string does not match the
transcript, the variant is marked ``ref_mismatch`` rather than guessed at.

Usage (PYTHONPATH=src):  python -m atlas.consequence --out results/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MUTALYZER = REPO / "data" / "raw" / "reference" / "mutalyzer"
FROZEN = REPO / "data" / "frozen" / "frozen-matrix-v1.parquet"

BASES = "TCAG"
AA = ("FFLLSSSSYY**CC*W" "LLLLPPPPHHQQRRRR"
      "IIIMTTTTNNKKSSRR" "VVVVAAAADDEEGGGG")
CODON_TABLE = {
    a + b + c: AA[i * 16 + j * 4 + k]
    for i, a in enumerate(BASES) for j, b in enumerate(BASES) for k, c in enumerate(BASES)
}

# c.123A>G  (plain coding SNV: no +/- offset, no leading - or *)
_CODING_SNV = re.compile(r"^c\.(\d+)([ACGT])>([ACGT])$")
_UTR5 = re.compile(r"^c\.-\d+")
_UTR3 = re.compile(r"^c\.\*")
_INTRONIC = re.compile(r"^c\.\*?-?\d+[+-]\d+")


def translate(codon: str) -> str:
    return CODON_TABLE.get(codon.upper(), "X")


def load_transcript(accession: str) -> tuple[str, int, int]:
    """(spliced transcript sequence, cds_start, cds_end) from the cached model.

    Falls back across accession versions: the assay's own accession and the MANE
    partner can differ by version (e.g. BRCA1 NM_007294.3 vs .4) while sharing
    the exon structure and c. numbering that MANE Select guarantees.
    """
    base = accession.split(".")[0]
    paths = sorted(MUTALYZER.glob(f"reference_model_{base}.*.json"))
    if not paths:
        raise FileNotFoundError(f"no cached Mutalyzer model for {accession}")
    model = json.loads(paths[0].read_text())
    seq = model["sequence"]["seq"]

    mrna = cds = None

    def _walk(feats):
        nonlocal mrna, cds
        for f in feats:
            if f.get("type") == "mRNA":
                mrna = f
            if f.get("type") == "CDS" and cds is None:
                cds = f
            _walk(f.get("features") or [])

    _walk(model["annotations"].get("features") or [])
    if cds is None:
        raise RuntimeError(f"{accession}: no CDS feature in the cached model")
    return seq, cds["location"]["start"]["position"], cds["location"]["end"]["position"]


def classify_variant(hgvs_c: str, seq: str, cds_start: int, cds_end: int) -> str:
    if not isinstance(hgvs_c, str):
        return "unparsed"
    if _INTRONIC.match(hgvs_c):
        return "intronic"
    if _UTR5.match(hgvs_c):
        return "utr5"
    if _UTR3.match(hgvs_c):
        return "utr3"
    if any(t in hgvs_c for t in ("del", "ins", "dup", "inv")):
        return "indel_or_complex"
    m = _CODING_SNV.match(hgvs_c)
    if not m:
        return "unparsed"

    pos, ref, alt = int(m.group(1)), m.group(2), m.group(3)
    t = cds_start + pos - 1
    if t >= cds_end or t >= len(seq):
        return "beyond_cds"
    if seq[t].upper() != ref:
        return "ref_mismatch"

    frame = (pos - 1) % 3
    c0 = cds_start + (pos - 1) - frame
    ref_codon = seq[c0:c0 + 3].upper()
    if len(ref_codon) < 3:
        return "beyond_cds"
    alt_codon = ref_codon[:frame] + alt + ref_codon[frame + 1:]
    a_ref, a_alt = translate(ref_codon), translate(alt_codon)

    if a_ref == a_alt:
        return "synonymous"
    if a_alt == "*":
        return "nonsense"
    if a_ref == "*":
        return "stop_lost"
    if pos <= 3:
        return "start_lost"
    return "missense"


def build(frozen: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out, prov = [], {"transcripts": {}, "counts": {}}
    for (gene, accession), g in frozen.groupby(["gene", "transcript"]):
        seq, cs, ce = load_transcript(accession)
        prov["transcripts"][gene] = {
            "requested": accession,
            "model": sorted(MUTALYZER.glob(
                f"reference_model_{accession.split('.')[0]}.*.json"))[0].name,
            "cds_start": cs, "cds_end": ce, "cds_len": ce - cs,
            "seq_len": len(seq),
        }
        cons = g["hgvs_c"].map(lambda h: classify_variant(h, seq, cs, ce))
        out.append(pd.DataFrame({"variant_id": g["variant_id"], "gene": gene,
                                 "consequence": cons.values}))
    res = pd.concat(out, ignore_index=True)
    prov["counts"] = res["consequence"].value_counts().to_dict()
    prov["counts_by_gene"] = (res.groupby("gene")["consequence"]
                              .value_counts().unstack(fill_value=0).to_dict("index"))
    return res, prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    frozen = pd.read_parquet(FROZEN)
    res, prov = build(frozen)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out / "consequence_v1.parquet", index=False)
    (out / "consequence_v1.provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

    print(f"consequence_v1.parquet: {len(res):,} variants")
    for k, v in sorted(prov["counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18} {v:>7,}")
    bad = res[res["consequence"].isin(["ref_mismatch", "unparsed", "beyond_cds"])]
    if len(bad):
        print(f"\nnot classified: {len(bad):,}")
        print(bad.groupby(["gene", "consequence"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
