"""atlas.external_coding: the external deposits' coding SNVs, their consequence, and the
synonymous ones' coordinates and SpliceAI (A4b.2)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas import external_coding as ec
from atlas import paths
from atlas.consequence import classify_variant

TABLE = paths.path("external_coding_consequence")
PROV = paths.path("external_coding_consequence_provenance")
needs = pytest.mark.skipif(not PROV.exists(), reason="run `python -m atlas.external_coding "
                                                     "assemble` first")
models = pytest.mark.skipif(not all((ec.MUTALYZER / f"reference_model_{a}.json").exists()
                                    for a in ec.TRANSCRIPT.values()),
                            reason="run `python -m atlas.external_coding fetch` first")


@models
def test_every_transcript_starts_with_atg_and_ends_with_a_stop():
    for gene, acc in ec.TRANSCRIPT.items():
        tx = ec.mutalyzer_transcript(acc)
        cds = tx["seq"][tx["cds_start"]:tx["cds_end"]]
        assert cds[:3] == "ATG", gene
        assert cds[-3:] in ("TAA", "TAG", "TGA"), gene
        assert len(cds) % 3 == 0, gene


@models
def test_the_ensembl_record_is_spliced_on_the_transcripts_strand():
    # SBDS: Mutalyzer's model of ENST00000246868.7 is a genomic record of a minus-strand
    # gene; its exons, spliced on that strand, match the genome
    tmap, rseq, check, tx = ec.transcript_map("SBDS", ec.Fasta())
    assert check["strand"] == -1 and check["spliced_genome_equals_model"]
    assert check["cds_t_offsets"][1] - check["cds_t_offsets"][0] == 753


@models
def test_known_codons_classify_as_the_genetic_code_says():
    tx = ec.mutalyzer_transcript(ec.TRANSCRIPT["CTCF"])
    seq, cs, ce = tx["seq"], tx["cds_start"], tx["cds_end"]
    first = seq[cs:cs + 3]
    assert first == "ATG"
    assert classify_variant("c.1A>G", seq, cs, ce) == "start_lost"
    stop = seq[ce - 3:ce]
    alt = "C" if stop[0] != "C" else "G"
    pos = ce - cs - 2
    assert classify_variant(f"c.{pos}{stop[0]}>{alt}", seq, cs, ce) == "stop_lost"
    assert classify_variant(f"c.{pos}{'A' if stop[0] != 'A' else 'C'}>G", seq, cs, ce) \
        == "ref_mismatch"


@needs
def test_the_consequences_agree_with_the_deposits_own_codons():
    prov = json.loads(PROV.read_text())
    assert not prov["failures"]
    for gene, r in prov["genes"].items():
        exp = r["expected_from_instruction"]
        assert r["coding_snvs"] == exp["coding_snvs"]
        assert r["complete_codon_disagreements"] == 0 and r["ref_mismatch"] == 0
        for k in ("synonymous", "missense", "nonsense"):
            assert r["deposit_codons"].get(k, 0) == exp[k]
        assert r["intronic_snvs_mapped_to_the_table"] == r["intronic_snvs_in_external_scores"]
    assert len(prov["genes"]["CTCF"]["incomplete_codon"]) == 30
    assert sum(r["intronic_snvs_in_external_scores"] for r in prov["genes"].values()) == 5095


@needs
def test_the_spliceai_rescore_equals_the_table_bit_for_bit():
    prov = json.loads(PROV.read_text())
    chk = prov["rescore_check"]
    assert chk["n"] == chk["bitwise_equal"] == 100 and chk["max_abs_diff"] == 0


@needs
def test_every_synonymous_snv_has_coordinates_and_spliceai():
    t = pd.read_parquet(TABLE)
    syn = t["consequence"] == "synonymous"
    assert syn.sum() == 3311
    assert t.loc[syn, ["variant_id", "chrom", "pos", "ref", "alt", "spliceai_ds"]].notna().all().all()
    assert t.loc[~syn, ["variant_id", "spliceai_ds"]].isna().all().all()
    want = (t.loc[syn, "chrom"] + "-" + t.loc[syn, "pos"].astype(int).astype(str) + "-"
            + t.loc[syn, "ref"] + "-" + t.loc[syn, "alt"])
    assert (want == t.loc[syn, "variant_id"]).all()
    assert np.allclose(t["functional_pathogenicity"], -t["score"], equal_nan=True)


@needs
def test_the_records_write_the_home_directory_as_a_tilde():
    from pathlib import Path
    assert str(Path.home()) not in PROV.read_text()
