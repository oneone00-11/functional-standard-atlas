"""Guards for the external-gene drivers under models/: SpliceAI's and Pangolin's
score_external.py and the reference builder. The parts tested here run without SpliceAI,
Pangolin or gffutils installed; nothing reads the network or data/refs."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sa = _load("models/spliceai/score_external.py", "spliceai_score_external")
pg = _load("models/pangolin/score_external.py", "pangolin_score_external")
refs = _load("models/pangolin/build_external_refs.py", "pangolin_build_external_refs")


@pytest.mark.parametrize("mod", [sa, pg])
def test_the_cache_keeps_every_value_to_the_last_bit(mod, tmp_path):
    cache = tmp_path / "c.tsv"
    values = {"a": 0.1 + 0.2, "b": float("nan"), "c": 1e-17, "d": 0.8170661926269531}
    for k, v in values.items():
        mod.append_cache(cache, k, v)
    back = mod.read_cache(cache)
    assert set(back) == set(values)
    for k, v in values.items():
        assert (math.isnan(v) and math.isnan(back[k][0])) or back[k][0] == v
    assert all(when.endswith("+00:00") for _, when in back.values())


def test_pangolin_workers_split_the_rows_without_overlap():
    df = pd.DataFrame({"variant_id": [f"v{i}" for i in range(13)]})
    parts = [set(pg.worker_rows(df, k, 6)["variant_id"]) for k in range(6)]
    assert set().union(*parts) == set(df["variant_id"])
    assert sum(len(p) for p in parts) == len(df)
    with pytest.raises(ValueError):
        pg.worker_rows(df, 6, 6)


def test_pangolin_refuses_more_than_one_thread(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    with pytest.raises(SystemExit, match="one thread"):
        pg.score(SimpleNamespace(worker=0, workers=1), pd.DataFrame())


@pytest.mark.parametrize("mod, extra", [(sa, ["--cache", "x", "--output", "y"]),
                                        (pg, ["--cache-dir", "x"])])
def test_a_changed_scorer_is_refused(mod, extra, tmp_path):
    with pytest.raises(SystemExit):
        mod.main(["--input", str(tmp_path / "none.parquet"), *extra,
                  "--expect-sha256", "0" * 64])


def test_the_release_hashes_come_from_its_run_logs(tmp_path, monkeypatch):
    sl, rl = tmp_path / "spliceai.json", tmp_path / "refs.json"
    sl.write_text(json.dumps({"inputs": [
        {"path": "phase1/data/evidence/refs/chr1.fa", "bytes": 10, "sha256": "a1"},
        {"path": "phase1/data/evidence/refs/chr7.fa", "bytes": 7, "sha256": "a7"},
        {"path": "x/grch38_subset.fa", "bytes": 1, "sha256": "s"}]}))
    rl.write_text(json.dumps({
        "inputs": [{"path": "x/Homo_sapiens.GRCh38.112.gtf.gz", "sha256": "gz"}],
        "outputs": [{"path": "c/chr1.gtf", "sha256": "g1"}, {"path": "c/chr7.gtf", "sha256": "g7"},
                    {"path": "c/chr1.gtf.db", "sha256": "d1"}]}))
    monkeypatch.setattr(refs, "SPLICEAI_LOG", sl)
    monkeypatch.setattr(refs, "REFS_LOG", rl)
    fa, gtf, gz = refs.release_hashes()
    assert fa == {"1": ("a1", 10), "7": ("a7", 7)} and gtf == {"1": "g1", "7": "g7"} and gz == "gz"


def _fasta(tmp_path):
    """A two-record FASTA and its samtools index."""
    recs = [(">2 dna:chromosome x", "ACGTAC\nGT\n"), (">16 dna:chromosome y", "GGTTAA\nCC\n")]
    data, fai, off = b"", [], 0
    for head, seq in recs:
        start = len(data) + len(head) + 1
        data += (head + "\n" + seq).encode()
        name = head[1:].split()[0]
        fai.append(f"{name}\t8\t{start}\t6\t7")
    p = tmp_path / "subset.fa"
    p.write_bytes(data)
    (tmp_path / "subset.fa.fai").write_text("\n".join(fai) + "\n")
    return p, recs


def test_a_chromosome_record_is_hashed_from_the_index(tmp_path, monkeypatch):
    p, recs = _fasta(tmp_path)
    monkeypatch.setattr(refs, "SUBSET", p)
    head, seq = recs[1]
    record = (head + "\n" + seq).encode()
    assert refs.subset_record("16", head, len(record)) == hashlib.sha256(record).hexdigest()
    with pytest.raises(SystemExit):                 # a header one byte off is caught
        refs.subset_record("16", head + "x", len(record))


def test_the_gtf_keeps_the_four_chromosomes_in_order(tmp_path, monkeypatch):
    rows = ["#!genome-build GRCh38\n", "1\tE\tgene\t1\t9\n", "2\tE\tgene\t1\t9\n",
            "7\tE\tgene\t1\t9\n", "16\tE\tgene\t1\t9\n", "1\tE\texon\t2\t3\n",
            "17\tE\tgene\t1\t9\n", "X\tE\tgene\t1\t9\n"]
    gz = tmp_path / "a.gtf.gz"
    with gzip.open(gz, "wt") as f:
        f.writelines(rows)
    monkeypatch.setattr(refs, "GTF_GZ", gz)
    monkeypatch.setattr(refs, "OUT_GTF", tmp_path / "out.gtf")
    res = refs.build_gtf()
    kept = [r for r in rows if r.split("\t", 1)[0] in {"1", "7", "16", "17"}]
    assert (tmp_path / "out.gtf").read_text() == "".join(kept)
    assert res["1"]["rows"] == 2 and res["17"]["rows"] == 1
    one = "".join(r for r in rows if r.startswith("1\t")).encode()
    assert res["1"]["sha256"] == hashlib.sha256(one).hexdigest()
