"""Guards for models/alphagenome/score_official.py: the key is read in one place and
never written; the runner's stops; the bit-for-bit comparison; an offline run. No test
touches the network, the service or a real key."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import threading
from collections import namedtuple
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("alphagenome_score_official",
                                              REPO / "models/alphagenome/score_official.py")
ag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ag)

Row = namedtuple("Row", "variant_id chrom pos ref alt")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home of its own, so that the key file lies where read_key looks."""
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    key = tmp_path / ".config" / "alphagenome" / "key"
    key.parent.mkdir(parents=True)
    monkeypatch.setattr(ag, "KEY_FILE", key)
    monkeypatch.delenv(ag.KEY_ENV, raising=False)
    return key


def test_the_key_comes_from_the_environment_or_a_0600_file_only(home, monkeypatch):
    assert ag.read_key() is None
    home.write_text("from-the-file\n")
    home.chmod(0o600)
    assert ag.read_key() == "from-the-file"
    home.chmod(0o644)
    with pytest.raises(SystemExit, match="600"):
        ag.read_key()
    monkeypatch.setenv(ag.KEY_ENV, " from-the-environment ")
    assert ag.read_key() == "from-the-environment"     # the file is not opened


def test_records_say_where_the_key_is_never_what_it_is(home, monkeypatch):
    home.write_text("k-123456789")
    home.chmod(0o600)
    rep = ag.key_report()
    assert rep == {"api_key_present": True, "api_key_source": "~/.config/alphagenome/key"}
    assert "k-123456789" not in json.dumps(rep)
    assert ag.clean("refused: k-123456789 is wrong") == "refused: <key> is wrong"
    monkeypatch.setenv(ag.KEY_ENV, "e-987")
    assert ag.key_report()["api_key_source"] == ag.KEY_ENV


def _rows(n):
    return [Row(f"v{i}", "1", 100 + i, "A", "G") for i in range(n)]


def test_an_authentication_error_stops_the_run(home):
    def score(r):
        raise RuntimeError("UNAUTHENTICATED: request had invalid credentials")
    with pytest.raises(ag.Stopped, match="authentication"):
        ag.run(_rows(5), score, lambda r, o: None, workers=1)


def test_a_passing_error_is_tried_again_after_doubling_pauses(home):
    calls = {"n": 0}
    got = []

    def score(r):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("unavailable")
        return {"x": 1.0}
    state = ag.run(_rows(1), score, lambda r, o: got.append(r.variant_id), workers=1,
                   backoff=0.001)
    assert got == ["v0"] and state["retries"] == 2 and state["waits"] == [0.001, 0.002]
    assert state["failures"] == []


def test_three_variants_in_a_row_out_of_attempts_stop_the_run(home):
    def score(r):
        raise ValueError("does not clear")
    with pytest.raises(ag.Stopped, match="3 variants in a row") as e:
        ag.run(_rows(10), score, lambda r, o: None, workers=1, attempts=2, backoff=0.0)
    assert len(e.value.state["failures"]) == 3


def test_no_more_than_two_threads(home):
    with pytest.raises(ValueError):
        ag.run(_rows(1), lambda r: {}, lambda r, o: None, workers=3)


def _write(path, vals):
    cols = ag.COMPONENTS + [ag.COMPOSITE]
    pd.DataFrame([{"variant_id": k, **dict(zip(cols, v))} for k, v in vals.items()]).to_csv(
        path, sep="\t", index=False)


def test_the_comparison_is_to_the_last_character(tmp_path):
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    _write(a, {"v1": ["0.1", "0.2", "0.3", "0.36"], "v2": ["1.0", "0.0", "0.0", "1.0"]})
    _write(b, {"v1": ["0.1", "0.2", "0.3", "0.36"], "v2": ["1.0", "0.0", "0.0", "1.0000000000000002"]})
    assert ag.differing(a, b, ["v1", "v2"]) == ["v2"]


def test_a_negative_raw_score_is_found():
    t = pd.DataFrame({f"min_raw_{c}": [0.1, 0.0] for c in ag.COMPONENTS})
    assert ag.negative_raw(t) == []
    t["min_raw_SPLICE_JUNCTIONS"] = [0.1, -1e-9]
    assert ag.negative_raw(t) == ["SPLICE_JUNCTIONS"]


def test_an_offline_run_writes_the_scores_and_a_record_without_the_key(home, tmp_path):
    home.write_text("never-read-in-a-mock-run")
    home.chmod(0o600)
    inp = tmp_path / "in.parquet"
    pd.DataFrame({"variant_id": ["1-100-A-G", "1-101-C-T", "7-5-G-A"], "gene": ["X", "X", "Y"],
                  "chrom": ["1", "1", "7"], "pos": [100, 101, 5], "ref": ["A", "C", "G"],
                  "alt": ["G", "T", "A"]}).to_parquet(inp, index=False)
    out = tmp_path / "out"
    assert ag.main(["--external", "--mock", "--input", str(inp), "--out", str(out)]) == 0
    t = pd.read_csv(out / "external_scores.tsv", sep="\t")
    assert len(t) == 3 and set(ag.COMPONENTS + [ag.COMPOSITE]) <= set(t.columns)
    rec = (out / "external_scores.tsv.provenance.json").read_text()
    assert "never-read-in-a-mock-run" not in rec and json.loads(rec)["mock"] is True
    again = json.loads(rec)["n_scored_this_run"]
    ag.main(["--external", "--mock", "--input", str(inp), "--out", str(out)])   # resumes
    assert again == 3 and json.loads((out / "external_scores.tsv.provenance.json")
                                     .read_text())["n_scored_this_run"] == 0


def test_a_check_starts_from_an_empty_cache_and_the_end_needs_the_start(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "check_start.jsonl").write_text('{"variant_id": "v"}\n')
    with pytest.raises(SystemExit, match="afresh"):
        ag.main(["--check-start", "--mock", "--out", str(out)])
    with pytest.raises(SystemExit, match="check-start"):
        ag.main(["--check-end", "--mock", "--out", str(out)])
