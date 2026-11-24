"""models/alphagenome_atlas/score.py and atlas.alphagenome_atlas: the key read in one
place and never written; the maximum over tracks and rows; the guards (an authentication
error, a batch out of attempts, the quota's doubling pause); a resumed run; the anchors'
bit-for-bit check; the input and the comparison with the release. No test touches the
network, the service or a real key: the client is the scorer's MockClient."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import alphagenome_atlas as aa  # noqa: E402

spec = importlib.util.spec_from_file_location("alphagenome_atlas_score",
                                              REPO / "models/alphagenome_atlas/score.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    key = tmp_path / ".config" / "alphagenome" / "key"
    key.parent.mkdir(parents=True)
    monkeypatch.setattr(sc, "KEY_FILE", key)
    monkeypatch.delenv(sc.KEY_ENV, raising=False)
    return key


def test_the_key_comes_from_the_environment_or_a_0600_file_and_is_never_recorded(home, monkeypatch):
    assert sc.read_key() is None
    home.write_text("k-123456789\n")
    home.chmod(0o644)
    with pytest.raises(SystemExit, match="600"):
        sc.read_key()
    home.chmod(0o600)
    assert sc.read_key() == "k-123456789"
    rep = sc.key_report()
    assert rep == {"api_key_present": True, "api_key_source": "~/.config/alphagenome/key"}
    assert "k-123456789" not in json.dumps(rep)
    assert sc.clean("denied for k-123456789") == "denied for <key>"
    monkeypatch.setenv(sc.KEY_ENV, "e-1")
    assert sc.read_key() == "e-1"


def test_each_column_is_the_maximum_over_tracks_and_rows():
    ad = sc._AnnData(np.array([[0.1, 0.7, 0.2], [0.9, 0.0, 0.3], [0.4, 0.4, 0.4]], dtype=np.float32),
                     ["chr1:5:A>G", "chr1:5:A>G", "chr1:6:C>T"])
    got = sc._aggregate(ad, ["chr1:6:C>T", "chr1:5:A>G", "chr1:7:G>A"])
    assert got[0] == pytest.approx(0.4) and got[1] == pytest.approx(0.9) and np.isnan(got[2])


def _todo(n):
    return pd.DataFrame({"variant_id": [f"v{i}" for i in range(n)], "gene": "G", "set": "core",
                         "chrom": "1", "pos": range(100, 100 + n), "ref": "A", "alt": "G"})


def test_a_run_scores_every_row_and_a_second_run_resumes(tmp_path):
    out = tmp_path / "out"
    client = sc.Guarded(sc.MockClient())
    st = sc.run(_todo(7), client, out, version="mock", batch=3, genome_variant=sc._Variant)
    got = pd.read_parquet(st["output"])
    assert st["complete"] and st["n_requests"] == 7 and st["calls"] == 3
    assert got[sc.COLUMNS].notna().all().all() and got["variant_id"].tolist() == [f"v{i}" for i in range(7)]
    again = sc.run(_todo(7), sc.Guarded(sc.MockClient()), out, version="mock", batch=3,
                   genome_variant=sc._Variant)
    assert again["n_requests"] == 0 and again["n_from_checkpoint"] == 7
    rec = json.loads((out / "scorers.json").read_text())
    assert "AVI_SCORE" in rec["scorers"] and rec["n_scorers"] == 5


class _Failing(sc.MockClient):
    def __init__(self, error, times):
        self.error, self.times, self.calls = error, times, 0

    def query_variants(self, *a, **k):
        self.calls += 1
        if self.calls <= self.times:
            raise RuntimeError(self.error)
        return super().query_variants(*a, **k)


def test_an_authentication_error_stops_at_once(tmp_path):
    with pytest.raises(sc.Stop, match="authentication"):
        sc.run(_todo(2), sc.Guarded(_Failing("UNAUTHENTICATED: bad", 1)), tmp_path,
               version="mock", genome_variant=sc._Variant, sleep=lambda s: None)


def test_quota_errors_wait_twenty_seconds_doubling_then_the_batch_goes_through(tmp_path):
    waits = []
    st = sc.run(_todo(2), sc.Guarded(_Failing("RESOURCE_EXHAUSTED: quota", 2)), tmp_path,
                version="mock", genome_variant=sc._Variant, sleep=waits.append)
    assert waits == [20.0, 40.0] and st["complete"] and st["retries"] == 2


def test_a_batch_out_of_attempts_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    with pytest.raises(sc.Stop, match="all 6 attempts"):
        sc.run(_todo(2), sc.Guarded(_Failing("UNAVAILABLE: down", 99)), tmp_path,
               version="mock", genome_variant=sc._Variant, sleep=lambda s: None)


def test_the_anchor_check_is_to_the_bit(tmp_path):
    ref = pd.DataFrame({"variant_id": ["a", "b"], **{c: [0.5, np.nan] for c in sc.COLUMNS}})
    p = tmp_path / "anchors.parquet"
    ref.to_parquet(p)
    assert sc.anchor_check(ref, p)["holds"]
    got = ref.copy()
    got.loc[0, "avi"] = 0.5 + 2 ** -45
    out = sc.anchor_check(got, p)
    assert not out["holds"] and out["avi"]["n_identical"] == 1


def test_the_input_holds_only_snvs_core_then_external():
    m = pd.DataFrame({"variant_id": ["a", "b", "c"], "gene": "G", "chrom": "1", "pos": [1, 2, 3],
                      "ref": ["A", "AT", "C"], "alt": ["G", "A", "T"]})
    x = pd.DataFrame({"variant_id": ["1-9-A-G"], "gene": "H", "chrom": "1", "pos": [9],
                      "ref": ["A"], "alt": ["G"]})
    t = aa.score_input(m, x)
    assert t["variant_id"].tolist() == ["a", "c", "1-9-A-G"]
    assert t["set"].tolist() == ["core", "core", "external"]


def test_the_comparison_with_the_release_is_row_by_row():
    t = pd.DataFrame({"variant_id": ["a", "x"], "gene": "G", "set": ["core", "external"],
                      **{c: [0.25, 0.5] for c in aa.COLUMNS}})
    avi = pd.DataFrame({"variant_id": ["a", "z"], "set": ["analysis", "ddx3x"],
                        **{c: [0.25, 9.0] for c in aa.COLUMNS}})
    new = pd.DataFrame({"variant_id": ["x"], **{c: [0.5] for c in aa.COLUMNS}})
    assert aa.concordance(t, avi, new)["holds"].all()
    new.loc[0, "avi_splice_junctions"] = 0.5000001
    c = aa.concordance(t, avi, new)
    assert c.loc[~c["holds"], "column"].tolist() == ["avi_splice_junctions"]


def test_the_environment_record_names_no_home_directory(tmp_path):
    env = {"interpreter": f"{tmp_path}/work/envs/x/bin/python", "python": "3.12.13",
           "pip_freeze": ["a==1", f"b @ file://{tmp_path}/b"], "api_key_present": True,
           "elsewhere": "/opt/x"}
    got = aa.home_as_tilde(env, home=tmp_path)
    assert got["interpreter"] == "~/work/envs/x/bin/python"
    assert got["pip_freeze"] == ["a==1", f"b @ file://{tmp_path}/b"]   # not at the start
    assert got["elsewhere"] == "/opt/x" and got["api_key_present"] is True
    assert str(tmp_path) not in json.dumps({k: v for k, v in got.items() if k != "pip_freeze"})
