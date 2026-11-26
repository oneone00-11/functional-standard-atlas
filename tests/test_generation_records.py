"""The records the current generation writes name no home directory (G15).

A provenance or summary JSON of this generation may name a file, an interpreter or an
environment, but under the home directory only as ``~``: the records ship with the
results, and an absolute home path carries the machine's account name. Files written
before this generation, and caches, are not checked here."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from atlas import paths  # noqa: E402

# the JSON records written by this generation (atlas.paths), and the MaveDB build record
CURRENT_JSON = ["score_matrix_provenance", "assay_aux_provenance",
                "external_scores_provenance", "reliability_summary",
                "external_reliability_summary", "matched_ceilings_summary",
                "robustness_summary", "attenuation_simulation_summary",
                "rna_readout_summary", "clinical_evidence_summary", "ensemble_summary",
                "mavedb_survey_reaudit_summary",
                # A4b
                "rank_ceiling_selftest_summary"]
HOME_PATH = re.compile(r"(/Users/|/home/|[A-Za-z]:\\\\Users\\\\)[^/\\\\\"\s]+")


def test_the_list_covers_every_json_name_of_this_generation():
    json_keys = {k for k, n in paths.NAMES.items() if n.endswith(".json")}
    earlier = {"alphagenome_official_provenance", "alphagenome_atlas_provenance",
               "clinvar_status_provenance"}
    assert json_keys - earlier == set(CURRENT_JSON)


@pytest.mark.parametrize("key", CURRENT_JSON)
def test_no_home_directory_in_the_records(key):
    p = paths.path(key)
    if not p.exists():
        pytest.skip(f"{p.name} not built in this checkout")
    text = p.read_text(encoding="utf-8")
    assert str(Path.home()) not in text, f"{p.name} names the home directory"
    hit = HOME_PATH.search(text)
    assert hit is None, f"{p.name} holds an absolute home path ({hit.group(1)}...)"
