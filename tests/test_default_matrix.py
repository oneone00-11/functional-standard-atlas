"""Every analysis entry point must default to the matrix the paper is built from.

Nine modules take a ``--matrix`` argument. Six of them -- evaluate_ext,
robustness, metapredictors, ensemble, clinical_evidence and rna_readout --
defaulted to ``score_matrix_atlas_v1.parquet``, which carries twelve predictor
columns and no dbNSFP meta-predictors, while every delivered table is built
from ``score_matrix_atlas_v2.parquet``, which carries nineteen. Running a
module exactly as the README documents therefore could not reproduce the
delivered table: the eval_ext grid came out with 193 rows instead of 304 and
splice-stratum correlations shifted, and nothing said why. reliability.py was
fixed first (commit ed10148, pinned by test_reliability_matrix.py); this file
pins the remaining entry points.

v1 is kept on disk deliberately -- it is the auditable pre-rescore matrix that
atlas.matrix_v2 diffs against, and the input the full-precision re-scorers
read -- but no analysis entry point may default to it.
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results"

# Every module whose main() takes --matrix. assemble.py and matrix_v2.py are
# absent on purpose: they *build* v1 and v2 respectively, so v1 is the correct
# input there, as it is for the full-precision re-scorers under models/.
ENTRYPOINTS = [
    "clinical_evidence",
    "definition_sweep",
    "ensemble",
    "evaluate_ext",
    "metapredictors",
    "reliability",
    "rna_readout",
    "robustness",
    "simulate_attenuation",
]


@pytest.mark.parametrize("module", ENTRYPOINTS)
def test_the_matrix_default_is_v2(module):
    """v1 has twelve columns and v2 nineteen. Defaulting any entry point to v1
    is what made the delivered tables unreproducible as documented."""
    mod = importlib.import_module(f"atlas.{module}")
    src = inspect.getsource(mod.main)
    assert 'default=str(RESULTS / "score_matrix_atlas_v2.parquet")' in src, module
    assert 'default=str(RESULTS / "score_matrix_atlas_v1.parquet")' not in src, module


def test_evaluate_ext_with_default_arguments_reproduces_the_delivered_grid(tmp_path):
    """A default run must regenerate the delivered 19 x 16 grid: 304 rows,
    cell for cell. With the old v1 default it produced 193."""
    pytest.importorskip("pyarrow")
    import pandas as pd

    delivered = RESULTS / "eval_ext_v1.tsv"
    if not ((RESULTS / "score_matrix_atlas_v2.parquet").exists()
            and delivered.exists()):
        pytest.skip("results/ not built in this checkout")

    from atlas import evaluate_ext
    evaluate_ext.main(["--out", str(tmp_path)])

    got = tmp_path / "eval_ext_v1.tsv"
    assert len(pd.read_csv(got, sep="\t")) == 304
    assert got.read_bytes() == delivered.read_bytes(), (
        "a default evaluate_ext run no longer reproduces the delivered grid")


def test_robustness_with_default_arguments_reproduces_the_delivered_files(tmp_path):
    """Same guard for the robustness checks: ties, power, paired tests, LOGO
    and the summary json, all five byte for byte. The bootstrap is seeded, so
    a default run is deterministic."""
    delivered = ["tie_audit_v1.tsv", "power_v1.tsv", "head_to_head_v1.tsv",
                 "logo_v1.tsv", "robustness_v1.json"]
    if not ((RESULTS / "score_matrix_atlas_v2.parquet").exists()
            and all((RESULTS / f).exists() for f in delivered)):
        pytest.skip("results/ not built in this checkout")

    from atlas import robustness
    robustness.main(["--out", str(tmp_path)])

    for name in delivered:
        assert (tmp_path / name).read_bytes() == (RESULTS / name).read_bytes(), (
            f"a default robustness run no longer reproduces {name}")
