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
# The entry points rebuilt for the current generation take the matrix from atlas.paths
# (score_matrix_atlas_v3 since A4). The two not rebuilt -- definition_sweep, which is
# bound to v5's design, and metapredictors, which builds an input -- keep v2.
CURRENT_GENERATION = {"clinical_evidence", "ensemble", "evaluate_ext", "reliability",
                      "rna_readout", "robustness", "simulate_attenuation"}


@pytest.mark.parametrize("module", sorted(set(ENTRYPOINTS) - CURRENT_GENERATION))
def test_the_matrix_default_is_v2(module):
    """v1 has twelve columns and v2 nineteen. Defaulting any entry point to v1
    is what made the delivered tables unreproducible as documented."""
    mod = importlib.import_module(f"atlas.{module}")
    src = inspect.getsource(mod.main)
    assert 'default=str(RESULTS / "score_matrix_atlas_v2.parquet")' in src, module
    assert 'default=str(RESULTS / "score_matrix_atlas_v1.parquet")' not in src, module


@pytest.mark.parametrize("module", sorted(CURRENT_GENERATION))
def test_the_matrix_default_is_the_current_one_from_atlas_paths(module):
    """A rebuilt entry point names no matrix of its own: the default is the current
    one, written once in atlas.paths."""
    from atlas import paths
    mod = importlib.import_module(f"atlas.{module}")
    src = inspect.getsource(mod.main)
    assert 'default=str(paths.path("score_matrix"))' in src, module
    assert "score_matrix_atlas_v" not in src, module
    assert paths.name("score_matrix") == "score_matrix_atlas_v3.parquet"


def test_evaluate_ext_with_default_arguments_reproduces_the_delivered_grid(tmp_path):
    """A default run must regenerate the delivered grid of the current generation (23
    predictors x 16 snv strata, and the non_snv strata) and its per-gene table, byte for
    byte. With the old v1 default it produced 193 rows of v5's 304."""
    pytest.importorskip("pyarrow")
    import pandas as pd
    from atlas import paths

    delivered = [paths.path("eval_ext"), paths.path("eval_ext_per_gene")]
    if not (paths.path("score_matrix").exists() and all(p.exists() for p in delivered)):
        pytest.skip("results/ not built in this checkout")

    from atlas import evaluate_ext
    evaluate_ext.main(["--out", str(tmp_path)])

    grid = pd.read_csv(tmp_path / paths.name("eval_ext"), sep="\t")
    assert (grid["population"] == "snv").sum() == 23 * 16
    for p in delivered:
        assert (tmp_path / p.name).read_bytes() == p.read_bytes(), (
            f"a default evaluate_ext run no longer reproduces {p.name}")


def test_robustness_with_default_arguments_reproduces_the_delivered_files(tmp_path):
    """Same guard for the robustness checks of the current generation: ties, power,
    paired tests (the eighteen claims and the splice family's pairs), LOGO and the
    summary json, byte for byte. The bootstrap is seeded, so a default run is
    deterministic."""
    from atlas import paths
    delivered = [paths.path(k) for k in ("tie_audit", "power", "head_to_head",
                                         "head_to_head_pvalues", "logo", "robustness_summary")]
    if not (paths.path("score_matrix").exists() and all(p.exists() for p in delivered)):
        pytest.skip("results/ not built in this checkout")

    from atlas import robustness
    robustness.main(["--out", str(tmp_path)])

    for p in delivered:
        assert (tmp_path / p.name).read_bytes() == p.read_bytes(), (
            f"a default robustness run no longer reproduces {p.name}")
