"""The one way in to the score matrix, and the two populations it is read as.

snv      ref and alt are each a single base: 46,392 variants. The main analysis.
non_snv  the other 17,786 (deletions, insertions, complex), reported on their own.

Every module that reads the matrix does so through `load` (or `assign`, for a frame it
already holds), which adds two columns: ``population`` and ``region``. The territory
(``region``) is still defined in one place, `atlas.evaluate.classify_region`; this entry
point only applies it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from atlas import paths
from atlas.evaluate import classify_region

SNV = "snv"
NON_SNV = "non_snv"
POPULATIONS = (SNV, NON_SNV)
EXPECTED = {SNV: 46392, NON_SNV: 17786}


def is_snv(ref: pd.Series, alt: pd.Series) -> pd.Series:
    return ref.astype(str).str.len().eq(1) & alt.astype(str).str.len().eq(1)


def assign(df: pd.DataFrame) -> pd.DataFrame:
    """The frame with ``population`` (from ref and alt) and ``region`` (from hgvs_c)."""
    df = df.copy()
    df["population"] = is_snv(df["ref"], df["alt"]).map({True: SNV, False: NON_SNV})
    df["region"] = df["hgvs_c"].map(classify_region)
    return df


def select(df: pd.DataFrame, population: str | None = SNV) -> pd.DataFrame:
    """Rows of one population (None: both), the columns added if they are not there."""
    if "population" not in df.columns or "region" not in df.columns:
        df = assign(df)
    if population is None:
        return df
    if population not in POPULATIONS:
        raise ValueError(f"unknown population {population!r}; one of {POPULATIONS}")
    return df[df["population"] == population].copy()


def load(population: str | None = SNV, matrix: Path | str | None = None,
         columns: list[str] | None = None) -> pd.DataFrame:
    """The score matrix (the current one by default), one population of it or both."""
    src = Path(matrix) if matrix is not None else paths.path("score_matrix")
    need = None
    if columns is not None:
        need = list(dict.fromkeys(list(columns) + ["ref", "alt", "hgvs_c"]))
    return select(pd.read_parquet(src, columns=need), population)
