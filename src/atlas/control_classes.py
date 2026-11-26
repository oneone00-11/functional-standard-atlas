"""The control classes of each gene (A4b): two classes of SNVs taken to share one true
effect each, neutral and nonsense, and the loss-of-function level they set.

Classes (SNVs with a functional score; y is functional_pathogenicity, higher further
from normal function):
  neutral   synonymous, spliceai_ds below 0.1 (atlas.error_models takes the same);
  nonsense  consequence "nonsense".
The core seven genes' from the score matrix, the external five's from
results/external_coding_consequence_v1.parquet (atlas.external_coding).

results/control_classes_v1.tsv, one row per gene:
  n_neutral, n_nonsense
  t_N, t_L        the classes' medians; gap = t_L - t_N; lof_threshold = (t_N + t_L) / 2.
                  A variant is at the loss-of-function level when its score is above
                  lof_threshold;
  sd_neutral, sd_nonsense
                  each class's robust SD, 1.4826 x the median absolute deviation from the
                  class's median, and its ratio to the gap (sd_*_over_gap);
  spread_ratio    sd_nonsense / sd_neutral;
  local_sd_ratio_*, local_corr_*
                  the position structure of each class. The class sorted by CDS position
                  (the N of c.N), one position's variants by alternative base A, C, G, T;
                  a variant's local median is the median score of the five variants
                  before it and the five after (as many as there are at the ends, the
                  variant itself left out). local_sd_ratio = the robust SD of score minus
                  local median over the class's robust SD; local_corr = the Pearson
                  correlation of score and local median;
  kappa           the eight SE genes: the robust SD of the nonsense residuals (score - t_L)
                  over sigma(SE), divided by the robust SD of the neutral residuals
                  (score - t_N) over sigma(SE); sigma is the gene's neutral error scale
                  (atlas.error_models.error_scale), variants with an SE only
                  (kappa_basis "sigma(SE)"). BRCA1: spread_ratio (kappa_basis
                  "spread_ratio"). RAD51C, BRCA2, VHL: empty.

The nonsense class's spread is not taken into the neutral model: it is not all noise (see
its position structure). It describes the loss-of-function end, and the table's levels
set where the neutral model applies.

Usage (PYTHONPATH=src):  python -m atlas.control_classes
"""

from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import error_models as em
from atlas import paths, population
from atlas import rank_ceiling as rc

CORE = ("BRCA1", "BARD1", "PALB2", "BAP1", "RAD51C", "BRCA2", "VHL")
EXTERNAL = em.EXTERNAL
GENES = CORE + EXTERNAL
NEIGHBOURS = 5
CDS_SNV = re.compile(r"^c\.(\d+)([ACGT])>([ACGT])$")
ALT_ORDER = {"A": 0, "C": 1, "G": 2, "T": 3}

COLUMNS = ["gene", "group", "source", "n_neutral", "n_nonsense", "t_N", "t_L", "gap",
           "lof_threshold", "sd_neutral", "sd_nonsense", "sd_neutral_over_gap",
           "sd_nonsense_over_gap", "spread_ratio", "local_sd_ratio_neutral",
           "local_corr_neutral", "local_sd_ratio_nonsense", "local_corr_nonsense", "kappa",
           "kappa_basis"]


# ==========================================================================
# the classes
# ==========================================================================
@lru_cache(maxsize=1)
def _core() -> pd.DataFrame:
    m = population.load(population.SNV,
                        columns=["variant_id", "gene", "functional_pathogenicity",
                                 "consequence", "spliceai_ds"])
    aux = pd.read_parquet(paths.path("assay_aux"),
                          columns=["variant_id", "gene", "se", "se_status"])
    return m.merge(aux, on=["variant_id", "gene"], how="left")


def _frame(gene: str) -> pd.DataFrame:
    """The gene's SNVs with a functional score: id, hgvs_c, y, se (the SE the gene's
    error model uses; NaN for a gene without one), consequence, spliceai_ds."""
    if gene in EXTERNAL:
        x = em.external_coding()
        g = x[x["gene"].eq(gene) & x["functional_pathogenicity"].notna()]
        return pd.DataFrame({"id": g["variant_urn"], "hgvs_c": g["hgvs_c"].astype(str),
                             "y": g["functional_pathogenicity"], "se": g["standard_error"],
                             "consequence": g["consequence"], "spliceai_ds": g["spliceai_ds"]})
    if gene not in CORE:
        raise ValueError(f"{gene} is not one of the twelve genes")
    c = _core()
    g = c[c["gene"].eq(gene) & c["functional_pathogenicity"].notna()]
    se = (g["se"].where(g["se_status"].eq("validated")) if gene in em.CORE_SE
          else pd.Series(np.nan, index=g.index))
    return pd.DataFrame({"id": g["variant_id"], "hgvs_c": g["hgvs_c"].astype(str),
                         "y": g["functional_pathogenicity"], "se": se,
                         "consequence": g["consequence"], "spliceai_ds": g["spliceai_ds"]})


@lru_cache(maxsize=None)
def classes(gene: str) -> dict[str, pd.DataFrame]:
    """The gene's two classes, each sorted by identifier."""
    d = _frame(gene)
    neutral = d[d["consequence"].eq("synonymous") & (d["spliceai_ds"] < em.SPLICEAI_NEUTRAL)]
    nonsense = d[d["consequence"].eq("nonsense")]
    return {k: v.sort_values("id", kind="stable").reset_index(drop=True)
            for k, v in (("neutral", neutral), ("nonsense", nonsense))}


# ==========================================================================
# the columns
# ==========================================================================
def cds_order(hgvs_c) -> np.ndarray:
    """The order that sorts CDS SNVs (c.N X>Y) by N, then by the alternative base A, C,
    G, T; ties keep the given order."""
    m = pd.Series(list(hgvs_c), dtype=str).str.extract(CDS_SNV)
    if m[0].isna().any():
        bad = pd.Series(list(hgvs_c))[m[0].isna()].head(3).tolist()
        raise ValueError(f"not a CDS SNV: {bad}")
    pos = m[0].astype(int).to_numpy()
    alt = m[2].map(ALT_ORDER).to_numpy()
    return np.lexsort((alt, pos))


def local_median(v, k: int = NEIGHBOURS) -> np.ndarray:
    """Per position, the median of the k values before it and the k after (fewer at the
    ends), itself left out."""
    v = np.asarray(v, float)
    return np.array([np.median(np.concatenate([v[max(0, i - k):i], v[i + 1:i + 1 + k]]))
                     for i in range(len(v))])


def local_structure(hgvs_c, y, k: int = NEIGHBOURS) -> tuple[float, float]:
    """(robust SD of y minus its local median over the robust SD of y, the Pearson
    correlation of y and its local median), along the CDS."""
    v = np.asarray(y, float)[cds_order(hgvs_c)]
    loc = local_median(v, k)
    return rc.robust_sd(v - loc) / rc.robust_sd(v), float(np.corrcoef(v, loc)[0, 1])


def class_columns(neutral: pd.DataFrame, nonsense: pd.DataFrame, scale=None,
                  kappa_is_spread: bool = False) -> dict:
    """One gene's columns from its two classes (frames with hgvs_c and y; with se where a
    scale is given). scale: sigma(SE), for kappa; kappa_is_spread: kappa = spread_ratio;
    neither: kappa empty."""
    yn, yl = neutral["y"].to_numpy(float), nonsense["y"].to_numpy(float)
    t_n, t_l = float(np.median(yn)), float(np.median(yl))
    gap = t_l - t_n
    sd_n, sd_l = rc.robust_sd(yn), rc.robust_sd(yl)
    ln, cn = local_structure(neutral["hgvs_c"], yn)
    ll, cl = local_structure(nonsense["hgvs_c"], yl)
    kappa, basis = np.nan, ""
    if scale is not None:
        a = neutral[neutral["se"].notna()]
        b = nonsense[nonsense["se"].notna()]
        za = (a["y"].to_numpy(float) - t_n) / scale(a["se"].to_numpy(float))
        zb = (b["y"].to_numpy(float) - t_l) / scale(b["se"].to_numpy(float))
        kappa, basis = rc.robust_sd(zb) / rc.robust_sd(za), "sigma(SE)"
    elif kappa_is_spread:
        kappa, basis = sd_l / sd_n, "spread_ratio"
    return {"n_neutral": int(len(yn)), "n_nonsense": int(len(yl)), "t_N": t_n, "t_L": t_l,
            "gap": gap, "lof_threshold": (t_n + t_l) / 2, "sd_neutral": sd_n,
            "sd_nonsense": sd_l, "sd_neutral_over_gap": sd_n / gap,
            "sd_nonsense_over_gap": sd_l / gap, "spread_ratio": sd_l / sd_n,
            "local_sd_ratio_neutral": ln, "local_corr_neutral": cn,
            "local_sd_ratio_nonsense": ll, "local_corr_nonsense": cl,
            "kappa": kappa, "kappa_basis": basis}


def gene_row(gene: str) -> dict:
    c = classes(gene)
    scale = em.error_scale(gene) if gene in em.SE_GENES else None
    cols = class_columns(c["neutral"], c["nonsense"], scale, kappa_is_spread=gene == "BRCA1")
    return {"gene": gene, "group": "external" if gene in EXTERNAL else "core",
            "source": paths.name("external_coding_consequence") if gene in EXTERNAL
            else paths.name("score_matrix"), **cols}


def table() -> pd.DataFrame:
    return pd.DataFrame([gene_row(g) for g in GENES]).reindex(columns=COLUMNS)


# ==========================================================================
# the levels, for the tables that read them
# ==========================================================================
@lru_cache(maxsize=1)
def levels(path: Path | None = None) -> pd.DataFrame:
    """t_N, t_L and lof_threshold per gene, from results/control_classes_v1.tsv."""
    p = Path(path) if path is not None else paths.path("control_classes")
    t = pd.read_csv(p, sep="\t", float_precision="round_trip")
    return t.set_index("gene")[["t_N", "t_L", "gap", "lof_threshold", "kappa"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(paths.RESULTS))
    args = ap.parse_args(argv)
    t = table()
    t.to_csv(Path(args.out) / paths.name("control_classes"), sep="\t", index=False)
    print(t.drop(columns=["source"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
