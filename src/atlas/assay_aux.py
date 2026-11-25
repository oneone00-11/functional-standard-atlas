"""Lift per-assay auxiliary measurements out of the raw MaveDB score files.

The frozen atlas keeps exactly one number per variant (``functional_pathogenicity``).
The MaveDB deposits carry considerably more: replicate scores, per-variant standard
errors, a second RNA-level readout, and the assay authors' own functional
classifications. Those columns support three analyses the atlas cannot otherwise
do — measurement-reliability ceilings, cross-readout replication, and
clinical-evidence calibration — so this module lifts them into one tidy table
keyed by ``variant_id``.

Column names and semantics differ per deposit, so every column is registered by
hand in ``AUX`` below rather than pattern-matched, and is *validated* wherever the
deposit makes validation possible (CONVENTIONS.md rule 4: never assume what a source
column means).

Validation performed
--------------------
* ``se``  — accepted only when it is checked against another quantity the deposit
  itself publishes (CONVENTIONS.md rule 4), and the column ``se_validated_by`` says
  which:

  ``ci``              the deposit's 95% CI: ``(ci_upper - score) / se == 1.96`` on
                      every row (BARD1, PALB2);
  ``test_statistic``  the deposit's own test statistic: ``score`` equals its LFC
                      column on every row, and ``score / se`` equals its z column to
                      within 1e-4 on every row (BAP1; the deposit rounds its values,
                      so equality to the bit is not required of the ratio);
  ``replicates``      the deposit's replicate scores, from which the SE of their mean
                      is derived (BRCA1).

  A check that fails on any row stops the build. Deposits with no such quantity are
  carried with ``se_status`` other than ``validated`` (RAD51C: ``scale_mismatch``) and
  stay out of the reliability estimate.
* ``rna`` — the atlas orientation rule is ``pathogenicity = -score``. An RNA
  readout is only carried if it is *positively* rank-correlated with the assay's
  own ``score``, i.e. the same orientation rule applies to it; the observed
  correlation is recorded in the provenance file.

Usage (PYTHONPATH=src):  python -m atlas.assay_aux   (writes results/assay_aux_v2.*)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import paths

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
RAW = REPO / "data" / "raw" / "mavedb"
FROZEN = paths.FROZEN

# --------------------------------------------------------------------------
# per-assay registry
# --------------------------------------------------------------------------
# se_status:
#   validated       — SE is on the score scale, checked against another quantity the
#                     deposit publishes (its CI or its test statistic; se_validated_by)
#   unvalidated     — SE plausibly on the score scale but nothing to check it against
#   scale_mismatch  — the deposit carries no error on the score's scale; do not use
AUX: dict[str, dict] = {
    "urn:mavedb:00000097-0-2": {
        "gene": "BRCA1",
        "fitness_reps": ("score_rep1", "score_rep2"),
        "rna": "score_rna",
        "rna_reps": ("score_rna_rep1", "score_rna_rep2"),
    },
    "urn:mavedb:00000662-0-1": {
        "gene": "BAP1",
        "se": "SE_bind_continuous",
        "se_status": "unvalidated",  # until the test statistic below reconciles it
        # no published CI; the deposit's own LFC and z of the same comparison instead
        "test_statistic": {"lfc": "processed_LFC_continuous",
                           "z": "processed_Z_continuous"},
    },
    "urn:mavedb:00000673-0-1": {
        "gene": "RAD51C",
        "se": "SE_bind_continuous",
        # score is processed_adj_lfc_D4_D14 / SE_bind_D4_D14 on every row (to 2.5e-13):
        # a z-score. The SE_bind columns are the SEs of the adjusted LFCs; the deposit
        # carries no error on the score's scale (sd(score) = 6.84, sd(SE_bind) = 0.018).
        "se_status": "scale_mismatch",
        "label_class": "functional_classification",
        "label_abnormal_values": ("fast depleted", "slow depleted"),
        "label_normal_values": ("unchanged",),
    },
    "urn:mavedb:00000675-a-1": {
        "gene": "VHL",
        "rna": "rna_score_d6",
        "rna_alt": "rna_score_d20",
    },
    "urn:mavedb:00001225-a-1": {"gene": "BRCA2"},  # score only
    "urn:mavedb:00001250-a-2": {
        "gene": "BARD1",
        "se": "standard_error",
        "se_status": "validated",
        "ci": ("95_ci_lower", "95_ci_upper"),
        "rna": "rna_score",
        "label_prob": "gmm_density_abnormal",
    },
    "urn:mavedb:00001259-a-2": {
        "gene": "PALB2",
        "se": "standard_error",
        "se_status": "validated",
        "ci": ("95_ci_lower", "95_ci_upper"),
        "rna": "rna_score",
        "label_prob": "gmm_density_abnormal",
    },
}

OUT_COLS = [
    "variant_id", "gene", "urn",
    "fit_rep1", "fit_rep2",
    "se", "se_status", "se_validated_by",
    "rna", "rna_alt", "rna_rep1", "rna_rep2",
    "label_abnormal", "label_class", "label_source",
]


def _urn_dir(urn: str) -> Path:
    return RAW / urn.replace(":", "_").replace("-", "-")


def _read_assay(urn: str) -> pd.DataFrame:
    d = RAW / ("urn_mavedb_" + urn.split(":")[-1])
    path = d / "scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"{urn}: {path} not found — run `make fetch` first")
    return pd.read_csv(path, low_memory=False)


Z_ATOL = 1e-4   # the deposit rounds its values; the ratio need not equal z to the bit


def check_test_statistic(raw: pd.DataFrame, se_col: str, cols: dict, gene: str) -> dict:
    """Is ``se`` the SE of ``score``, by the deposit's own test statistic? ``score`` must
    equal the LFC column on every row, and ``score / se`` the z column to within
    Z_ATOL on every row. One row that does not hold stops the build."""
    score = pd.to_numeric(raw["score"], errors="coerce")
    lfc = pd.to_numeric(raw[cols["lfc"]], errors="coerce")
    se = pd.to_numeric(raw[se_col], errors="coerce")
    z = pd.to_numeric(raw[cols["z"]], errors="coerce")
    same_na = score.isna().equals(lfc.isna())
    equal = bool(same_na and (score[score.notna()] == lfc[lfc.notna()]).all())
    if not equal:
        raise ValueError(f"{gene}: score does not equal {cols['lfc']} on every row")
    m = score.notna() | se.notna() | z.notna()
    if (score[m].isna() | se[m].isna() | z[m].isna()).any():
        raise ValueError(f"{gene}: score, {se_col} and {cols['z']} are not all present "
                         "on the same rows")
    diff = (score[m] / se[m] - z[m]).abs()
    if not (diff < Z_ATOL).all():
        raise ValueError(f"{gene}: score / {se_col} differs from {cols['z']} by "
                         f"{float(diff.max()):.3g} on some row (tolerance {Z_ATOL})")
    return {"score_equals": cols["lfc"], "rows_equal": int(score.notna().sum()),
            "z": cols["z"], "rows_checked": int(m.sum()),
            "max_abs_diff_score_over_se_vs_z": float(diff.max()), "tolerance": Z_ATOL}


def build(min_rna_rho: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Return (aux table, provenance record)."""
    prov: dict = {"assays": {}, "min_rna_rho": min_rna_rho}
    frames = []

    for urn, spec in AUX.items():
        gene = spec["gene"]
        raw = _read_assay(urn).set_index("accession")
        rec: dict = {"gene": gene, "n_rows": int(len(raw)), "carried": []}
        out = pd.DataFrame(index=raw.index)
        out["gene"] = gene
        out["urn"] = urn

        # -- fitness replicates -------------------------------------------
        if "fitness_reps" in spec:
            a, b = spec["fitness_reps"]
            out["fit_rep1"], out["fit_rep2"] = raw[a], raw[b]
            rec["carried"].append(f"fitness_reps={a},{b}")
            rec["rep_rho"] = float(raw[a].corr(raw[b], method="spearman"))
            # SE of the replicate mean, so this assay can also be scored by the
            # SE estimator and the two estimators cross-checked on one dataset.
            out["se"] = raw[[a, b]].std(axis=1, ddof=1) / np.sqrt(2)
            out["se_status"] = "derived_from_replicates"
            out["se_validated_by"] = "replicates"
            rec["carried"].append("se=sd(reps)/sqrt(2) (derived_from_replicates)")

        # -- standard error, checked against another quantity the deposit publishes
        if "se" in spec:
            se = pd.to_numeric(raw[spec["se"]], errors="coerce")
            status = spec.get("se_status", "unvalidated")
            validated_by = np.nan
            if "ci" in spec:
                lo, hi = spec["ci"]
                ratio = (raw[hi] - raw["score"]) / se
                ok = np.allclose(ratio.dropna(), 1.96, atol=1e-3)
                rec["se_ci_ratio_median"] = float(ratio.median())
                if not ok:
                    raise ValueError(
                        f"{gene}: {spec['se']} does not reconcile with {hi} "
                        f"(median (ci_hi-score)/se = {ratio.median():.4f}, expected 1.96)"
                    )
                status, validated_by = "validated", "ci"
            elif "test_statistic" in spec:
                rec["se_test_statistic"] = check_test_statistic(raw, spec["se"],
                                                                spec["test_statistic"], gene)
                status, validated_by = "validated", "test_statistic"
            out["se"] = se
            out["se_status"] = status
            out["se_validated_by"] = validated_by
            rec["carried"].append(f"se={spec['se']} ({status}"
                                  + (f", by {validated_by})" if isinstance(validated_by, str)
                                     else ")"))

        # -- RNA readout, orientation-checked against the assay's own score --
        for key, col in (("rna", spec.get("rna")), ("rna_alt", spec.get("rna_alt"))):
            if not col:
                continue
            v = pd.to_numeric(raw[col], errors="coerce")
            rho = float(v.corr(raw["score"], method="spearman"))
            rec[f"{key}_rho_with_score"] = rho
            if rho < min_rna_rho:
                rec["carried"].append(f"{key}={col} DROPPED (rho={rho:+.3f} < {min_rna_rho})")
                continue
            out[key] = v
            rec["carried"].append(f"{key}={col} (rho={rho:+.3f})")
        if "rna_reps" in spec:
            a, b = spec["rna_reps"]
            out["rna_rep1"], out["rna_rep2"] = raw[a], raw[b]
            rec["rna_rep_rho"] = float(raw[a].corr(raw[b], method="spearman"))
            rec["carried"].append(f"rna_reps={a},{b}")

        # -- functional labels ---------------------------------------------
        if "label_prob" in spec:
            p = pd.to_numeric(raw[spec["label_prob"]], errors="coerce")
            if not p.between(-1e-6, 1 + 1e-6).all():
                raise ValueError(f"{gene}: {spec['label_prob']} is not a probability")
            out["label_abnormal"] = p
            out["label_source"] = f"posterior:{spec['label_prob']}"
            rec["carried"].append(f"label_prob={spec['label_prob']}")
        elif "label_class" in spec:
            cls = raw[spec["label_class"]].astype(str)
            abn = set(spec["label_abnormal_values"])
            nor = set(spec["label_normal_values"])
            lab = pd.Series(np.nan, index=cls.index, dtype=float)
            lab[cls.isin(abn)] = 1.0
            lab[cls.isin(nor)] = 0.0
            out["label_abnormal"] = lab
            out["label_class"] = cls
            out["label_source"] = f"class:{spec['label_class']}"
            rec["carried"].append(f"label_class={spec['label_class']}")
            rec["label_counts"] = cls.value_counts().to_dict()

        prov["assays"][urn] = rec
        frames.append(out.reset_index().rename(columns={"accession": "variant_id"}))

    aux = pd.concat(frames, ignore_index=True)
    for c in OUT_COLS:
        if c not in aux.columns:
            aux[c] = np.nan
    aux = aux[OUT_COLS]

    # keep only variants that survived into the frozen atlas
    frozen = pd.read_parquet(FROZEN, columns=["variant_id"])
    before = len(aux)
    aux = aux[aux["variant_id"].isin(set(frozen["variant_id"]))].reset_index(drop=True)
    prov["n_rows_raw"] = before
    prov["n_rows_in_atlas"] = int(len(aux))
    prov["coverage"] = {
        c: int(aux[c].notna().sum())
        for c in ("fit_rep1", "se", "rna", "rna_alt", "rna_rep1", "label_abnormal")
    }
    prov["se_validated_by"] = {k: int(v) for k, v in
                               aux["se_validated_by"].value_counts().items()}
    return aux, prov


PREVIOUS = RESULTS / "assay_aux_v1.parquet"   # the generation v2 is checked against


def against_previous(aux: pd.DataFrame, previous: Path = PREVIOUS) -> dict:
    """v2 against v1: the same rows, and every column but se_status and se_validated_by
    equal to the bit; se_status may differ only where se_validated_by is test_statistic
    (unvalidated -> validated)."""
    from atlas.matrix_v3 import bitwise_equal
    v1 = pd.read_parquet(previous)
    same_rows = bool(aux["variant_id"].equals(v1["variant_id"]))
    cols = [c for c in v1.columns if c != "se_status"]
    equal = {c: bool(same_rows and bitwise_equal(aux[c], v1[c])) for c in cols}
    changed = aux["se_status"].fillna("") != v1["se_status"].fillna("")
    moved = aux.loc[changed, ["gene", "se_validated_by"]].assign(
        before=v1.loc[changed, "se_status"], after=aux.loc[changed, "se_status"])
    status_ok = bool(same_rows and (moved["se_validated_by"] == "test_statistic").all()
                     and (moved["before"] == "unvalidated").all()
                     and (moved["after"] == "validated").all())
    return {"previous": PREVIOUS.name, "same_rows": same_rows, "columns_equal": equal,
            "se_status_changed_rows": int(changed.sum()),
            "se_status_changed": {f"{g}: {b} -> {a}": int(n) for (g, b, a), n in
                                  moved.groupby(["gene", "before", "after"]).size().items()},
            "holds": bool(same_rows and all(equal.values()) and status_ok)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)

    aux, prov = build()
    if PREVIOUS.exists():
        prov["against_v1"] = against_previous(aux)
        if not prov["against_v1"]["holds"]:
            print("v2 differs from v1 beyond BAP1's se_status:", prov["against_v1"])
            return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    aux.to_parquet(out / paths.name("assay_aux"), index=False)
    (out / paths.name("assay_aux_provenance")).write_text(json.dumps(prov, indent=2) + "\n")

    print(f"{paths.name('assay_aux')}: {len(aux):,} rows (of {prov['n_rows_raw']:,} raw)")
    for k, v in prov["coverage"].items():
        print(f"  {k:<16} {v:>7,}")
    print("\nper-assay:")
    for urn, rec in prov["assays"].items():
        print(f"  {rec['gene']:<7} " + "; ".join(rec["carried"]) if rec["carried"] else f"  {rec['gene']:<7} (score only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
