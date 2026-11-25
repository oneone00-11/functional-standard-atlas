"""Assemble score matrix v3: AlphaGenome by its official definition, the AlphaGenome
Atlas's four columns, the variant class and the ClinVar record status.

``score_matrix_atlas_v2.parquet`` is left untouched (CONVENTIONS.md rule 1). v3 has its
64,178 rows in the frozen matrix's order and, per column:

  v2's 33 columns but ``alphagenome``   carried over as they are
  ``alphagenome``                      the official definition's merged splicing score
                                       (``alphagenome_official`` of A3.1,
                                       results/alphagenome_official_scores_v1.parquet);
                                       v2's older ``alphagenome`` stays in v2 only
  ``alphagenome_splice_sites``,        the three scorers beside it, of the same file;
  ``_splice_site_usage``,              auxiliary columns, not predictors
  ``_splice_junctions``
  ``avi``, ``avi_splice_sites``,       the Atlas's four columns (A3.2,
  ``avi_splice_site_usage``,           results/alphagenome_atlas_scores_v1.parquet, the
  ``avi_splice_junctions``             rows of set == core); empty for the non-SNVs, which
                                       the Atlas does not hold
  ``variant_class``                    atlas.alphagenome_official.kind(ref, alt): snv,
                                       deletion, insertion or complex
  ``clinvar_arm``, ``clinvar_arm_strict``  A3.3's arm and arm_strict
                                       (results/clinvar_status_v1.parquet)

The build stops unless every acceptance check holds (`checks`): the carried columns equal
v2's to the bit, NaN where v2 has NaN; the AlphaGenome columns equal A3.1's to the bit,
with no empty value; the Atlas columns equal A3.2's to the bit on the 46,392 SNVs, with no
empty value, and are empty on the 17,786 others; the variant classes and the ClinVar arms
have the counts A3 recorded.

Usage (PYTHONPATH=src):  python -m atlas.matrix_v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import paths
from atlas.alphagenome_official import kind
from atlas.manifest import sha256_file
from atlas.population import NON_SNV, SNV, is_snv

REPO = paths.REPO
RESULTS = paths.RESULTS
SOURCE = RESULTS / "score_matrix_atlas_v2.parquet"   # the generation v3 is built from

AG_COLUMNS = {"alphagenome": "alphagenome_official",
              "alphagenome_splice_sites": "alphagenome_splice_sites",
              "alphagenome_splice_site_usage": "alphagenome_splice_site_usage",
              "alphagenome_splice_junctions": "alphagenome_splice_junctions"}
ATLAS_COLUMNS = ["avi", "avi_splice_sites", "avi_splice_site_usage", "avi_splice_junctions"]
CLINVAR_COLUMNS = {"clinvar_arm": "arm", "clinvar_arm_strict": "arm_strict"}
ADDED = list(AG_COLUMNS)[1:] + ATLAS_COLUMNS + ["variant_class"] + list(CLINVAR_COLUMNS)

EXPECTED_CLASSES = {"snv": 46392, "deletion": 9714, "insertion": 64, "complex": 8008}
EXPECTED_ARMS = {"classified": 8597, "recorded_unclassified": 13120, "unrecorded": 42461}
EXPECTED_NO_ASSERTION = 2046


def bitwise_equal(a: pd.Series, b: pd.Series) -> bool:
    """Equal value by value, NaN where the other is NaN; floats to the bit."""
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    if len(a) != len(b):
        return False
    if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
        x, y = a.to_numpy(np.float64), b.to_numpy(np.float64)
        nx, ny = np.isnan(x), np.isnan(y)
        return bool(np.array_equal(nx, ny)
                    and np.array_equal(x[~nx].view(np.uint64), y[~ny].view(np.uint64)))
    na, nb = a.isna().to_numpy(), b.isna().to_numpy()
    if not np.array_equal(na, nb):
        return False
    return bool((a[~na].astype(object).to_numpy() == b[~nb].astype(object).to_numpy()).all())


def _in_order(frame: pd.DataFrame, ids: pd.Series, what: str) -> pd.DataFrame:
    """frame, which must hold the variants of ids in the same order, or stop."""
    if not frame["variant_id"].reset_index(drop=True).equals(ids.reset_index(drop=True)):
        raise SystemExit(f"{what}: its rows are not the frozen matrix's, in its order")
    return frame.reset_index(drop=True)


def build(results: Path = RESULTS, source: Path = SOURCE) -> tuple[pd.DataFrame, dict]:
    frozen = pd.read_parquet(paths.FROZEN, columns=["variant_id"])["variant_id"]
    v2 = _in_order(pd.read_parquet(source), frozen, "matrix v2")
    ag = _in_order(pd.read_parquet(paths.path("alphagenome_official_scores", results)),
                   frozen, "alphagenome_official_scores")
    cv = _in_order(pd.read_parquet(paths.path("clinvar_status", results)), frozen,
                   "clinvar_status")
    at = pd.read_parquet(paths.path("alphagenome_atlas_scores", results))
    at = at[at["set"] == "core"].reset_index(drop=True)
    snv = is_snv(v2["ref"], v2["alt"]).to_numpy()
    at = _in_order(at, frozen[snv], "alphagenome_atlas_scores (core)")

    df = v2.copy()
    for col, src in AG_COLUMNS.items():
        df[col] = ag[src].to_numpy()
    for col in ATLAS_COLUMNS:
        vals = np.full(len(df), np.nan)
        vals[snv] = at[col].to_numpy(np.float64)
        df[col] = vals
    df["variant_class"] = pd.Series([kind(r, a) for r, a in zip(df["ref"], df["alt"])],
                                    dtype=v2["ref"].dtype)
    for col, src in CLINVAR_COLUMNS.items():
        df[col] = cv[src].to_numpy()
    df = df[list(v2.columns) + ADDED]

    # the checks read the sources afresh, so that none of them is true by construction
    v2r = pd.read_parquet(source)
    agr = pd.read_parquet(paths.path("alphagenome_official_scores", results))
    atr = pd.read_parquet(paths.path("alphagenome_atlas_scores", results))
    atr = atr[atr["set"] == "core"].reset_index(drop=True)
    carried = [c for c in v2r.columns if c != "alphagenome"]
    checks = {
        "rows": int(len(df)),
        "order_is_the_frozen_matrix": bool(df["variant_id"].reset_index(drop=True)
                                           .equals(frozen.reset_index(drop=True))),
        "carried_columns": len(carried),
        "carried_equal_v2": {c: bitwise_equal(df[c], v2r[c]) for c in carried},
        "alphagenome_equal_A3_1": {c: bool(df["variant_id"].equals(agr["variant_id"])
                                           and bitwise_equal(df[c], agr[s]))
                                   for c, s in AG_COLUMNS.items()},
        "alphagenome_empty": {c: int(df[c].isna().sum()) for c in AG_COLUMNS},
        "atlas_equal_A3_2_on_snv": {
            c: bool(len(atr) == int(snv.sum())
                    and df.loc[snv, "variant_id"].reset_index(drop=True).equals(atr["variant_id"])
                    and bitwise_equal(df.loc[snv, c], atr[c]))
            for c in ATLAS_COLUMNS},
        "atlas_empty_on_snv": {c: int(df.loc[snv, c].isna().sum()) for c in ATLAS_COLUMNS},
        "atlas_filled_on_non_snv": {c: int(df.loc[~snv, c].notna().sum()) for c in ATLAS_COLUMNS},
        "population": {SNV: int(snv.sum()), NON_SNV: int((~snv).sum())},
        "variant_class": {k: int(v) for k, v in df["variant_class"].value_counts().items()},
        "clinvar_arm": {k: int(v) for k, v in df["clinvar_arm"].value_counts().items()},
        "clinvar_arm_strict": {k: int(v) for k, v in df["clinvar_arm_strict"].value_counts().items()},
    }
    failures = []
    if not checks["order_is_the_frozen_matrix"]:
        failures.append("row order")
    failures += [f"carried {c}" for c, ok in checks["carried_equal_v2"].items() if not ok]
    failures += [f"AlphaGenome {c}" for c, ok in checks["alphagenome_equal_A3_1"].items() if not ok]
    failures += [f"AlphaGenome {c} empty" for c, n in checks["alphagenome_empty"].items() if n]
    failures += [f"Atlas {c}" for c, ok in checks["atlas_equal_A3_2_on_snv"].items() if not ok]
    failures += [f"Atlas {c} empty on an SNV" for c, n in checks["atlas_empty_on_snv"].items() if n]
    failures += [f"Atlas {c} on a non-SNV" for c, n in checks["atlas_filled_on_non_snv"].items() if n]
    if checks["variant_class"] != EXPECTED_CLASSES:
        failures.append(f"variant classes {checks['variant_class']}")
    if checks["clinvar_arm"] != EXPECTED_ARMS:
        failures.append(f"ClinVar arms {checks['clinvar_arm']}")
    if checks["clinvar_arm_strict"].get("recorded_no_assertion") != EXPECTED_NO_ASSERTION:
        failures.append(f"ClinVar strict arms {checks['clinvar_arm_strict']}")
    checks["failures"] = failures
    return df, checks


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else str(p)


def provenance(df: pd.DataFrame, checks: dict, out_file: Path, results: Path = RESULTS,
               source: Path = SOURCE) -> dict:
    agp = json.loads(paths.path("alphagenome_official_provenance", results).read_text())
    atp = json.loads(paths.path("alphagenome_atlas_provenance", results).read_text())

    def src(key: str) -> dict:
        p = paths.path(key, results)
        return {"file": _rel(p), "sha256": sha256_file(p)}

    v2 = {"file": _rel(source), "sha256": sha256_file(source)}
    columns = {c: {"from": v2, "column": c} for c in df.columns
               if c not in AG_COLUMNS and c not in ADDED}
    for c, s in AG_COLUMNS.items():
        columns[c] = {"from": src("alphagenome_official_scores"), "column": s}
    for c in ATLAS_COLUMNS:
        columns[c] = {"from": src("alphagenome_atlas_scores"), "column": c,
                      "rows": "set == core; empty for the non-SNVs"}
    columns["variant_class"] = {"from": "atlas.alphagenome_official.kind(ref, alt)"}
    for c, s in CLINVAR_COLUMNS.items():
        columns[c] = {"from": src("clinvar_status"), "column": s}
    return {
        "file": _rel(out_file), "sha256": sha256_file(out_file), "rows": int(len(df)),
        "columns_n": int(df.shape[1]),
        "written_by": "atlas.matrix_v3",
        "built_from": v2,
        "row_order": "the frozen matrix's (data/frozen/frozen-matrix-v1.parquet)",
        "columns": columns,
        "alphagenome": {
            "definition": ("the official variant-scoring notebook's merged splicing score: "
                           "per variant, max(SPLICE_SITES) + max(SPLICE_SITE_USAGE) + "
                           "max(SPLICE_JUNCTIONS) / 5 over the raw scores, 1 Mb interval "
                           "centred on the variant (compute_merged_splicing_score, as "
                           "written)"),
            "client": agp["client"], "scorer": agp["scorer"], "definition_source": agp["definition"],
            "provenance": src("alphagenome_official_provenance"),
            "v2_column": "v2's alphagenome (the older definition, quantile scores, 16 kb) "
                         "is not carried; it stays in score_matrix_atlas_v2.parquet"},
        "atlas": {"product": atp["product"], "client_version": atp["client_version"],
                  "aggregation": atp["aggregation"],
                  "provenance": src("alphagenome_atlas_provenance"),
                  "dependency": atp["dependency"],
                  "selection_overlap": "atlas.predictors.SELECTION_OVERLAP: avi on BRCA1 "
                                       "and RAD51C is flagged and kept out of every "
                                       "quantity across genes"},
        "terms_of_use": {
            "alphagenome_columns": {"licence": agp["licence"],
                                    "licence_source": atp["licence_source"]},
            "atlas_columns": {"licence": atp["licence"], "licence_source": atp["licence_source"]},
            "use": "research only; the scores train no other model"},
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)
    out = Path(args.out)
    df, checks = build()
    if checks["failures"]:
        print("acceptance fails:", checks["failures"])
        return 1
    out.mkdir(parents=True, exist_ok=True)
    target = out / paths.name("score_matrix")
    df.to_parquet(target, index=False)
    prov = provenance(df, checks, target)
    (out / paths.name("score_matrix_provenance")).write_text(
        json.dumps(prov, indent=1, ensure_ascii=False) + "\n")
    print(f"{target.name}: {df.shape[0]:,} x {df.shape[1]}; checks hold")
    print("  variant_class", checks["variant_class"])
    print("  clinvar_arm", checks["clinvar_arm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
