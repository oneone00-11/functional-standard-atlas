"""Reliability ceilings for six external deposits, from their own error models.

The v5 attenuation ceiling (`atlas.reliability`) rests on three genes: BRCA1 from
replicates, BARD1 and PALB2 from validated standard errors. In the +-1-2 and 11-50 bp
strata those three already differ by more than 0.20. This module asks whether the
pattern "reliability varies by territory" holds in six further experiments.
It applies the v5 estimators, unchanged, to each deposit's own error information.
It estimates ceilings only. Realisation needs predictor scores and is not computed
here.

Deposits
--------
The external genes of the companion splice-region benchmark, fetched from MaveDB
with `atlas.fetch_mavedb --out data/raw/mavedb_external` and registered in that
directory's MANIFEST.json. They are kept out of `data/raw/mavedb/` and out of
`config/assays.yaml`: that registry feeds `atlas.mapping` and hence the frozen atlas,
and these deposits are not part of it.

``se``          CTCF, SFPQ, RAD51D, XRCC2, SBDS. The standard error is accepted only if
                ``(ci_upper - score) / se == 1.96`` and ``(score - ci_lower) / se == 1.96``
                on every row, the check `atlas.assay_aux` applies to BARD1 and PALB2.
                A deposit that fails is reported with status ``ci_mismatch`` and kept
                out of the main estimate. Reliability is computed as in v5:
                ``rel = 1 - mean(SE^2) / var(score)``, clipped to [0, 1].
``replicates``  TP53. Three replicate scores. The main estimate needs the deposited
                score to be their mean, so that Spearman-Brown applies. In this
                deposit it is their median (every row), so TP53 does not enter the
                main estimate (CONVENTIONS.md rule 4). It is reported on rows of its
                own, from the mean of the three pairwise Spearman r. The k = 1 row is
                the reliability of a single replicate and k = 3 that of the mean of
                three. With light-tailed (for example normal) errors the median of
                three lies between them. These rows are labelled ``lower_bound`` and
                ``upper_bound`` on that understanding.

DDX3X carries neither a standard error nor replicate scores and is not estimated
(listed under ``not_estimated`` in the JSON).

Variants are placed in the `atlas.reliability.STRATA` territories with
`atlas.evaluate.classify_region`. That function expects a bare c. string: its pattern
is anchored at ``c.``, so a transcript-prefixed name such as ``NM_006565.4:c.782-23T>A``
would be filed as coding. Every deposit here names variants on its MANE Select
transcript. The prefix is checked against the registry and removed. A row on any
other transcript is an error, not a silent drop.

Outputs
-------
results/external_reliability_v1.tsv   one row per deposit x stratum x estimator
results/external_reliability_v1.json  per-stratum gene counts, median and range of the
    ceiling. The main estimate combines the three v5 genes (read from
    results/reliability_v1.tsv, not recomputed, v5's primary route) with the external
    rows marked ``main``. The JSON also gives v5 alone, the external genes alone, and
    the main estimate with TP53 added at each bound.

Usage (PYTHONPATH=src):  python -m atlas.external_reliability --out results/
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.evaluate import classify_region
from atlas.manifest import load_manifest, register_file, sha256_file
from atlas.reliability import MIN_N, STRATA, _stratum_mask, spearman_brown

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "mavedb_external"
RESULTS = REPO / "results"
V5_TABLE = RESULTS / "reliability_v1.tsv"

# Each deposit names its variants on this transcript. All six are MANE Select
# (MANE v1.5); SBDS uses the Ensembl member of its MANE pair (NM_016038.4).
EXTERNAL: dict[str, dict] = {
    "urn:mavedb:00001262-a-3": {"gene": "CTCF", "transcript": "NM_006565.4",
                                "method": "se", "se": "standard_error",
                                "ci": ("95_ci_lower", "95_ci_upper")},
    "urn:mavedb:00001265-a-2": {"gene": "SFPQ", "transcript": "NM_005066.3",
                                "method": "se", "se": "standard_error",
                                "ci": ("95_ci_lower", "95_ci_upper")},
    "urn:mavedb:00001260-a-2": {"gene": "RAD51D", "transcript": "NM_002878.4",
                                "method": "se", "se": "standard_error",
                                "ci": ("95_ci_lower", "95_ci_upper")},
    "urn:mavedb:00001264-a-2": {"gene": "XRCC2", "transcript": "NM_005431.2",
                                "method": "se", "se": "standard_error",
                                "ci": ("95_ci_lower", "95_ci_upper")},
    "urn:mavedb:00001284-a-1": {"gene": "SBDS", "transcript": "ENST00000246868.7",
                                "method": "se", "se": "standard_error",
                                "ci": ("95_ci_lower", "95_ci_upper")},
    "urn:mavedb:00001277-a-1": {"gene": "TP53", "transcript": "NM_000546.6",
                                "method": "replicates",
                                "replicates": ("replicate_1_score", "replicate_2_score",
                                               "replicate_3_score")},
}

NOT_ESTIMATED = [{
    "gene": "DDX3X",
    "urns": "urn:mavedb:00000658-0-1 and its tile score sets -a-1 to -p-1",
    "why": ("the deposits carry the score, time-point log fold changes and FDRs, "
            "but no per-variant standard error and no replicate scores"),
}]

CI_Z = 1.96
CI_ATOL = 1e-3           # the tolerance atlas.assay_aux uses
SUMMARY_ATOL = 1e-12     # "equal" for the replicate-summary check

COLUMNS = ["gene", "urn", "stratum", "method", "n", "rep_rho", "k", "score_var",
           "mean_se2", "reliability", "ceiling", "status", "role"]


# --------------------------------------------------------------------------
# reading a deposit
# --------------------------------------------------------------------------
def hgvs_c(hgvs_nt: str, transcript: str) -> str:
    """The bare c. name, after checking the transcript prefix."""
    acc, sep, rest = str(hgvs_nt).partition(":")
    if not sep or acc != transcript or not rest.startswith("c."):
        raise ValueError(f"{hgvs_nt!r} is not a c. name on {transcript}")
    return rest


def load_deposit(urn: str, raw: Path = RAW) -> pd.DataFrame:
    """The deposit's score table with bare c. names and territories added."""
    spec = EXTERNAL[urn]
    path = raw / ("urn_mavedb_" + urn.split(":")[-1]) / "scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"{urn}: {path} not found; fetch it with "
                                f"`python -m atlas.fetch_mavedb --urn {urn} "
                                f"--out {raw.relative_to(REPO) if raw.is_relative_to(REPO) else raw}`")
    df = pd.read_csv(path, low_memory=False)
    df["hgvs_c"] = [hgvs_c(h, spec["transcript"]) for h in df["hgvs_nt"]]
    df["region"] = df["hgvs_c"].map(classify_region)
    return df


# --------------------------------------------------------------------------
# checks on the error model (CONVENTIONS.md rule 4)
# --------------------------------------------------------------------------
def check_se(df: pd.DataFrame, se_col: str, ci: tuple[str, str]) -> dict:
    """Does the SE reproduce the deposit's own 95% interval on every row?"""
    score = pd.to_numeric(df["score"], errors="coerce")
    se = pd.to_numeric(df[se_col], errors="coerce")
    lo = pd.to_numeric(df[ci[0]], errors="coerce")
    hi = pd.to_numeric(df[ci[1]], errors="coerce")
    upper = ((hi - score) / se).dropna()
    lower = ((score - lo) / se).dropna()
    ok = (len(upper) > 0 and len(lower) > 0
          and bool(np.allclose(upper, CI_Z, atol=CI_ATOL))
          and bool(np.allclose(lower, CI_Z, atol=CI_ATOL)))
    return {"status": "validated" if ok else "ci_mismatch",
            "rows_checked": int(len(upper)),
            "upper_ratio_min": float(upper.min()) if len(upper) else None,
            "upper_ratio_max": float(upper.max()) if len(upper) else None,
            "lower_ratio_min": float(lower.min()) if len(lower) else None,
            "lower_ratio_max": float(lower.max()) if len(lower) else None}


def replicate_summary(df: pd.DataFrame, reps: tuple[str, ...]) -> str:
    """Which summary of the replicates the deposited score is: mean, median or other."""
    r = df[list(reps)].apply(pd.to_numeric, errors="coerce")
    s = pd.to_numeric(df["score"], errors="coerce")
    keep = s.notna() & r.notna().all(axis=1)
    s, r = s[keep], r[keep]
    if len(s) == 0:
        return "other"
    if np.allclose(s, r.mean(axis=1), rtol=0, atol=SUMMARY_ATOL):
        return "mean"
    if np.allclose(s, r.median(axis=1), rtol=0, atol=SUMMARY_ATOL):
        return "median"
    return "other"


# --------------------------------------------------------------------------
# estimators (the v5 formulas)
# --------------------------------------------------------------------------
def _ceiling(rel: float) -> float:
    return float(np.sqrt(max(rel, 0.0))) if np.isfinite(rel) else float("nan")


def se_rows(df: pd.DataFrame, gene: str, urn: str, se_col: str, status: str) -> list[dict]:
    score = pd.to_numeric(df["score"], errors="coerce")
    se = pd.to_numeric(df[se_col], errors="coerce")
    x = pd.DataFrame({"score": score, "se": se, "region": df["region"]}).dropna(
        subset=["score", "se"])
    rows = []
    for stratum in STRATA:
        s = x[_stratum_mask(x["region"], stratum)]
        if len(s) < MIN_N:
            continue
        var = float(s["score"].var())
        err = float((s["se"] ** 2).mean())
        rel = float(np.clip(1 - err / var, 0.0, 1.0)) if var > 0 else float("nan")
        rows.append({"gene": gene, "urn": urn, "stratum": stratum, "method": "se",
                     "n": int(len(s)), "rep_rho": np.nan, "k": np.nan,
                     "score_var": var, "mean_se2": err, "reliability": rel,
                     "ceiling": _ceiling(rel), "status": status,
                     "role": "main" if status == "validated" else "excluded"})
    return rows


def replicate_rows(df: pd.DataFrame, gene: str, urn: str, reps: tuple[str, ...],
                   summary: str) -> list[dict]:
    """Mean pairwise Spearman r, stepped up with Spearman-Brown.

    A mean-of-replicates score enters the main estimate at k = len(reps). Any other
    summary is reported as two bounding rows, k = 1 and k = len(reps).
    """
    r = df[list(reps)].apply(pd.to_numeric, errors="coerce")
    x = r.assign(region=df["region"]).dropna(subset=list(reps))
    k_all = len(reps)
    if summary == "mean":
        plan = [(k_all, "validated", "main")]
    else:
        status = f"{summary}_of_replicates"
        plan = [(1, status, "lower_bound"), (k_all, status, "upper_bound")]
    rows = []
    for stratum in STRATA:
        s = x[_stratum_mask(x["region"], stratum)]
        if len(s) < MIN_N:
            continue
        rbar = float(np.mean([s[a].corr(s[b], method="spearman")
                              for a, b in combinations(reps, 2)]))
        for k, status, role in plan:
            rel = spearman_brown(rbar, k)
            rows.append({"gene": gene, "urn": urn, "stratum": stratum,
                         "method": "replicates", "n": int(len(s)), "rep_rho": rbar,
                         "k": k, "score_var": np.nan, "mean_se2": np.nan,
                         "reliability": rel, "ceiling": _ceiling(rel),
                         "status": status, "role": role})
    return rows


def external_table(raw: Path = RAW) -> tuple[pd.DataFrame, dict]:
    """(one row per deposit x stratum x estimator, per-deposit check record)."""
    rows: list[dict] = []
    checks: dict = {}
    for urn, spec in EXTERNAL.items():
        df = load_deposit(urn, raw)
        rec = {"gene": spec["gene"], "transcript": spec["transcript"],
               "rows": int(len(df)),
               "rows_by_region": {k: int(v) for k, v in
                                  df["region"].value_counts().sort_index().items()}}
        if spec["method"] == "se":
            chk = check_se(df, spec["se"], spec["ci"])
            rec["se_check"] = chk
            rows += se_rows(df, spec["gene"], urn, spec["se"], chk["status"])
        else:
            summary = replicate_summary(df, spec["replicates"])
            rec["score_is"] = f"{summary} of {', '.join(spec['replicates'])}"
            rows += replicate_rows(df, spec["gene"], urn, spec["replicates"], summary)
        checks[urn] = rec
    return pd.DataFrame(rows, columns=COLUMNS), checks


# --------------------------------------------------------------------------
# combining with v5
# --------------------------------------------------------------------------
from atlas.reliability import METHOD_PREFERENCE  # noqa: E402  (one order, shared)


def v5_primary(v5: pd.DataFrame) -> pd.DataFrame:
    """v5's main-estimate route: validated rows only, and the replicate estimate where
    a gene x stratum carries two validated rows.

    The preference is explicit. `atlas.reliability.realisation_table` states the same
    intent but sorts the method names in descending order, which would keep 'se'.
    No gene x stratum in the delivered v5 table has two validated rows, so the two
    selections coincide there (tests/test_external_reliability.py checks this).
    """
    p = v5[v5["status"] == "validated"].copy()
    p["_order"] = p["method"].map(METHOD_PREFERENCE).fillna(len(METHOD_PREFERENCE))
    p = (p.sort_values(["gene", "stratum", "_order"], kind="stable")
          .drop_duplicates(["gene", "stratum"], keep="first"))
    return p[["gene", "stratum", "method", "n", "ceiling"]].reset_index(drop=True)


def stratum_summary(cells: pd.DataFrame) -> list[dict]:
    """Per stratum: genes taking part, median and range of the ceiling."""
    out = []
    for stratum in STRATA:
        s = cells[cells["stratum"] == stratum]
        if s.empty:
            continue
        out.append({"stratum": stratum, "n_genes": int(s["gene"].nunique()),
                    "genes": sorted(s["gene"].unique().tolist()),
                    "ceiling_median": float(s["ceiling"].median()),
                    "ceiling_min": float(s["ceiling"].min()),
                    "ceiling_max": float(s["ceiling"].max()),
                    "ceiling_spread": float(s["ceiling"].max() - s["ceiling"].min())})
    return out


def summarise(ext: pd.DataFrame, v5: pd.DataFrame) -> dict:
    base = v5_primary(v5)[["gene", "stratum", "ceiling"]]
    main_ext = ext[ext["role"] == "main"][["gene", "stratum", "ceiling"]]
    main = pd.concat([base, main_ext], ignore_index=True)
    with_bounds = {}
    for role in ("lower_bound", "upper_bound"):
        extra = ext[ext["role"] == role][["gene", "stratum", "ceiling"]]
        if len(extra):
            with_bounds[role] = stratum_summary(pd.concat([main, extra], ignore_index=True))
    return {
        "main": stratum_summary(main),
        "v5_three_genes": stratum_summary(base),
        "external_main_only": stratum_summary(main_ext),
        "main_plus_bounded_deposit": with_bounds,
        "bounded_rows": ext[ext["role"].isin(["lower_bound", "upper_bound"])][
            ["gene", "stratum", "n", "rep_rho", "k", "reliability", "ceiling", "role"]
        ].to_dict("records"),
        "excluded_rows": ext[ext["role"] == "excluded"][
            ["gene", "stratum", "status"]].to_dict("records"),
        "not_estimated": NOT_ESTIMATED,
    }


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def licences(raw: Path = RAW) -> dict[str, dict]:
    """The licence of each deposit, as its MaveDB record states it."""
    out = {}
    for urn in EXTERNAL:
        meta = raw / ("urn_mavedb_" + urn.split(":")[-1]) / "metadata.json"
        lic = json.loads(meta.read_text(encoding="utf-8")).get("license") or {}
        out[urn] = {k: lic.get(k) for k in ("shortName", "longName", "version", "link")}
    return out


def register_licences(raw: Path = RAW) -> int:
    """Add each deposit's licence to its entries in raw/MANIFEST.json.

    `atlas.fetch_mavedb` records the URN, gene and title only. The hash is
    recomputed by `register_file`, so a file that changed since it was fetched
    is recorded as it is now. Run `verify_manifest` first if that matters.
    """
    manifest_path = raw / "MANIFEST.json"
    entries = load_manifest(manifest_path).get("files", {})
    lic = licences(raw)
    n = 0
    for key, entry in sorted(entries.items()):
        urn = entry.get("urn")
        if urn not in lic:
            continue
        meta = {k: v for k, v in entry.items()
                if k not in ("sha256", "bytes", "registered_at")}
        meta["licence"] = lic[urn]
        register_file(manifest_path, raw / key, **meta)
        n += 1
    return n


def inputs(raw: Path = RAW) -> dict:
    files = {}
    for key, entry in sorted(load_manifest(raw / "MANIFEST.json").get("files", {}).items()):
        files[key] = {"sha256": entry["sha256"], "bytes": entry["bytes"],
                      "urn": entry.get("urn"), "gene": entry.get("gene")}
    return {"raw_dir": raw.relative_to(REPO).as_posix() if raw.is_relative_to(REPO) else str(raw),
            "files": files, "licences": licences(raw),
            "v5_table": V5_TABLE.relative_to(REPO).as_posix(),
            "v5_table_sha256": sha256_file(V5_TABLE) if V5_TABLE.exists() else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--v5", default=str(V5_TABLE))
    ap.add_argument("--out", default="results")
    ap.add_argument("--register-licences", action="store_true",
                    help="only add the deposits' licences to raw/MANIFEST.json")
    args = ap.parse_args(argv)
    raw = Path(args.raw).resolve()

    if args.register_licences:
        n = register_licences(raw)
        print(f"[external] licence added to {n} manifest entries in {raw / 'MANIFEST.json'}")
        return 0

    ext, checks = external_table(raw)
    v5 = pd.read_csv(args.v5, sep="\t")
    summary = summarise(ext, v5)
    summary = {"note": ("Ceilings are sqrt(reliability). 'main' combines the three v5 genes "
                        "(results/reliability_v1.tsv, v5's primary route, not recomputed) "
                        "with the external rows whose role is 'main'. Rows with role "
                        "lower_bound / upper_bound do not enter 'main'."),
               **summary, "checks": checks, "inputs": inputs(raw)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ext.to_csv(out / "external_reliability_v1.tsv", sep="\t", index=False)
    (out / "external_reliability_v1.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n")

    show = pd.DataFrame(summary["main"])
    print("Attenuation ceiling by stratum, main estimate (v5 genes + external 'main' rows)")
    print(show.drop(columns="genes").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if summary["bounded_rows"]:
        b = pd.DataFrame(summary["bounded_rows"])
        print("\nBounded rows (not in the main estimate):")
        print(b.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
