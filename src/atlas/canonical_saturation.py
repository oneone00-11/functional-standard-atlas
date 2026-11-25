"""How saturated the canonical splice sites are: the +-1-2 bp SNVs of the twelve genes.

One row per gene (the seven core genes of score matrix v3, the five external genes of
results/external_scores_v2.parquet), on its SNVs at +-1-2 bp that carry a functional
score: n; the medians of ``spliceai_ds`` and ``pangolin_score`` and how many are at least
0.2 and at least 0.5; the functional score's quartiles; and, for a deposit that carries
its own functional classification, how many of these variants it classes abnormal.

The deposit's own classes: the record's investigator-provided classes given as score
ranges (atlas.external_scores.deposit_classes; BRCA1, BARD1, PALB2, the external genes),
or given by class with the classes in a column of the score table (RAD51C's
functional_classification, mapped by the record). BAP1 and VHL give their classes by
class without assigning them in the score table, and BRCA2 carries none: they have no
count.

Output: results/canonical_site_saturation_v1.tsv.

Usage (PYTHONPATH=src):  python -m atlas.canonical_saturation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import paths, population

REPO = paths.REPO
RESULTS = paths.RESULTS
RAW = REPO / "data" / "raw"
STRATUM = "splice_1_2"
CORE_DEPOSITS = {"BRCA1": "urn:mavedb:00000097-0-2", "BAP1": "urn:mavedb:00000662-0-1",
                 "RAD51C": "urn:mavedb:00000673-0-1", "VHL": "urn:mavedb:00000675-a-1",
                 "BRCA2": "urn:mavedb:00001225-a-1", "BARD1": "urn:mavedb:00001250-a-2",
                 "PALB2": "urn:mavedb:00001259-a-2"}
THRESHOLDS = (0.2, 0.5)
COLUMNS = ["group", "gene", "n",
           "spliceai_ds_n", "spliceai_ds_median", "spliceai_ds_n_ge_0.2", "spliceai_ds_n_ge_0.5",
           "pangolin_score_n", "pangolin_score_median", "pangolin_score_n_ge_0.2",
           "pangolin_score_n_ge_0.5", "functional_q25", "functional_q50", "functional_q75",
           "classes_source", "n_classified", "n_abnormal"]


def _dir(urn: str, external: bool) -> Path:
    return RAW / ("mavedb_external" if external else "mavedb") / ("urn_mavedb_" + urn.split(":")[-1])


def own_classes(d: Path) -> tuple[str, pd.Series | None]:
    """(source, accession -> the deposit's class), or (why there is none, None)."""
    from atlas.external_scores import deposit_classes
    meta = json.loads((d / "metadata.json").read_text())
    dep = pd.read_csv(d / "scores.csv", low_memory=False)
    cal = [c for c in meta.get("scoreCalibrations") or []
           if c.get("investigatorProvided") and c.get("title", "").startswith("Investigator-provided")]
    if cal and all("range" in fc for fc in cal[0]["functionalClassifications"]):
        src, cls = deposit_classes(meta, dep)
        return src, pd.Series(cls.to_numpy(), index=dep["accession"])
    if cal:
        by_class = {str(fc.get("class", "")).lower(): fc["functionalClassification"]
                    for fc in cal[0]["functionalClassifications"]}
        col = next((c for c in dep.columns if c.lower() == "functional_classification"), None)
        if col is None:
            return ("the record's classes are given by class and not assigned in the score "
                    "table", None)
        cls = dep[col].astype(str).str.lower().map(by_class).fillna("unclassified")
        return (f"scoreCalibrations: {cal[0]['title']} (by class, column {col})",
                pd.Series(cls.to_numpy(), index=dep["accession"]))
    if "functional_consequence" in dep.columns:
        src, cls = deposit_classes(meta, dep)
        return src, pd.Series(cls.to_numpy(), index=dep["accession"])
    return "the deposit carries no functional classification", None


def _row(group: str, gene: str, s: pd.DataFrame, ids: pd.Series, classes) -> dict:
    rec = {"group": group, "gene": gene, "n": int(len(s))}
    for col in ("spliceai_ds", "pangolin_score"):
        v = s[col].dropna()
        rec[f"{col}_n"] = int(len(v))
        rec[f"{col}_median"] = float(v.median()) if len(v) else np.nan
        for t in THRESHOLDS:
            rec[f"{col}_n_ge_{t}"] = int((v >= t).sum())
    q = s["functional_pathogenicity"].quantile([0.25, 0.5, 0.75])
    rec |= {"functional_q25": float(q[0.25]), "functional_q50": float(q[0.5]),
            "functional_q75": float(q[0.75])}
    src, cls = classes
    rec["classes_source"] = src
    if cls is None:
        rec["n_classified"] = rec["n_abnormal"] = np.nan
    else:
        c = ids.map(cls)
        rec["n_classified"] = int(c.isin(["abnormal", "normal"]).sum())
        rec["n_abnormal"] = int((c == "abnormal").sum())
    return rec


def table() -> pd.DataFrame:
    rows = []
    core = population.load(population.SNV)
    core = core[(core["region"] == STRATUM) & core["functional_pathogenicity"].notna()]
    for gene, urn in sorted(CORE_DEPOSITS.items()):
        s = core[core["gene"] == gene]
        rows.append(_row("core", gene, s, s["variant_id"], own_classes(_dir(urn, False))))
    from atlas.external_scores import GENES, URN
    ext = pd.read_parquet(paths.path("external_scores"))
    ext = ext[(ext["stratum"] == STRATUM) & ext["functional_pathogenicity"].notna()
              & population.is_snv(ext["ref"], ext["alt"])]
    for gene in sorted(GENES):
        s = ext[ext["gene"] == gene]
        rows.append(_row("external", gene, s, s["variant_urn"], own_classes(_dir(URN[gene], True))))
    t = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("n_classified", "n_abnormal"):
        t[c] = t[c].astype("Int64")
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)
    t = table()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / paths.name("canonical_site_saturation"), sep="\t", index=False)
    print(t.drop(columns=["classes_source"]).to_string(index=False,
                                                       float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
