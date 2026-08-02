"""Assemble the atlas score matrix: frozen variants × all model score columns.

Left-joins every scored model parquet onto the frozen matrix by variant_id.
Score files are the single source of truth (AGENTS.md); this script only
merges, never recomputes.

Usage (PYTHONPATH=src):
    python -m atlas.assemble --out results/score_matrix_atlas_v1.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "data" / "frozen" / "frozen-matrix-v1.parquet"

# (score column, parquet under results/) — every model scorer writes one file
# per score column, keyed by variant_id.
SCORE_FILES = {
    "alphagenome": "alphagenome_scores.parquet",
    "cadd": "cadd_scores.parquet",
    "alphamissense": "alphamissense_scores.parquet",
    "evo2": "evo2_scores.parquet",
    "gpn_msa": "gpn_msa_scores.parquet",
    "nucleotide_transformer": "nt_scores.parquet",
    "phylop100way": "phylop100way_scores.parquet",
    "phastcons100way": "phastcons100way_scores.parquet",
    "gnomad_af_global": "gnomad_af_global_scores.parquet",
    "gnomad_af_popmax": "gnomad_af_popmax_scores.parquet",
    "spliceai_ds": "spliceai_ds_scores.parquet",
    "pangolin_score": "pangolin_scores.parquet",
}

BASE_COLS = ["variant_id", "gene", "urn", "chrom", "pos", "ref", "alt",
             "transcript", "hgvs_c", "functional_pathogenicity", "mapping_status"]


def assemble(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    mat = pd.read_parquet(FROZEN)
    coverage: dict[str, int] = {}
    for col, fname in SCORE_FILES.items():
        path = results_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"missing score file: {path} — run the scorer first")
        sc = pd.read_parquet(path, columns=["variant_id", col])
        dup = sc["variant_id"].duplicated().sum()
        if dup:
            raise ValueError(f"{fname}: {dup} duplicated variant_id rows")
        mat = mat.merge(sc, on="variant_id", how="left")
        coverage[col] = int(mat[col].notna().sum())
    return mat, coverage


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    args = ap.parse_args(argv)

    mat, coverage = assemble(Path(args.results_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mat.to_parquet(out, index=False)

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    summary = {
        "matrix": str(out),
        "sha256": sha,
        "n_variants": int(len(mat)),
        "base_columns": BASE_COLS,
        "score_columns": list(SCORE_FILES),
        "coverage": coverage,
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
