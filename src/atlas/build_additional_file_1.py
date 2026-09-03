"""Build Additional file 1 from results/, instead of assembling it by hand.

The delivered workbook was hand-assembled, which is why it drifted: Table S11
was added to the repository and never reached the file the manuscript points at,
and nothing could notice because nothing tied the workbook to anything.

Each sheet is exactly one output file. That mapping is the whole design: if a
sheet cannot name its source, it does not belong in the workbook.

Usage (PYTHONPATH=src):
    python -m atlas.build_additional_file_1 --build <out.xlsx>
    python -m atlas.build_additional_file_1 --compare <delivered.xlsx>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
DECIMALS = 4   # as delivered

# sheet name -> the single results/ file it renders
SHEETS: dict[str, str] = {
    "S1":   "tableS3_per_gene_rho.tsv",
    "S2":   "eval_ext_v1.tsv",
    "S3":   "attenuation_v1.tsv",
    "S4":   "attenuation_simulation_v1.tsv",
    "S5":   "tie_audit_v1.tsv",
    "S5b":  "tie_audit_rounded_vs_fullprec_v1.tsv",
    "S6":   "head_to_head_v1.tsv",
    "S7":   "ensemble_logo_v1.tsv",
    "S8":   "definition_sweep_performance_v1.tsv",
    "S9":   "table2_model_inventory.tsv",
    "S10":  "fdr_grid_v1.tsv",
    "S10b": "fdr_head_to_head_v1.tsv",
    "S11":  "table_s11_training_data_v1.tsv",
    "S12":  "table_s12_quoted_summary_v1.tsv",
}


def _rows(path: Path) -> list[list[object]]:
    with path.open(newline="") as fh:
        raw = list(csv.reader(fh, delimiter="\t"))
    out: list[list[object]] = []
    for i, row in enumerate(raw):
        if i == 0:
            out.append(list(row))
            continue
        conv: list[object] = []
        for cell in row:
            s = cell.strip()
            if s == "":
                conv.append(None)
                continue
            try:
                if s.lstrip("+-").isdigit():
                    conv.append(int(s))
                else:
                    # The delivered workbook carries four decimal places. That is
                    # a presentation choice made when it was assembled by hand;
                    # the generator reproduces it rather than silently widening
                    # every published figure.
                    conv.append(round(float(s), DECIMALS))
            except ValueError:
                conv.append(s)
        conv.append  # noqa: B018 - keep mypy quiet about the branch above
        out.append(conv)
    return out


def build(dest: Path, results: Path = RESULTS) -> tuple[Path, int]:
    import openpyxl

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    n = 0
    for sheet, rel in SHEETS.items():
        src = results / rel
        if not src.exists():
            sys.exit(f"[af1] missing source for sheet {sheet}: {src}")
        ws = wb.create_sheet(sheet)
        for row in _rows(src):
            ws.append(row)
        ws.freeze_panes = "A2"
        n += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    wb.save(dest)
    return dest, n


def compare(delivered: Path, results: Path = RESULTS) -> dict:
    """Cell-by-cell against what results/ would produce right now."""
    import openpyxl

    wb = openpyxl.load_workbook(delivered, read_only=True, data_only=True)
    diffs: list[str] = []
    delivered_sheets = list(wb.sheetnames)
    missing_sheets = [s for s in SHEETS if s not in delivered_sheets]
    extra_sheets = [s for s in delivered_sheets if s not in SHEETS]

    for sheet, rel in SHEETS.items():
        if sheet not in delivered_sheets:
            continue
        want = _rows(results / rel)
        got = [list(r) for r in wb[sheet].iter_rows(values_only=True)]
        if len(want) != len(got):
            diffs.append(f"{sheet}: {len(got)} rows delivered, {len(want)} from results/")
            continue
        for i, (a, b) in enumerate(zip(want, got), start=1):
            if len(a) != len(b):
                diffs.append(f"{sheet} row {i}: {len(b)} columns delivered, {len(a)} expected")
                continue
            for j, (x, y) in enumerate(zip(a, b), start=1):
                if _same(x, y):
                    continue
                diffs.append(f"{sheet} r{i}c{j}: delivered {y!r}, results/ {x!r}")
    return {"missing_sheets": missing_sheets, "extra_sheets": extra_sheets,
            "cell_differences": diffs}


def _same(x: object, y: object) -> bool:
    if x is None and y is None:
        return True
    if isinstance(x, float) and isinstance(y, (int, float)):
        return abs(x - float(y)) <= 1e-9 * max(1.0, abs(x))
    if isinstance(x, int) and isinstance(y, (int, float)):
        return float(x) == float(y)
    return str(x).strip() == str(y).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", metavar="XLSX")
    g.add_argument("--compare", metavar="XLSX")
    a = ap.parse_args()
    if a.build:
        dest, n = build(Path(a.build).expanduser())
        print(f"[af1] wrote {dest} with {n} sheets: {', '.join(SHEETS)}")
        return 0
    r = compare(Path(a.compare).expanduser())
    print(f"[af1] comparing {a.compare} against results/")
    for s in r["missing_sheets"]:
        print(f"  SHEET MISSING FROM THE DELIVERED FILE: {s} ({SHEETS[s]})")
    for s in r["extra_sheets"]:
        print(f"  SHEET IN THE DELIVERED FILE WITH NO SOURCE: {s}")
    if r["cell_differences"]:
        print(f"  {len(r['cell_differences'])} cell differences; first 15:")
        for d in r["cell_differences"][:15]:
            print(f"    {d}")
    elif not r["missing_sheets"] and not r["extra_sheets"]:
        print("  identical to what results/ produces")
    return 1 if (r["missing_sheets"] or r["extra_sheets"] or r["cell_differences"]) else 0


if __name__ == "__main__":
    sys.exit(main())
