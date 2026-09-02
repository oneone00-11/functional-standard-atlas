#!/usr/bin/env python3
"""Assemble supplementary figures S1-S18 from pipeline PNGs in results/.

S1-S12 : per-predictor forest-plot sheets (montage of results/forest_<model>_*.png)
S13    : concordance-scatter sheet (results/concordance_*.png)
S14-S18: single existing figures renumbered for the supplement.

Only the canonical stratum panels are montaged for each predictor. AlphaGenome
has an extra splice-restricted panel set (forest_alphagenome_splice_*) which is
omitted here; the files remain in results/ and the Zenodo archive.

Usage (PYTHONPATH=src):  python figures/assemble_supp_figures.py
Outputs: results/supp_fig_S01.png ... results/supp_fig_S18.png
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

# (sheet number, results/ file prefix, display label)
MODELS = [
    (1, "alphagenome", "AlphaGenome"),
    (2, "spliceai_ds", "SpliceAI"),
    (3, "pangolin_score", "Pangolin"),
    (4, "cadd", "CADD"),
    (5, "alphamissense", "AlphaMissense"),
    (6, "evo2", "Evo2-7B"),
    (7, "gpn_msa", "GPN-MSA"),
    (8, "nucleotide_transformer", "NT-v2-500M"),
    (9, "phylop100way", "phyloP-100way"),
    (10, "phastcons100way", "phastCons-100way"),
    (11, "gnomad_af_global", "gnomAD AF (global)"),
    (12, "gnomad_af_popmax", "gnomAD AF (popmax)"),
]

STRATUM_ORDER = [
    ("all", "All variants"),
    ("coding_or_utr", "Coding / UTR"),
    ("clinvar_recorded", "ClinVar-recorded"),
    ("splice_region", "Splice region"),
    ("splice_1_2", "Splice ±1–2"),
    ("splice_3_10", "Splice 3–10 bp"),
    ("splice_11_50", "Splice 11–50 bp"),
    ("splice_deep", "Splice >50 bp"),
]

SINGLES: dict[int, str] = {}  # S14/S15 are shipped by atlas.supplement (DEMOTED_FIGS)

CELL_W = 520          # px, thumbnail width per panel
COLS = 3
TITLE_H = 70
LABEL_H = 34
PAD = 12


def font(size: int) -> ImageFont.FreeTypeFont:
    for cand in ("/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def montage(panels: list[tuple[Path, str]], title: str, out: Path) -> None:
    thumbs: list[tuple[Image.Image, str]] = []
    for path, label in panels:
        im = Image.open(path).convert("RGB")
        h = round(im.height * CELL_W / im.width)
        thumbs.append((im.resize((CELL_W, h), Image.LANCZOS), label))
    cell_h = max(im.height for im, _ in thumbs) + LABEL_H
    rows = (len(thumbs) + COLS - 1) // COLS
    W = COLS * CELL_W + (COLS + 1) * PAD
    H = TITLE_H + rows * cell_h + (rows + 1) * PAD
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    tf, lf = font(30), font(20)
    d.text((PAD, 18), title, fill="black", font=tf)
    for i, (im, label) in enumerate(thumbs):
        r, c = divmod(i, COLS)
        x = PAD + c * (CELL_W + PAD)
        y = TITLE_H + PAD + r * (cell_h + PAD)
        d.text((x + 4, y), label, fill="black", font=lf)
        sheet.paste(im, (x, y + LABEL_H))
    sheet.save(out, dpi=(150, 150))
    # PNG is not among the figure formats submission systems generally accept
    # (GIF, TIFF, EPS, PDF, JPEG are), so every sheet is written as PDF too.
    sheet.convert("RGB").save(out.with_suffix(".pdf"), "PDF", resolution=150.0)
    print(f"  {out.name}: {len(thumbs)} panels, {W}x{H} (+ PDF)")


def main() -> None:
    for num, prefix, label in MODELS:
        panels = []
        for suffix, disp in STRATUM_ORDER:
            p = RESULTS / f"forest_{prefix}_{suffix}.png"
            if p.exists():
                panels.append((p, disp))
        if not panels:
            raise SystemExit(f"no forest panels found for {prefix}")
        montage(panels, f"Supplemental Fig. S{num} — {label}: per-gene correlation "
                        f"with functional score, by stratum",
                RESULTS / f"supp_fig_S{num:02d}.png")

    conc = sorted(RESULTS.glob("concordance_*.png"))
    montage([(p, p.stem.replace("concordance_", "")) for p in conc],
            "Supplemental Fig. S13 — Concordance with the companion benchmark, per ported score",
            RESULTS / "supp_fig_S13.png")

    for num, src in SINGLES.items():
        shutil.copyfile(RESULTS / src, RESULTS / f"supp_fig_S{num}.png")
        print(f"  supp_fig_S{num}.png <- {src}")


if __name__ == "__main__":
    main()
