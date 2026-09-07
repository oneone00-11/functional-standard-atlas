# functional-standard-atlas

An attenuation-corrected, territory-resolved benchmark of variant effect
predictors against saturation genome editing (SGE) measurements: 64,178
variants, seven cancer susceptibility genes, nineteen predictors.

Accompanies the manuscript listed in `CITATION.cff`. A separate companion
manuscript on ranking versus calibration of splice-region predictors draws on
an overlapping subset of the same MaveDB assays but estimates no measurement
reliability and applies no attenuation correction, and shares no results
section, figure or table with this work.

## What this repository is

1. **A frozen, content-hashed data asset.** Seven MaveDB SGE score sets mapped
   to GRCh38, orientation-harmonised, frozen into immutable matrices with a
   SHA-256 manifest.
2. **A uniform scoring harness.** One directory per predictor, each with a
   pinned environment, a `score.py` entry point and cached outputs.
3. **An evaluation pipeline.** Per-gene Spearman ρ pooled by DerSimonian–Laird
   random-effects meta-analysis across sixteen strata, with per-stratum
   measurement-reliability estimates, attenuation correction, paired
   dependent-correlation tests and leave-one-gene-out validation.

Every number in the manuscript is read from a file under `results/`; every
figure and table is produced by a committed script.

## Reproduction: what you can run, and what you cannot

**Read this before cloning.** `results/`, `data/raw/`, `data/frozen/`,
`data/refs/` and `models/*/cache/` are gitignored. **A `git clone` gives you
code only and reproduces no figure.** The four tiers below are honest about
where each stops.

| Tier | You have | You can | Wall clock | Disk |
|---|---|---|---|---|
| 1 | git clone | run the 130 code-only guardrails (72 skip without data) | ~25 s | 20 MB |
| 2 | **+ Zenodo archive** | **regenerate every figure and table in the paper** | **< 1 min** | 60 MB |
| 3 | + network | rebuild the frozen matrix from MaveDB | ~20 min | 2.2 GB |
| 4 | + API key + GPU | re-score every predictor from scratch | ~40 h | 6 GB |

**Tier 2 is the one that matters for review.** Download the Zenodo archive
(DOI in `CITATION.cff`), which ships `results/` and `data/frozen/`, then:

```bash
make setup                                     # pinned dependencies, ~2 min
PYTHONPATH=src python figures/atlas_overview.py                # Fig 1
PYTHONPATH=src python figures/hardening_figures.py             # Figs 2, 4, 5, 6 and Supplemental Figs S15, S16
PYTHONPATH=src python figures/mavedb_ceiling_distribution.py   # Fig 3
PYTHONPATH=src python figures/mavedb_ceiling_correlates.py     # Supplemental Fig S17
PYTHONPATH=src python figures/splice_territory_comparison.py   # Supplemental Fig S14
PYTHONPATH=src python figures/reproducibility.py               # Supplemental Fig S13
```

Each script takes under three seconds and writes PNG + PDF into `results/`.
The analysis modules that produce the underlying tables are equally cheap:

```bash
PYTHONPATH=src python -m atlas.reliability          # attenuation ceilings
PYTHONPATH=src python -m atlas.evaluate_ext         # the 19 x 16 grid
PYTHONPATH=src python -m atlas.robustness           # ties, power, paired tests
PYTHONPATH=src python -m atlas.simulate_attenuation # 151,200-trial validation, ~4 min
PYTHONPATH=src python -m atlas.definition_sweep     # reads the cached sweep
```

**Every analysis entry point reads `results/score_matrix_atlas_v2.parquet` by
default** — the 19-predictor matrix the paper is built from — so the commands
above regenerate the delivered tables as-is, with no flags. The older
`score_matrix_atlas_v1.parquet` ships in the archive only as the auditable
pre-rescore matrix (it lacks the seven dbNSFP meta-predictors and carries the
CLI-rounded splice scores; see `atlas.matrix_v2`). Pass
`--matrix results/score_matrix_atlas_v1.parquet` only if you are studying that
earlier state; the output will then differ from the delivered tables by
design.

### What needs credentials, and what a reviewer without them loses

Tier 4 needs two things this repository cannot ship:

- **An AlphaGenome API key** (`ALPHAGENOME_API_KEY`; free for non-commercial
  use). Needed to regenerate the `alphagenome` column and to re-run the
  definition sweep. **A reviewer without a key can still reproduce every
  figure and number**, because the scored columns and the 10,888 cached sweep
  responses are in the archive. What they cannot do is re-derive those scores
  from the API.
- **A GPU** (one NVIDIA A6000, 48 GB) for Evo2-7B and NT-v2-500M. Evo2-7B alone
  took 25 h for 46,392 SNVs. Again, the scored columns are archived; only
  re-derivation needs the hardware.

Everything else — SpliceAI, Pangolin, CADD, AlphaMissense, GPN-MSA, the
conservation tracks, gnomAD and the seven dbNSFP meta-predictors — is
reproducible on CPU with network access.

## Full pipeline (tiers 3–4)

```bash
make setup     # install pinned dependencies  (needs Python >= 3.12)
make fetch     # download registered assays + reference genome, build hash manifest
make freeze    # map to GRCh38 (Mutalyzer + Ensembl), build the frozen matrix
make test      # 202 collected. 179 pass from a clean extract of the release
               # archive; a bare clone runs 130 and skips 72 for want of data.
```

**Python >= 3.12 is required** — numpy 2.5.1 and scipy 1.18.0 both declare it.
`make setup` checks this before installing and stops with an explanation if the
interpreter is older, which matters because a bare `python3` is still 3.9 on
macOS. If your default is older, point it at a newer one:

```bash
make setup PYTHON=python3.12
```

`make fetch` pulls ~2.1 GB of reference sequence, which is why `data/refs/` is
neither committed nor archived.

## Layout

```
config/assays.yaml     assay registry (MaveDB URNs, tiers, notes)
data/raw/              source downloads, hash-tracked          (gitignored; in archive)
data/frozen/           frozen matrices + SHA-256 manifest      (gitignored; in archive)
data/refs/             GRCh38 FASTA + annotation               (gitignored; NOT archived)
models/<name>/         score.py, pinned env, NOTES.md provenance
src/atlas/             fetch, mapping, evaluation, reliability, robustness,
                       clinical evidence, ensembles, simulation, definition sweep
figures/               figure-generating scripts
tests/                 pytest guardrails; must pass before any result is trusted
results/               all pipeline outputs                    (gitignored; in archive)
```

## Adding a predictor or an assay

Copy `models/_template/`, or add the MaveDB URN to `config/assays.yaml` and
re-run `make fetch`. `CONVENTIONS.md` states the rules that keep results trustworthy
— frozen data are immutable, dependencies are pinned exactly, statistics are
read from pipeline outputs rather than transcribed, and every figure comes from
a committed script.

## Author

**Ningyi Zhang** — Department of Biological Sciences, National University of
Singapore · cliffzhang@u.nus.edu ·
[ORCID 0009-0004-3020-4044](https://orcid.org/0009-0004-3020-4044)

Development was AI-assisted (Claude, Anthropic) under the author's direction;
the author designed the study, verified all reported numbers, and takes full
responsibility for the content.

## Licence

MIT for code (`LICENSE`). Data terms are in `LICENSE-DATA`, and they are not
uniform:

- **`data/frozen/frozen-matrix-v1.parquet`** — the assay measurements, no
  predictor columns — is CC BY 4.0 without qualification. The upstream MaveDB
  deposits carry CC0 or CC BY 4.0 terms that pass through.
- **The score matrices under `results/`**, and every figure and table derived
  from them, carry nineteen predictor columns with **mixed terms**. Eight are
  more restrictive than CC BY 4.0 — `alphagenome`, `cadd`, `alphamissense`,
  `nucleotide_transformer`, `spliceai_ds`, `revel`, `primateai`, `vest4` — of
  which `alphamissense` and `nucleotide_transformer` are ShareAlike as well as
  NonCommercial. Three more — `bayesdel_addaf`, `clinpred`, `metarnn` — have no
  licence terms that could be located, which is not the same as permissive.

Per-column licences, each with a source URL, are in
`results/predictor_resources_v1.tsv` — the single source of truth. `results/`
is generated rather than tracked, so that table ships in the archived release
and appears in a checkout after `python -m atlas.predictor_resources`;
`LICENSE-DATA` restates the full summary so it does not depend on the file
being present. Each restriction travels with its column and with anything
derived from it.
