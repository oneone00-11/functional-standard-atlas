# Model directory template

Copy this folder to `models/<model_name>/` and implement the contract below.
Do not edit this template in place.

## Contract

- `score.py` is the ONLY entry point. It must be runnable as:

  ```bash
  python score.py --input <variant_table.parquet> --output scores.parquet
  ```

- **Input** parquet columns: `variant_id, chrom, pos, ref, alt, transcript, hgvs_c`
  (GRCh38 coordinates; `hgvs_c` present when the variant has a transcript mapping).
- **Output** parquet columns: all input columns PLUS exactly one score column named
  after the model (e.g. `alphagenome`). Larger must mean more damaging. If the raw
  model output is oriented the other way, flip it here and record the flip in this
  model's `NOTES.md`.
- Variants the model cannot score must be DROPPED from the output (coverage is
  measured downstream), never imputed.

## Files you must create

- `score.py` — the scorer (start from the provided skeleton).
- `requirements.txt` — exact pinned versions (`pkg==x.y.z`). No unpinned git URLs.
- `NOTES.md` — model version/checkpoint, date scored, orientation rule, quirks.
- `cache/` — cached outputs; cache key MUST include the SHA-256 of the input file.

## Rules (see ../../AGENTS.md)

- Model versions, checkpoint IDs and API identifiers must be obtained from the
  source (API response, model card file) and written into `NOTES.md` by the script
  or by you from that output — never from an AI assistant's memory.
- If scoring uses a remote API, record the request date and the API's version
  string in `NOTES.md`.
