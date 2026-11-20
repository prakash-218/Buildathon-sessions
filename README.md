# Buildathon: Shopping Sessions

Our entry for the Builders Buildathon Databricks track: turn messy click-stream CSVs into two exact daily Parquet tables, as fast as possible. The full spec is in [rules.md](rules.md).

## Layout

| Path | What |
| --- | --- |
| `rules.md` | The track's rules. Where anything disagrees with it, it wins. |
| `reference/reference.py` | Slow, literal Python implementation of every rule. A test oracle, not the submission. |
| `fixtures/` | The 36 hand-written edge-case fixtures from the starter. |
| `scripts/` | `run_fixtures.py` (run a command on every fixture) and `compare.py` (exact diff of two outputs). |
| `dev-answer/` | The published correct output for the dev set (seed 1, 1M events). |
| `generator/` | The dataset generator (Rust source) from the track page. |

The fast pipeline and the top-level `Dockerfile` the scorer builds are still to come.

## Status

`reference/reference.py` passes 36/36 fixtures and matches `dev-answer/` exactly (about 100 s on 1M events).

## Testing

```sh
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # Windows: .venv\Scripts\...

# Edge-case fixtures
python scripts/run_fixtures.py fixtures -- python reference/reference.py

# Dev set: build the generator, make the data, run, compare
docker run --rm -v "$PWD:/work" -w /work/generator rust:1.85 sh -c \
  'cargo build --release && ./target/release/generate --seed 1 --events 1000000 --clicks 250000 --output /work/dev'
python reference/reference.py dev out-ref
python scripts/compare.py out-ref dev-answer
```

Any other `--seed` makes a new dataset; diff a fast pipeline's output against the reference's on several seeds before submitting. Datasets and outputs (`dev/`, `out-*/`) are git-ignored: the repo must stay under 50 MB.
