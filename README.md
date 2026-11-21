# Buildathon: Shopping Sessions

Our entry for the Builders Buildathon Databricks track: turn messy click-stream CSVs into two exact daily Parquet tables, as fast as possible. The full spec is in [rules.md](rules.md).

## Layout

| Path | What |
| --- | --- |
| `Dockerfile` | What the scorer builds: the Rust pipeline on `debian:bookworm-slim`. |
| `pipeline/` | The fast pipeline (Rust, rayon, parquet). |
| `reference/reference.py` | Slow, literal Python implementation of every rule. The test oracle. |
| `rules.md` | The track's rules. Where anything disagrees with it, it wins. |
| `fixtures/` | The 36 hand-written edge-case fixtures from the starter. |
| `scripts/` | `run_fixtures.py` (run a command on every fixture), `compare.py` (exact diff of two outputs), `docker_run.py` (run the image with the scorer's container flags). |
| `dev-answer/` | The published correct output for the dev set (seed 1, 1M events). |
| `generator/` | The dataset generator (Rust source) from the track page. |

## How the pipeline works

1. mmap every input file and cut them all into line-aligned chunks, so 8 threads share the parsing however the file sizes are skewed. Each chunk is scanned once with AVX2: 64 bytes at a time become newline, comma and quote bitmasks, and the parser walks the bits.
   IDs are packed into integers that keep byte-wise order: `u64` (up to 9 ASCII bytes, which covers every generated ID), else `u128` (up to 15 bytes), else the run is redone with byte-slice keys. `KEYS=128` or `KEYS=slice` forces a fallback, for testing.
2. Links: union-find over the IDs; each person is named by its byte-wise smallest ID.
3. Events: every copy is parsed and validated in parallel, sharded by `event_id`. Each shard keeps the copy with the smallest `ingest_ts`, and that copy's validity decides. Survivors are resharded by person.
4. Clicks: validated and sharded by person. No dedup: copies are identical, so they cannot change a credit or an assist.
5. Per person: sort, split sessions on gaps over 30 min, credit each session with a two-pointer over the clicks, and count assists by merging each campaign's `[t, t + 7 days]` windows into intervals.
6. Write the tables as Parquet, one file per thread.

## Status

| Check | Result |
| --- | --- |
| Fixtures, on each key type (`u64`, `u128`, slices) | 36/36 |
| Dev set (seed 1, 1M events) vs `dev-answer/` | exact match |
| Seeds 2, 3 (1M events) and 11 (5M events) vs the reference | exact match |
| 20M events (seed 7), local laptop, 8 CPUs | 5.8 to 8.4 s in the container (too noisy to rank changes) |

The track's baseline is 171.91 s. Commit a6ee49b (before the AVX2 parser and packed keys) scored 4.09 s on the practice board.

## Testing

Needs Docker and Python 3.10+ with `pip install -r requirements-dev.txt`.

```sh
docker build --platform linux/amd64 -t shopping-sessions:latest .

# Edge-case fixtures through the image, with the scorer's flags
python scripts/run_fixtures.py fixtures -- python scripts/docker_run.py

# Dev set: build the generator, make the data, run, compare
docker run --rm -v "$PWD:/work" -w /work/generator rust:1.85 sh -c \
  'cargo build --release && ./target/release/generate --seed 1 --events 1000000 --clicks 250000 --output /work/dev'
python scripts/docker_run.py dev out-dev
python scripts/compare.py out-dev dev-answer

# Any other seed: diff against the reference
python reference/reference.py dev-s2 ref-s2
python scripts/compare.py out-s2 ref-s2
```

Set `MEM=12g` if Docker has less than 24 GB, and `CPUSET=0-7` to pin CPUs as the scorer does. On Docker Desktop, host folders are slow to read, so time runs on a Docker volume: `docker_run.py vol:<volume>/<path> vol:<volume>/<out>`. Datasets and outputs (`dev*/`, `out-*/`, `ref-*/`) are git-ignored: the repo must stay under 50 MB.
