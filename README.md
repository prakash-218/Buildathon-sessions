# Buildathon: Shopping Sessions

Our entry for the Builders Buildathon Databricks track: turn messy click-stream CSVs into two exact daily Parquet tables, as fast as possible. The full spec is in [rules.md](rules.md).

## Layout

| Path | What |
| --- | --- |
| `Dockerfile` | What the scorer builds: one static Rust binary in a `FROM scratch` image (about 3 MB). |
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
2. Links: IDs are numbered in 256 hash shards in parallel, union-find runs over the numbers, and each person is named by its byte-wise smallest ID.
   A 2 MB bit filter marks the IDs that are not their person's name, so most rows skip the ID -> name maps.
3. Events: every copy is parsed and validated in parallel. A valid copy's 24-byte payload is written once, straight into its person's shard; every copy also sends (`event_id`, `ingest_ts`, payload location) to its `event_id` shard. Dedup keeps the copy with the smallest `ingest_ts` per ID and marks every other copy's payload dead in place (each payload belongs to one `event_id` shard, so no two threads write it). Canonical `p<digits>` product IDs are looked up in a small array.
4. Clicks: validated and sharded by person. No dedup: copies are identical, so they cannot change a credit or an assist.
5. Per person (dead payloads skipped): sort, split sessions on gaps over 30 min, credit each session with a two-pointer over the clicks, and count assists by merging each campaign's `[t, t + 7 days]` windows into intervals.
6. Write the tables as Parquet, one file per thread, without dictionaries or statistics. Then hand the big buffers and input mappings back with `madvise(MADV_DONTNEED)` on all threads and exit, since the kernel's teardown at exit is single-threaded.

The scanner prefetches its input 1 KB ahead.

Memory comes from mimalloc with `MIMALLOC_PURGE_DELAY=-1` (set in the Dockerfile), and each worker reuses its hash maps and buffers across shards, so later phases reuse memory instead of faulting in fresh pages.

## Status

| Check | Result |
| --- | --- |
| Fixtures, on each key type (`u64`, `u128`, slices) | 36/36 |
| Dev set (seed 1, 1M events) vs `dev-answer/` | exact match |
| Seeds 2, 3 (1M events) and 11 (5M events) vs the reference | exact match |
| 20M events (seed 7), local laptop, 8 CPUs | 4.3 to 4.8 s in the container; about 16k page faults |

The track's baseline is 171.91 s. Practice board: a6ee49b (first version) 4.09 s; 581d3f3 (AVX2 parser, packed keys) 2.07 s; f1a698d (parallel links, mimalloc, buffer reuse) 1.64 s; a17ca33 (write-once events, renamed-ID filter) 1.51 s; a50022a (SWAR timestamps) 1.41 s.

Local wall times on a laptop under WSL2 swing by ±30%, so page faults (`perf stat -e page-faults`, deterministic) and profiles are better guides than single timings.

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
