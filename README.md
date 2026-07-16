# Sen.jl

[![CI](https://github.com/53n90ku/Sen.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/53n90ku/Sen.jl/actions/workflows/ci.yml)
[![Julia 1.12](https://img.shields.io/badge/Julia-1.12-9558B2.svg)](https://julialang.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Sen is a vector search engine written in Julia.

I started this because I wanted to see how far I could take a vector search engine in Julia without just wrapping another engine. At this point it is not only a nearest-neighbor demo anymore. Sen can store vectors with metadata, run exact and IVF search, filter results, handle mutations, save everything to disk and recover it after a crash.

The main use case right now is an embedded search engine for a Julia project. You give Sen embeddings from Ollama, an API or any model you want, and Sen handles the part after that: indexing, filtering, searching and persistence.

Sen is currently at `v0.1.0`. I am keeping the claims narrow on purpose: it is a single-node embedded engine, not a distributed vector database.

## What works right now

- Exact and IVF approximate top-k search
- Cosine similarity and dot product
- Metadata filters with `Eq`, `In`, `Range`, `And`, `Or` and `Not`
- Automatic planning between exact, IVF and filter-aware strategies
- Batch insert, upsert, delete and search
- User-provided document IDs
- Insert, upsert, update and delete without making new writes invisible
- WAL-backed durable mutations and snapshot persistence
- Crash recovery and snapshot fallback
- Memory-mapped vector loading for larger saved databases
- Background indexing, rebuilding and compaction
- A stable public API for the `v0.1` line

## Install

Sen is not registered yet, so install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/53n90ku/Sen.jl")
```

Sen currently supports Julia 1.12.

## A small example

```julia
using Sen

db = create_db("my-vectors"; dim=3, metric=:cosine)

vectors = Float32[
    1 0 0
    0 1 0
    0 0 1
]

metadata = [
    (title="Julia internals", topic="julia", year=2025),
    (title="Vector databases", topic="search", year=2024),
    (title="Building an index", topic="search", year=2025),
]

insert!(db, vectors, metadata; ids=["julia", "databases", "indexing"])
build!(db; nlists=2)

hits = search(
    db,
    Float32[0, 1, 0];
    k=2,
    filter=And(Eq(:topic, "search"), Range(:year, 2024, 2025)),
)

for hit in hits
    println("$(hit.id): $(hit.metadata.title) [score=$(hit.score)]")
end

save!(db)
close(db)
```

The database can be opened again without rebuilding everything:

```julia
db = load_db("my-vectors"; mmap_vectors=:auto)
hits = search(db, Float32[0, 1, 0]; k=2)
close(db)
```

If an interrupted write or damaged current snapshot needs recovery, use `recover_db` instead of `load_db`.

## Semantic search

Sen stores and searches vectors; it does not decide how text becomes a vector. The repository includes a complete semantic-search example using Ollama so the boundary is clear.

```bash
git clone https://github.com/53n90ku/Sen.jl
cd Sen.jl

ollama pull all-minilm
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/semantic_search.jl \
  "How do scientists spot planets orbiting distant stars?"
```

That query finds the document about exoplanet detection even though the query and document use different wording. The example embeds the documents, builds a Sen database, saves it, reopens it with memory mapping and runs IVF search.

The example also has a deterministic test that does not need a running Ollama server:

```bash
julia --project=examples examples/test_semantic_search.jl
```

## What happens during a search

Before an index is built, Sen can search the stored vectors exactly. After `build!`, the query planner can choose exact search, IVF, pre-filtering, post-filtering or a filter-aware path based on the database and filter.

New mutations are kept searchable while the main index is being rebuilt. They are not hidden until the next manual rebuild. Maintenance can publish a newer index in the background, and the amount of unindexed work searched per query is bounded by configuration.

For durable databases, acknowledged mutations go through a write-ahead log. `save!` publishes a snapshot, and `load_db` replays committed WAL records newer than that snapshot. Sen also uses a single-writer lock so two processes cannot silently write over the same database.

## Tests and the engine contract

I did not want “real vector search engine” to be a vague label, so the repository has an [engine contract](engine_contract.toml). It lists the behavior that must pass before Sen makes that claim: search primitives, durable writes, mutation visibility, safe vector validation, writable reopen, atomic mutations, continuous indexing, bounded search cost, recall and latency gates, crash recovery and full CI.

Run the complete package suite with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run the engine and crash-recovery gates directly with:

```bash
julia --project=. scripts/validate_engine_contract.jl --enforce
julia --project=. scripts/test_crash_recovery.jl
```

CI runs the full suite on Linux and macOS with Julia 1.12. It also checks formatting, the semantic-search example, release metadata and the frozen benchmark quality contract.

## Benchmarks

Sen has a separate benchmark package under `benchmark/SenBench`. The release workloads use held-out queries and freeze their dataset fingerprints, target recall and latency ceiling in [quality_contract.toml](quality_contract.toml).

The current `v0.1` snapshot was measured on an Apple M2 using Julia 1.12.6:

| Workload | Strategy | Recall@10 | p95 latency |
|---|---|---:|---:|
| Cosine, unfiltered | IVF | 1.0000 | 0.006 ms |
| Cosine, filtered | Filter-aware | 1.0000 | 0.021 ms |
| Cosine, filtered | Bound filter-aware | 0.9958 | 0.025 ms |
| Dot product, unfiltered | IVF | 1.0000 | 0.006 ms |

These are small frozen release workloads with 1,024 vectors, not a claim that every machine or real dataset will have the same latency. The complete environment, methods and results are in [BENCHMARKS.md](BENCHMARKS.md).

Reproduce the release benchmark with:

```bash
julia --project=benchmark/SenBench -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/SenBench scripts/benchmark_release.jl
```

There are also external benchmark drivers for Faiss and Qdrant under `benchmark/external`.

## Repository layout

```text
src/                     engine implementation
test/                    package tests grouped by subsystem
examples/                semantic-search example
benchmark/SenBench/      reproducible benchmark package
benchmark/external/      Faiss and Qdrant comparison drivers
scripts/                 contract, recovery, formatting and release checks
engine_contract.toml     behavior required for the engine claim
quality_contract.toml    frozen recall and latency workloads
```

## Current limits

I would use Sen today for an embedded Julia project, a local semantic-search feature or as a base for experimenting with filtered vector search. I would not pretend it is already a replacement for a distributed production service.

The current limits are:

- One writable process per database
- Single-node and embedded only
- IVF indexing; no HNSW or product quantization yet
- No built-in embedding model or HTTP server
- Julia 1.12 is the only version tested in CI right now
- The API is stable for `v0.1`, but the storage format is still young

If your project fits inside those limits, the engine is usable now. If it does not, the limits should at least be obvious before you build around it.

## Contributing

Issues and small focused pull requests are welcome. If you are changing search behavior, persistence or the planner, please add the test that proves the new behavior and run the engine contract before opening the PR.

## License

Sen is available under the [MIT License](LICENSE).
