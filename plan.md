# Vortex.jl — Filter-Aware Vector Search Engine

> **Project type:** systems + data infrastructure + information retrieval  
> **Primary language:** Julia  
> **Current project status:** planning / research design  
> **Main goal:** build a vector search engine that is not merely another vector database, but a controlled experimental system for making **filtered vector search** faster under specific, measurable workloads.

---

## 1. Brief project description

**Vortex.jl** is a research-style vector search engine written in Julia. It focuses on **filtered approximate nearest-neighbor search**, where a query is not just:

```text
find the nearest vectors to q
```

but instead:

```text
find the nearest vectors to q
where metadata satisfies some filter
```

Example:

```sql
SEARCH top 10 vectors similar to query_vector
WHERE topic = 'systems'
AND language = 'julia'
AND year >= 2024
```

The project’s central idea is that generic vector search engines often waste work when metadata filters are present. Vortex.jl will use metadata-aware planning and indexing to choose a faster execution strategy depending on filter selectivity.

The project should not claim to be “faster than all vector databases.” A credible research claim is narrower:

> **Vortex.jl is faster than naive pre-filter and post-filter baselines for selected filtered vector search workloads, at comparable Recall@k.**

That is a real systems claim because it is specific, measurable, and falsifiable.

---

## 2. Why this project is resume-worthy

This project gives strong resume signals because it involves:

- vector search internals
- approximate nearest-neighbor indexing
- query planning
- metadata filtering
- bitset operations
- benchmark design
- latency/recall tradeoff analysis
- storage layout
- Julia performance engineering

A weak version of this project would be:

> Built a vector database using cosine similarity.

A strong version is:

> Built a filter-aware vector search engine in Julia with IVF indexing, metadata bitset filters, adaptive query planning, and recall/latency benchmarks against exact, pre-filter, and post-filter baselines.

The second version sounds like a real systems project.

---

## 3. Background and research context

### 3.1 Vector search baseline

A vector search system stores dense vectors and returns vectors closest to a query vector under a similarity metric such as cosine similarity, dot product, or Euclidean distance.

FAISS is one of the mature existing baselines in this space. Its documentation describes it as a library for efficient similarity search and clustering of dense vectors, with support for large vector collections and multiple index types.  
Reference: https://faiss.ai/index.html

### 3.2 Benchmarking culture

Approximate nearest-neighbor search must be evaluated by both speed and quality. ANN-Benchmarks exists specifically to benchmark approximate nearest-neighbor algorithms over datasets and metrics.  
Reference: https://ann-benchmarks.com/index.html  
Reference: https://github.com/erikbern/ann-benchmarks

This project should copy that mindset:

```text
Never report latency alone.
Always report latency at a given recall target.
```

### 3.3 Why filtered vector search matters

Real systems often need vector search with metadata constraints:

```text
tenant_id = company_17
permission = allowed
file_type = pdf
language = english
created_after = 2025-01-01
```

This is commonly called **Filtered Approximate Nearest Neighbor Search**, or FANNS.

Recent research has focused specifically on how filtering changes vector database performance. A 2026 paper, *Filtered Approximate Nearest Neighbor Search in Vector Databases: System Design and Performance Analysis*, evaluates FAISS, Milvus, and pgvector under filtered ANN workloads and highlights that engine-level filtering strategies can dominate raw index performance.  
Reference: https://arxiv.org/abs/2602.11443

Another 2025 survey frames filtered ANN over vector-scalar hybrid data as an important and still-developing research area.  
Reference: https://arxiv.org/abs/2505.06501

This means the project is not random. It sits inside an active systems/retrieval problem.

---

## 4. Core research question

The project should be guided by this question:

> **Can a vector search engine improve filtered ANN latency by using filter selectivity, metadata bitsets, and per-cluster statistics to choose a better execution plan?**

In simpler words:

> When filters are present, can we avoid searching irrelevant vectors faster than naive approaches?

---

## 5. Main hypothesis

Vortex.jl should test this hypothesis:

> For selective metadata filters, a filter-aware IVF planner can achieve lower p50/p95 latency than naive IVF post-filtering while maintaining higher Recall@k.

More concrete:

```text
Dataset: 100k to 1M vectors
Dimension: 128 / 384 / 768
Query type: vector + metadata filter
Metric: Recall@10, p50 latency, p95 latency
Baseline: naive IVF + post-filter
Goal: lower latency at equal or better Recall@10
```

---

## 6. What “faster” means

Do not define speed vaguely.

Use this definition:

```text
A method is faster if it has lower p50 and p95 query latency
at the same or higher Recall@k
under the same dataset, hardware, vector dimension, filter selectivity, and k.
```

Important metrics:

| Metric | Meaning |
|---|---|
| Recall@k | How many true nearest filtered neighbors appear in top k |
| p50 latency | Median query time |
| p95 latency | Tail latency |
| QPS | Queries per second |
| Build time | Time needed to build the index |
| Index size | Memory/disk size of the index |
| Candidate count | Number of vectors actually scored per query |

The strongest table in the README should look like this:

| Method | Recall@10 | p50 latency | p95 latency | Avg candidates scored |
|---|---:|---:|---:|---:|
| Exact filtered search | 1.00 | 42 ms | 70 ms | 10,000 |
| IVF post-filter | 0.72 | 11 ms | 21 ms | 2,000 |
| IVF high-nprobe post-filter | 0.94 | 39 ms | 82 ms | 18,000 |
| Vortex filter-aware IVF | 0.95 | 17 ms | 31 ms | 5,200 |

The actual numbers will come from experiments. Do not invent final results.

---

## 7. Non-goals

These are things **not** to do in the first serious version:

- Do not build a chatbot.
- Do not build a full Pinecone/Qdrant clone.
- Do not build authentication, tenants, billing, dashboards, or APIs first.
- Do not use FAISS internally and call it your engine.
- Do not start with HNSW.
- Do not start with GPU optimization.
- Do not start with distributed storage.
- Do not claim universal superiority.
- Do not make UI the main project.

The goal is a focused engine and benchmark system.

---

## 8. High-level system architecture

```text
                         ┌──────────────────────┐
                         │      Query API       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Query Planner      │
                         │ selectivity estimate │
                         │ strategy selection   │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ Pre-filter Exact │  │ Global IVF Search│  │ Filter-aware IVF │
   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
            │                     │                     │
            ▼                     ▼                     ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                    Candidate Generator                       │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                    Vector Scoring Layer                      │
   │              dot / cosine over normalized vectors             │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                      Top-k Result Heap                       │
   └──────────────────────────────────────────────────────────────┘
```

---

## 9. Core execution strategies

### 9.1 ExactSearch

Search all vectors or all vectors matching the filter.

Best for:

```text
small datasets
very selective filters
baseline ground truth
```

Pseudo-logic:

```text
candidate_ids = all ids matching filter
score query against every candidate
return top k
```

This is slow at scale but gives perfect recall. It is also needed to compute ground truth.

---

### 9.2 Naive IVF

Build an IVF index over all vectors.

At query time:

```text
find nearest centroids
scan vectors inside those lists
return top k
```

This is the normal approximate-search baseline.

---

### 9.3 IVF + post-filter

Run IVF first, then remove results that do not match the metadata filter.

Pseudo-logic:

```text
candidates = ivf_search(query, nprobe)
filtered_candidates = candidates matching filter
return top k from filtered_candidates
```

Problem:

```text
If the filter is selective, many good vector candidates may be removed.
Recall can collapse.
```

---

### 9.4 Pre-filter exact

Apply metadata filter first, then exact-search only matching vectors.

Pseudo-logic:

```text
candidate_ids = metadata_bitset_filter(filter)
score query against candidate_ids
return top k
```

This is good when filters are very selective.

Problem:

```text
If the filter matches too many vectors, it becomes slow.
```

---

### 9.5 FilterAwareIVF — the actual contribution

Use both vector closeness and metadata statistics to choose IVF lists.

Normal IVF chooses lists by centroid distance only:

```text
list_score = similarity(query, centroid)
```

Vortex should include expected filter usefulness:

```text
list_score = vector_weight * similarity(query, centroid)
           + filter_weight * estimated_filter_density(list, filter)
```

Where:

```text
estimated_filter_density(list, filter)
= matching_docs_in_list / total_docs_in_list
```

The planner should probe lists that are both:

1. close to the query vector
2. likely to contain vectors satisfying the metadata filter

This is the main algorithmic idea.

---

## 10. Data structures

### 10.1 Vector storage

Store vectors as normalized `Float32` arrays.

If vectors are normalized at insert/build time, cosine similarity becomes dot product:

```text
cosine(q, x) = dot(normalize(q), normalize(x))
```

This avoids recomputing norms on every query.

Potential storage layout:

```text
vectors.bin
  raw Float32 values
  row-major layout:
  vector_1[1:dim], vector_2[1:dim], ...
```

Julia is column-major internally, so you need to deliberately benchmark the layout. A simple first implementation can use a `Matrix{Float32}` and optimize later.

---

### 10.2 ID mapping

Need stable external IDs and internal integer IDs.

```text
external ID: "doc-abc-123"
internal ID: 42
```

Files:

```text
ids.jsonl
id_map.bin or id_map.jls
```

V1 can use JSONL or Julia serialization. Later, use a more explicit binary format.

---

### 10.3 Metadata store

Metadata example:

```json
{
  "id": "doc-1",
  "topic": "systems",
  "language": "julia",
  "year": 2025
}
```

V1 metadata can support:

```text
categorical equality filters:
  topic = systems
  language = julia

integer range filters:
  year >= 2024
```

Start with equality filters first.

---

### 10.4 Metadata bitsets

For each metadata predicate, store a bitset of matching internal IDs.

Example:

```text
topic=systems
  101001000111...

language=julia
  001101010011...
```

A filter becomes bitset operations:

```text
topic=systems AND language=julia
= bitset(topic=systems) & bitset(language=julia)
```

This is important because it makes filtering very fast.

Julia implementation options:

```julia
BitVector
```

Start with `BitVector`. Later, if needed, investigate compressed bitmaps.

---

### 10.5 IVF index

IVF structure:

```text
centroids::Matrix{Float32}
lists::Vector{Vector{Int32}}
```

Each vector is assigned to one centroid/list.

```text
centroid 1 -> [2, 91, 500, ...]
centroid 2 -> [3, 80, 120, ...]
```

You need:

```text
build_ivf(vectors, nlists)
assign vector to nearest centroid
search nearest centroids at query time
scan selected lists
```

For V1, k-means can be simple and not perfect. The benchmarking must mention the clustering implementation.

---

### 10.6 Per-list metadata statistics

For every IVF list, store counts for metadata values.

Example:

```json
{
  "list_id": 12,
  "total": 1840,
  "topic=systems": 420,
  "topic=ml": 97,
  "language=julia": 52
}
```

This allows the planner to estimate whether a list is useful for a filter.

---

## 11. Proposed repository structure

```text
Vortex.jl/
├── Project.toml
├── README.md
├── Manifest.toml
├── src/
│   ├── Vortex.jl
│   ├── api.jl
│   ├── types.jl
│   ├── storage/
│   │   ├── vector_store.jl
│   │   ├── id_store.jl
│   │   ├── metadata_store.jl
│   │   └── manifest_store.jl
│   ├── metrics/
│   │   ├── dot.jl
│   │   ├── cosine.jl
│   │   └── topk.jl
│   ├── filters/
│   │   ├── filter_expr.jl
│   │   ├── bitset_index.jl
│   │   └── selectivity.jl
│   ├── indexes/
│   │   ├── exact.jl
│   │   ├── ivf.jl
│   │   └── filter_aware_ivf.jl
│   ├── planner/
│   │   ├── planner.jl
│   │   ├── cost_model.jl
│   │   └── strategies.jl
│   ├── bench/
│   │   ├── datasets.jl
│   │   ├── groundtruth.jl
│   │   ├── metrics.jl
│   │   └── runner.jl
│   └── cli/
│       └── main.jl
├── test/
│   ├── runtests.jl
│   ├── test_metrics.jl
│   ├── test_filters.jl
│   ├── test_exact.jl
│   ├── test_ivf.jl
│   ├── test_planner.jl
│   └── test_bench.jl
├── experiments/
│   ├── synthetic_selectivity.jl
│   ├── ivf_vs_filtered_ivf.jl
│   ├── selectivity_sweep.jl
│   └── latency_recall_tradeoff.jl
├── docs/
│   ├── design.md
│   ├── benchmark_protocol.md
│   ├── research_log.md
│   └── results.md
└── scripts/
    ├── generate_synthetic_dataset.jl
    ├── run_all_benchmarks.jl
    └── plot_results.jl
```

---

## 12. What to write in each file

### `src/Vortex.jl`

Purpose:

- main module definition
- include all submodules/files
- export public API

Should contain:

```julia
module Vortex

include("types.jl")
include("api.jl")
...

export VectorDB, open_db, insert!, build_index!, search

end
```

Do not put algorithm logic here.

---

### `src/types.jl`

Purpose:

- core structs and type definitions

Should define:

```julia
struct VectorDB
    path::String
    dim::Int
    metric::Symbol
    # storage handles
    # metadata index
    # vector index
end

abstract type AbstractVectorIndex end
struct ExactIndex <: AbstractVectorIndex end
struct IVFIndex <: AbstractVectorIndex end
struct FilterAwareIVFIndex <: AbstractVectorIndex end
```

Also define result types:

```julia
struct SearchResult
    id::String
    score::Float32
    metadata::Dict{String,Any}
end
```

---

### `src/api.jl`

Purpose:

- public functions used by the user

Should define:

```julia
open_db(path)
create_db(path; dim, metric)
insert!(db, id, vector, metadata)
build_index!(db, index_config)
search(db, query; k=10, filter=nothing, strategy=:auto)
```

This file should call lower-level modules but not implement all internals.

---

### `src/storage/vector_store.jl`

Purpose:

- store and load vectors

V1 responsibilities:

- append vector
- read vector by internal ID
- load all vectors for exact search
- normalize vector if metric is cosine

Possible functions:

```julia
append_vector!(store, vector)::Int
read_vector(store, internal_id)::Vector{Float32}
read_vectors(store, ids)::Matrix{Float32}
num_vectors(store)::Int
```

Implementation advice:

Start simple with an in-memory matrix and a disk save/load function. After correctness, move to binary append-only storage.

---

### `src/storage/id_store.jl`

Purpose:

- map external document IDs to internal integer IDs

Functions:

```julia
external_to_internal(store, external_id)
internal_to_external(store, internal_id)
add_id!(store, external_id)::Int
```

Use simple dictionaries in V1.

---

### `src/storage/metadata_store.jl`

Purpose:

- store metadata rows
- retrieve metadata by internal ID

Functions:

```julia
add_metadata!(store, internal_id, metadata)
get_metadata(store, internal_id)
```

V1 can store metadata in memory and persist as JSONL.

---

### `src/storage/manifest_store.jl`

Purpose:

- database metadata

Manifest should store:

```json
{
  "version": 1,
  "dim": 384,
  "metric": "cosine",
  "count": 100000,
  "index_type": "filter_aware_ivf",
  "created_at": "..."
}
```

---

### `src/metrics/dot.jl`

Purpose:

- fast dot product scoring

Functions:

```julia
dot_score(q, x)::Float32
batch_dot_scores(q, matrix)::Vector{Float32}
```

Performance note:

Use `Float32`, avoid unnecessary allocations, and benchmark matrix-vector multiplication options.

---

### `src/metrics/cosine.jl`

Purpose:

- normalization and cosine similarity

Functions:

```julia
normalize_vector(v)::Vector{Float32}
cosine_score(q, x)::Float32
```

Design note:

For cosine databases, normalize vectors at insert/build time and normalize query vector once per query. Then use dot product.

---

### `src/metrics/topk.jl`

Purpose:

- maintain top-k results efficiently

V1 can sort all scores.

V2 should use a min-heap or partial selection.

Functions:

```julia
topk(scores, ids, k)
```

---

### `src/filters/filter_expr.jl`

Purpose:

- represent filter expressions

Start with simple equality filters:

```julia
Filter(:topic, :eq, "systems")
AndFilter(filter1, filter2)
```

Later add:

```text
range filters
OR filters
NOT filters
```

Do not implement a full SQL parser early.

---

### `src/filters/bitset_index.jl`

Purpose:

- build metadata bitsets
- evaluate filters quickly

Functions:

```julia
build_bitset_index(metadata_rows)
evaluate_filter(bitset_index, filter)::BitVector
```

Example:

```julia
evaluate_filter(index, AndFilter(
    Filter(:topic, :eq, "systems"),
    Filter(:language, :eq, "julia")
))
```

---

### `src/filters/selectivity.jl`

Purpose:

- estimate how selective a filter is

Functions:

```julia
estimate_selectivity(bitset_index, filter)::Float64
```

Formula:

```text
selectivity = count_ones(filter_bitset) / total_vectors
```

This feeds the query planner.

---

### `src/indexes/exact.jl`

Purpose:

- exact search baseline

Functions:

```julia
search_exact(db, q; k, filter)
```

Must support:

- unfiltered exact search
- filtered exact search

This is also ground truth for recall measurement.

---

### `src/indexes/ivf.jl`

Purpose:

- basic IVF index

Should define:

```julia
struct IVFIndex
    centroids::Matrix{Float32}
    lists::Vector{Vector{Int32}}
    nlists::Int
end
```

Functions:

```julia
build_ivf(vectors; nlists)
search_ivf(index, q; k, nprobe)
```

Do not overcomplicate k-means at first. Correctness and measurement matter first.

---

### `src/indexes/filter_aware_ivf.jl`

Purpose:

- actual project contribution

Should define:

```julia
struct FilterAwareIVFIndex
    ivf::IVFIndex
    list_metadata_counts::Dict
    list_bitsets::Vector{BitVector}
end
```

Functions:

```julia
build_filter_aware_ivf(vectors, metadata; nlists)
search_filter_aware_ivf(index, q, filter; k, nprobe)
rank_lists_for_filter(index, q, filter)
```

List ranking idea:

```text
score(list) = alpha * centroid_similarity(query, list_centroid)
            + beta  * filter_density(list, filter)
```

Experiment with alpha/beta values.

---

### `src/planner/planner.jl`

Purpose:

- choose the execution strategy

Strategy decision example:

```text
if no filter:
    use IVF
else if selectivity < 0.005:
    use PreFilterExact
else if selectivity < 0.20:
    use FilterAwareIVF
else:
    use IVFPostFilter or normal IVF
```

Functions:

```julia
choose_strategy(db, q, filter, k)::SearchStrategy
```

---

### `src/planner/cost_model.jl`

Purpose:

- estimate candidate counts and costs

V1 can be rule-based.

V2 can learn/fit simple cost formulas from previous benchmark runs.

Cost model inputs:

```text
N = total vectors
s = filter selectivity
nlists = number of IVF lists
nprobe = lists probed
avg_list_size
estimated matching vectors per list
```

---

### `src/planner/strategies.jl`

Purpose:

- define named strategy types

Examples:

```julia
abstract type SearchStrategy end
struct ExactStrategy <: SearchStrategy end
struct PreFilterExactStrategy <: SearchStrategy end
struct IVFPostFilterStrategy <: SearchStrategy end
struct FilterAwareIVFStrategy <: SearchStrategy end
```

---

### `src/bench/datasets.jl`

Purpose:

- create or load benchmark datasets

Start with synthetic datasets.

Synthetic generator should control:

```text
number of vectors
vector dimension
number of metadata categories
filter selectivity
correlation between metadata and vector clusters
```

This last point is important. Filtered search difficulty depends on whether metadata filters are correlated with vector space.

---

### `src/bench/groundtruth.jl`

Purpose:

- compute exact filtered nearest neighbors

Functions:

```julia
compute_groundtruth(dataset, queries, filters; k)
```

This is needed for Recall@k.

---

### `src/bench/metrics.jl`

Purpose:

- benchmark metrics

Implement:

```julia
recall_at_k(predicted, truth, k)
latency_summary(times)
candidate_count_summary(counts)
```

Optional later:

```text
NDCG@k
MRR
QPS
```

---

### `src/bench/runner.jl`

Purpose:

- run full benchmark experiments

Example:

```julia
run_benchmark(
    dataset,
    methods=[Exact, IVFPostFilter, PreFilterExact, FilterAwareIVF],
    selectivities=[0.001, 0.01, 0.05, 0.1, 0.5],
    k=10
)
```

Output should be a table and JSON/CSV file.

---

### `experiments/synthetic_selectivity.jl`

Purpose:

- first serious experiment

Experiment question:

> How does filter selectivity affect latency and recall for each strategy?

Variables:

```text
selectivity = 0.1%, 1%, 5%, 10%, 50%
methods = Exact, IVFPostFilter, PreFilterExact, FilterAwareIVF
```

Expected result:

```text
PreFilterExact wins at extremely low selectivity.
FilterAwareIVF should win at medium selectivity.
Normal IVF wins when filters are weak or absent.
```

---

### `experiments/ivf_vs_filtered_ivf.jl`

Purpose:

- compare normal IVF and filter-aware IVF directly

Experiment question:

> Does metadata-aware list ranking improve recall under filters?

---

### `experiments/latency_recall_tradeoff.jl`

Purpose:

- compare methods at different `nprobe` values

Important because ANN systems trade speed for accuracy.

Output:

```text
Recall@10 vs p95 latency curve
```

A curve is better than one cherry-picked number.

---

### `docs/design.md`

Purpose:

- explain architecture and design decisions

Should include:

```text
problem statement
why filtered vector search matters
architecture diagram
execution strategies
data structures
query planner design
```

---

### `docs/benchmark_protocol.md`

Purpose:

- define how experiments are run

Should include:

```text
hardware
Julia version
dataset sizes
vector dimensions
filter selectivities
warmup policy
number of queries
metric definitions
how ground truth is computed
```

This file prevents fake benchmark vibes.

---

### `docs/research_log.md`

Purpose:

- record your thinking over time

Use this format:

```markdown
## 2026-07-11

### Question
Can filter-aware list ranking improve recall over IVF post-filtering?

### Experiment
Synthetic dataset with 100k vectors, dim=128, selectivity=1%.

### Result
TBD.

### Interpretation
TBD.

### Next step
TBD.
```

This makes you look like someone doing real research, not random coding.

---

### `docs/results.md`

Purpose:

- final results and analysis

Should include:

```text
benchmark tables
plots
what worked
what failed
limitations
future work
```

Do not hide failures. Research-style projects look stronger when limitations are honestly explained.

---

## 13. Development phases

### Phase 0 — Reading and setup

Goal:

> Understand vector search, ANN benchmarking, and filtered ANN enough to design experiments.

Read:

1. FAISS documentation overview  
   https://faiss.ai/index.html
2. ANN-Benchmarks overview  
   https://ann-benchmarks.com/index.html
3. Filtered ANN system design paper  
   https://arxiv.org/abs/2602.11443
4. FANNS survey  
   https://arxiv.org/abs/2505.06501

Deliverables:

```text
docs/research_log.md
docs/design.md first draft
```

Do not code engine internals before this.

---

### Phase 1 — Evaluation harness

Goal:

> Build the benchmark/evaluation system before optimizing anything.

Implement:

```text
synthetic vector dataset generator
synthetic metadata generator
query generator
filter generator
exact ground truth
Recall@k
latency measurement
```

Why first?

Because without this, you cannot prove anything is faster.

Deliverable:

```text
Able to generate 10k vectors, 100 queries, filters, and exact top-k ground truth.
```

---

### Phase 2 — Exact search baseline

Goal:

> Implement perfect but slow search.

Implement:

```text
normalized vector insertion
exact unfiltered search
exact filtered search
metadata bitset filtering
```

Deliverable:

```text
Exact search works and all tests pass.
```

---

### Phase 3 — Naive IVF baseline

Goal:

> Implement standard approximate search.

Implement:

```text
k-means centroid training
vector assignment to lists
nprobe-based search
IVF post-filter search
```

Deliverable:

```text
IVF returns approximate results.
Benchmark shows speed/recall tradeoff against exact search.
```

---

### Phase 4 — Filter-aware IVF

Goal:

> Implement the actual research contribution.

Implement:

```text
per-list metadata counts
filter density estimation
list ranking using vector closeness + filter density
adaptive nprobe or candidate target
```

Deliverable:

```text
FilterAwareIVF competes against IVFPostFilter and PreFilterExact.
```

---

### Phase 5 — Query planner

Goal:

> Automatically choose strategy based on filter selectivity.

Implement:

```text
selectivity estimator
rule-based cost model
auto strategy selection
planner explanation output
```

Example planner explanation:

```text
Strategy: PreFilterExact
Reason:
  filter selectivity = 0.18%
  estimated candidates = 1,800
  exact filtered scan cheaper than IVF probing
```

This is great for demo and debugging.

---

### Phase 6 — Performance pass

Goal:

> Make it fast after it is correct.

Optimize:

```text
avoid allocations
use Float32
normalize once
batch dot products
profile hot paths
use Julia @time / @btime / Profile
try Threads.@threads for candidate scoring
```

Do not optimize before Phase 4 works.

---

### Phase 7 — Final experiments and writeup

Goal:

> Produce resume-grade results.

Run experiments over:

```text
N = 10k, 100k, maybe 1M if hardware allows
dim = 128, 384, 768
selectivity = 0.1%, 1%, 5%, 10%, 50%
k = 10
methods = Exact, PreFilterExact, IVFPostFilter, FilterAwareIVF, AutoPlanner
```

Write:

```text
README.md
docs/results.md
benchmark plots
limitations
future work
```

---

## 14. Testing plan

### Unit tests

Test these separately:

```text
cosine/dot scoring
vector normalization
top-k correctness
filter expression evaluation
bitset AND/OR behavior
exact search correctness
IVF list assignment
planner strategy choice
recall@k calculation
```

### Integration tests

Test full workflows:

```text
create DB
insert vectors
build bitset index
build IVF
run filtered query
compare against exact ground truth
```

### Benchmark correctness tests

Before trusting benchmarks:

```text
exact search Recall@10 must be 1.0 against ground truth
IVF recall must improve as nprobe increases
filtered exact result must only contain matching metadata
```

---

## 15. Benchmark methodology

A benchmark must include:

```text
hardware
OS
Julia version
number of vectors
vector dimension
number of queries
filter selectivity
index parameters
warmup runs
measurement runs
recall definition
latency summary
```

Example benchmark protocol:

```text
Warmup:
  run 100 queries and ignore timings

Measurement:
  run 1000 queries
  record per-query latency
  record Recall@10
  record candidate count

Report:
  mean recall
  p50 latency
  p95 latency
  average candidate count
```

Never benchmark only one query.

---

## 16. Possible datasets

### Start with synthetic data

Advantages:

```text
full control over selectivity
full control over correlation
easy to generate ground truth
no dataset download friction
```

Synthetic metadata fields:

```text
topic: 20 categories
language: 5 categories
year: 2018-2026
source: 10 categories
```

### Later use real embeddings

Possible real datasets:

```text
Wikipedia passages
StackOverflow posts
arXiv abstracts
GitHub README files
```

Do not start with real data. Synthetic first is better for algorithm development.

---

## 17. Main experiments

### Experiment 1 — Selectivity sweep

Question:

> Which strategy wins as filter selectivity changes?

Variables:

```text
selectivity: 0.1%, 1%, 5%, 10%, 50%, 100%
```

Expected interpretation:

```text
Very selective filters -> PreFilterExact should do well.
Medium filters -> FilterAwareIVF should do well.
Weak filters -> normal IVF should do well.
```

---

### Experiment 2 — Recall-latency curve

Question:

> At the same Recall@10, which method is faster?

This is the most important experiment.

Plot:

```text
x-axis: p95 latency
y-axis: Recall@10
one curve per method
```

---

### Experiment 3 — Candidate efficiency

Question:

> How many vectors does each method score per query?

Why useful:

```text
If Vortex is faster, candidate count should explain why.
```

Report:

```text
method
avg candidates scored
latency
recall
```

---

### Experiment 4 — Metadata/vector correlation

Question:

> Does filter-aware search work better when metadata is correlated with vector clusters?

Generate datasets where:

```text
metadata strongly aligned with vector clusters
metadata randomly assigned
metadata anti-correlated with clusters
```

This is research-quality because it shows where the method works and where it fails.

---

## 18. Success criteria

Minimum success:

```text
exact search works
metadata filtering works
IVF works
benchmarks run
```

Good success:

```text
FilterAwareIVF beats IVFPostFilter on selective filters at similar recall.
```

Excellent success:

```text
AutoPlanner chooses the best or near-best strategy across selectivity regimes.
```

Research-quality success:

```text
The project identifies when filter-aware IVF works, when it fails, and why.
```

---

## 19. How to keep this from becoming AI-generated junk

You specifically want to implement and understand it yourself. Use this rule:

> AI can explain concepts, but you must write the design notes, tests, experiments, and final implementation decisions yourself.

Recommended workflow:

1. Write the design in your own words before coding.
2. Implement one module at a time.
3. Write tests before optimization.
4. Record failed experiments in `docs/research_log.md`.
5. Do not paste large generated code blocks.
6. If you use AI for help, ask for explanations or review, not full files.
7. Every benchmark result must be produced by your local code.
8. Every README claim must point to a script that reproduces it.

This will make you actually own the project.

---

## 20. Suggested weekly plan

### Week 1 — Research and design

Deliverables:

```text
docs/design.md
docs/benchmark_protocol.md
repository skeleton
```

Learn:

```text
cosine similarity
Recall@k
IVF index intuition
filtered ANN strategies
Julia package structure
```

---

### Week 2 — Dataset and exact search

Deliverables:

```text
synthetic dataset generator
metadata generator
exact filtered search
Recall@k implementation
unit tests
```

---

### Week 3 — Metadata bitsets

Deliverables:

```text
BitVector metadata index
AND filters
selectivity estimator
pre-filter exact strategy
```

---

### Week 4 — Basic IVF

Deliverables:

```text
k-means training
IVF list assignment
unfiltered IVF search
IVF post-filter baseline
```

---

### Week 5 — Filter-aware IVF

Deliverables:

```text
per-list metadata counts
filter density calculation
filter-aware list ranking
benchmark against baselines
```

---

### Week 6 — Planner

Deliverables:

```text
strategy selector
cost model
planner explanation output
selectivity sweep experiment
```

---

### Week 7 — Optimization

Deliverables:

```text
profiling results
reduced allocations
threaded scoring if useful
latency improvements documented
```

---

### Week 8 — Final results

Deliverables:

```text
README
benchmark tables
plots
limitations
resume bullets
technical blog draft
```

---

## 21. README structure

Your final README should be structured like this:

```markdown
# Vortex.jl

One-line description.

## Why filtered vector search?

Explain problem.

## Core idea

Explain filter-aware IVF planner.

## Features

- exact search
- IVF
- metadata bitset filters
- filter-aware IVF
- auto planner
- benchmark harness

## Quick demo

Show commands.

## Benchmark results

Tables and plots.

## Architecture

Diagram.

## Reproducing benchmarks

Commands.

## Limitations

Honest limitations.

## References

Papers/tools.
```

---

## 22. Final resume bullets

Use these only after the features actually exist.

### Bullet 1

> Built Vortex.jl, a filter-aware vector search engine in Julia with exact search, IVF indexing, metadata bitset filters, and adaptive query planning.

### Bullet 2

> Implemented a FilterAwareIVF strategy using per-cluster metadata statistics to reduce wasted candidate scoring under selective metadata filters.

### Bullet 3

> Designed a reproducible benchmark suite measuring Recall@10, p50/p95 latency, candidate counts, and selectivity sensitivity across exact, pre-filter, post-filter, and filter-aware ANN strategies.

### Stronger bullet if results are good

> Achieved X% lower p95 latency than naive IVF post-filtering at comparable Recall@10 on synthetic filtered vector-search workloads.

Only fill in X after real benchmarks.

---

## 23. Final project identity

The project should be presented as:

> **A research-oriented Julia vector search engine focused on filtered ANN query planning.**

Not:

> A vector database clone.

Not:

> A RAG app.

Not:

> A wrapper around existing libraries.

The final story:

```text
I studied where vector search gets slow under metadata filters.
I built exact and approximate baselines.
I implemented metadata bitsets and IVF.
I designed a filter-aware candidate generation strategy.
I benchmarked recall-latency tradeoffs and documented where my strategy works.
```

That is a serious third-year CS project.

---

## 24. Immediate next action

Before writing engine code, create only these files:

```text
README.md
docs/design.md
docs/benchmark_protocol.md
docs/research_log.md
src/Vortex.jl
src/types.jl
test/runtests.jl
```

Then write the first version of `docs/design.md` in your own words.

The first code you should write is **not IVF**.

The first code should be:

```text
synthetic dataset generator
exact filtered search
Recall@k
```

Because the benchmark harness is the foundation of the whole project.
