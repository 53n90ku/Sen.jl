# Sen v0.1 benchmark snapshot

Generated  with Julia 1.12.6 on Darwin aarch64, 1 Julia thread(s), CPU: Apple M2

All rows use the frozen `quality_contract.toml` datasets, held-out queries, index format v2, target recall 0.85, and a p95 latency ceiling of 10 ms.

| Workload | Metric | Filter | Method | Vectors | Dim | nprobe | Recall@10 | p95 ms | Build ms | Index KiB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cosine-unfiltered | cosine | none | ivf | 1024 | 32 | 2 | 1.0 | 0.006 | 12.38 | 80.48 |
| cosine-filtered | cosine | selected | ivf_prefilter | 1024 | 32 | 16 | 1.0 | 0.022 | 11.1 | 83.97 |
| cosine-filtered | cosine | selected | ivf_postfilter | 1024 | 32 | 16 | 1.0 | 0.395 | 11.1 | 83.97 |
| cosine-filtered | cosine | selected | filter_aware | 1024 | 32 | 8 | 1.0 | 0.021 | 11.1 | 83.97 |
| cosine-filtered | cosine | selected | filter_aware_bound | 1024 | 32 | 4 | 0.9958 | 0.025 | 11.1 | 83.97 |
| dot-unfiltered | dot | none | ivf | 1024 | 32 | 2 | 1.0 | 0.006 | 19.94 | 84.3 |
| dot-filtered | dot | selected | ivf_prefilter | 1024 | 32 | 8 | 0.9292 | 0.016 | 12.15 | 82.03 |
| dot-filtered | dot | selected | ivf_postfilter | 1024 | 32 | 8 | 0.9292 | 0.216 | 12.15 | 82.03 |
| dot-filtered | dot | selected | filter_aware | 1024 | 32 | 2 | 0.9083 | 0.016 | 12.15 | 82.03 |

Reproduce the report from the repository root:

```bash
julia --project=benchmark/SenBench scripts/benchmark_release.jl
```


