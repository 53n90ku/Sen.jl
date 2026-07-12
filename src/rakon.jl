module rakon

include("types.jl")
include("api.jl")
include("bench/datasets.jl")
include("metrics/dot.jl")
include("metrics/cosine.jl")
include("metrics/topk.jl")
include("filters/filter_expr.jl")
include("filters/bitset_index.jl")
include("indexes/exact.jl")
include("filters/selectivity.jl")
include("bench/metrics.jl")
include("bench/groundtruth.jl")
include("indexes/ivf.jl")


export VectorDB
export create_db
export generate_synthetic_dataset
export dot_similarity,cosine_similarity,top_k
export search_exact
export matches_filter
export BitsetIndex, build_bitset_index,evaluate_filter
export estimate_selectivity
export recall_at_k
export compute_groundtruth
export IVFIndex, squared_distance, train_centroids
export nearest_centroid, build_ivf, search_ivf

end