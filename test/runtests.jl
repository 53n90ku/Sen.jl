using Test
using Sen

using Sen: BitsetIndex, BoundedFilterAwareIVFStrategy, DatabaseManifest, ExactStrategy
using Sen: FilterAwareIVFIndex, FilterAwareIVFStrategy, IDStore, IVFIndex
using Sen: IVFPostFilterStrategy, IVFPreFilterStrategy, IVFStrategy, IndexBuildConfig
using Sen: MetadataStore, PreFilterExactStrategy, QueryPlan, RecallCalibration
using Sen: RecallCalibrationEntry
using Sen: SearchStrategy, VectorStore, add_calibration_entry!, build_bitset_index
using Sen: build_filter_aware_ivf, build_ivf, choose_strategy, classify_filter_workload
using Sen: collect_filtered_list_candidates, collect_ivf_candidates, compute_list_radii
using Sen: column_dot, compute_vector_norms, cosine_similarity, create_database_manifest
using Sen: create_id_store, create_metadata_store, create_vector_store
using Sen: current_database_snapshot, database_snapshot_generations, dot_similarity
using Sen: database_current_path, database_writer_lock_path
using Sen: append_database_wal_put!, database_wal_path, read_database_wal
using Sen: estimate_candidate_count, estimate_filter_concentration
using Sen: estimate_list_filter_count, estimate_list_filter_density
using Sen: estimate_required_nprobe, estimate_selectivity, estimate_strategy_costs
using Sen: evaluate_filter, evaluate_list_filter, filtered_list_candidates, get_id
using Sen: get_metadata, get_position, get_vector, has_id, insert_id!, insert_metadata!
using Sen: insert_vector!, list_score_upper_bound, list_score_upper_bounds
using Sen: load_id_store, load_ivf_index, load_manifest, load_metadata_store
using Sen: is_mapped, load_recall_calibration, load_vector_store, lookup_calibrated_nprobe
using Sen: matches_filter, nearest_centroid, next_available_id, rank_bound_lists
using Sen: rank_filter_aware_lists, rank_ivf_lists, resolve_postfilter_oversample
using Sen: save_id_store, save_ivf_index, save_manifest, save_metadata_store
using Sen: save_recall_calibration, save_vector_store, search_exact
using Sen: score_ivf_candidates
using Sen: search_filter_aware_bound, search_filter_aware_bound_with_stats
using Sen: search_filter_aware_ivf, search_ivf, search_ivf_postfilter
using Sen: search_ivf_prefilter, select_filter_aware_lists, squared_distance
using Sen:
    stored_ids, stored_metadata, stored_vectors, strategy_from_symbol, swap_delete_vector!
using Sen: strategy_name, top_k, train_centroids, vector_norm
using Sen: update_vector!

@testset "Sen" begin
    db = create_db("test-db"; dim = 128, metric = :cosine, durable = false)

    @test db.path == "test-db"
    @test db.dim == 128
    @test db.metric == :cosine
end

include("helpers.jl")
include("core/test_metrics.jl")
include("core/test_execution_core.jl")
include("core/test_api.jl")
include("core/test_database_info.jl")
include("core/test_public_api.jl")
include("stores/test_storage.jl")
include("stores/test_metadata_store.jl")
include("stores/test_id_store.jl")
include("search/test_exact.jl")
include("search/test_ivf.jl")
include("search/test_batch_search.jl")
include("filters/test_filter_expr.jl")
include("filters/test_filters.jl")
include("filters/test_filter_properties.jl")
include("filters/test_metadata_index_expr.jl")
include("filters/test_expression_exact_ivf.jl")
include("filters/test_expression_filter_aware.jl")
include("filters/test_expression_bound_selectivity.jl")
include("filters/test_range_metadata_index.jl")
include("filters/test_range_filter_aware.jl")
include("filters/test_range_properties.jl")
include("planner/test_expression_planner.jl")
include("planner/test_planner.jl")
include("mutations/test_mutations.jl")
include("mutations/test_delta.jl")
include("mutations/test_segments.jl")
include("concurrency/test_concurrency.jl")
include("maintenance/test_maintenance.jl")
include("persistence/test_manifest_store.jl")
include("persistence/test_vector_persistence.jl")
include("persistence/test_index_persistence.jl")
include("persistence/test_metadata_persistence.jl")
include("persistence/test_database_persistence.jl")
include("persistence/test_wal.jl")
include("persistence/test_writer_lock.jl")
