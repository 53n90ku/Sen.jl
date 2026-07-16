function percentile(values::AbstractVector{<:Real}, probability::Real)
    isempty(values)&&throw(ArgumentError("values cannot be empty"))
    0<=probability<=1||throw(ArgumentError("probability must be between zero and one"))
    ordered=sort(Float64.(values))
    position=1+(length(ordered)-1)*Float64(probability)
    lower=floor(Int, position)
    upper=ceil(Int, position)
    lower==upper&&return ordered[lower]
    weight=position-lower
    return ordered[lower]*(1-weight)+ordered[upper]*weight
end

function gini_coefficient(values::AbstractVector{<:Real})
    isempty(values)&&throw(ArgumentError("values cannot be empty"))
    any(value->value<0, values)&&throw(ArgumentError("gini values cannot be negative"))
    ordered=sort(Float64.(values))
    total=sum(ordered)
    iszero(total)&&return 0.0
    weighted=sum(index*value for (index, value) in enumerate(ordered))
    count=length(ordered)
    return (2*weighted)/(count*total)-(count+1)/count
end

function selectivity_bucket(selectivity::Real)
    0<=selectivity<=1||throw(ArgumentError("selectivity must be between zero and one"))
    selectivity<=0.01&&return :rare
    selectivity<0.10&&return :medium
    return :broad
end

function probe_schedule(nlists::Int)
    nlists>0||throw(ArgumentError("nlists must be positive"))
    probes=Int[]
    probe=1
    while probe<nlists
        push!(probes, probe)
        probe*=2
    end
    append!(probes, ceil.(Int, nlists .* [0.125, 0.25, 0.375, 0.5, 0.75]))
    push!(probes, nlists)
    return sort!(unique(probes))
end

function ivf_vector_lists(index::IVFIndex)
    vector_count=sum(length, index.lists)
    assignments=zeros(Int, vector_count)
    for (list_index, list) in enumerate(index.lists)
        for vector_index in list
            1<=vector_index<=vector_count||throw(BoundsError(1:vector_count, vector_index))
            iszero(assignments[vector_index])||throw(
                ArgumentError("vector occurs in more than one IVF list"),
            )
            assignments[vector_index]=list_index
        end
    end
    all(value->!iszero(value), assignments)||throw(
        ArgumentError("IVF index does not assign every vector"),
    )
    return assignments
end

function ivf_index_diagnostics(index::IVFIndex, vectors::AbstractMatrix)
    size(vectors, 2)==sum(length, index.lists)||throw(
        DimensionMismatch("vectors do not match IVF index"),
    )
    size(vectors, 1)==size(index.centroids, 1)||throw(
        DimensionMismatch("vector dimensions do not match IVF index"),
    )
    assignments=ivf_vector_lists(index)
    list_sizes=length.(index.lists)
    average_size=sum(list_sizes)/length(list_sizes)
    variance=sum((size-average_size)^2 for size in list_sizes)/length(list_sizes)
    losses=Vector{Float64}(undef, length(assignments))

    for vector_index in eachindex(assignments)
        list_index=assignments[vector_index]
        norm=index.vector_norms[vector_index]
        iszero(norm)&&throw(ArgumentError("vector cannot be zero"))
        similarity=column_dot(@view(index.centroids[:, list_index]), vectors, vector_index)/norm
        losses[vector_index]=index.metric===:cosine ? 1-clamp(similarity, -1, 1) :
                             -similarity
    end

    centroid_similarity=transpose(index.centroids)*index.centroids
    maximum_similarity=-Inf
    duplicate_pairs=0
    for right in axes(centroid_similarity, 2)
        for left = 1:(right-1)
            similarity=centroid_similarity[left, right]
            maximum_similarity=max(maximum_similarity, similarity)
            similarity>=0.999&&(duplicate_pairs+=1)
        end
    end
    size(centroid_similarity, 2)==1&&(maximum_similarity=NaN)

    return (
        vector_count = length(assignments),
        dimension = size(vectors, 1),
        nlists = length(index.lists),
        empty_lists = count(iszero, list_sizes),
        minimum_list_size = minimum(list_sizes),
        median_list_size = percentile(list_sizes, 0.50),
        p95_list_size = percentile(list_sizes, 0.95),
        maximum_list_size = maximum(list_sizes),
        average_list_size = average_size,
        list_size_cv = iszero(average_size) ? 0.0 : sqrt(variance)/average_size,
        list_size_gini = gini_coefficient(list_sizes),
        maximum_to_average_list_ratio = iszero(average_size) ? 0.0 :
                                        maximum(list_sizes)/average_size,
        mean_quantization_loss = sum(losses)/length(losses),
        p50_quantization_loss = percentile(losses, 0.50),
        p95_quantization_loss = percentile(losses, 0.95),
        maximum_quantization_loss = maximum(losses),
        maximum_centroid_similarity = maximum_similarity,
        near_duplicate_centroids = duplicate_pairs,
        p50_list_radius = percentile(index.list_radii, 0.50),
        p95_list_radius = percentile(index.list_radii, 0.95),
        maximum_list_radius = maximum(index.list_radii),
    )
end

function truth_probe_requirement(ranks::AbstractVector{<:Integer}, fraction::Real)
    isempty(ranks)&&return 0
    count=clamp(ceil(Int, fraction*length(ranks)), 1, length(ranks))
    return sort(Int.(ranks))[count]
end

function ivf_routing_diagnostics(
    index::IVFIndex,
    queries::AbstractMatrix,
    truth_ids::AbstractVector;
    metadata = nothing,
    filters = nothing,
)
    size(queries, 1)==size(index.centroids, 1)||throw(
        DimensionMismatch("queries do not match IVF index"),
    )
    size(queries, 2)==length(truth_ids)||throw(
        DimensionMismatch("truth count does not match queries"),
    )
    filters===nothing||length(filters)==size(queries, 2)||throw(
        DimensionMismatch("filter count does not match queries"),
    )
    metadata===nothing||length(metadata)==sum(length, index.lists)||throw(
        DimensionMismatch("metadata does not match IVF index"),
    )
    assignments=ivf_vector_lists(index)
    selectivities=metadata===nothing||filters===nothing ? fill(NaN, size(queries, 2)) :
                  dataset_selectivities(metadata, filters)
    rows=NamedTuple[]

    for query_index in axes(queries, 2)
        query=@view queries[:, query_index]
        distances=centroid_distances(index, query)
        ranking=sortperm(distances)
        rank_by_list=zeros(Int, length(ranking))
        for (rank, list_index) in enumerate(ranking)
            rank_by_list[list_index]=rank
        end
        truth=truth_ids[query_index]
        truth_ranks=Int[rank_by_list[assignments[vector_index]] for vector_index in truth]
        margin=length(distances)==1 ? Inf : distances[ranking[2]]-distances[ranking[1]]
        selectivity=selectivities[query_index]
        push!(
            rows,
            (
                query_index = query_index,
                selectivity = selectivity,
                bucket = isnan(selectivity) ? :unknown : selectivity_bucket(selectivity),
                centroid_margin = margin,
                truth_list_count = length(
                    unique(assignments[vector_index] for vector_index in truth),
                ),
                nprobe_50 = truth_probe_requirement(truth_ranks, 0.50),
                nprobe_90 = truth_probe_requirement(truth_ranks, 0.90),
                nprobe_95 = truth_probe_requirement(truth_ranks, 0.95),
                nprobe_100 = truth_probe_requirement(truth_ranks, 1.00),
                maximum_truth_list_rank = isempty(truth_ranks) ? 0 : maximum(truth_ranks),
            ),
        )
    end

    return rows
end

function ivf_probe_curve(
    index::IVFIndex,
    queries::AbstractMatrix,
    truth_ids::AbstractVector,
    nprobes::AbstractVector{<:Integer};
    metadata = nothing,
    filters = nothing,
)
    routing=ivf_routing_diagnostics(
        index,
        queries,
        truth_ids;
        metadata = metadata,
        filters = filters,
    )
    assignments=ivf_vector_lists(index)
    rows=NamedTuple[]

    for query_index in axes(queries, 2)
        query=@view queries[:, query_index]
        ranking=rank_ivf_lists(index, query; nprobe = maximum(nprobes))
        truth_lists=Int[assignments[index] for index in truth_ids[query_index]]
        list_sizes=Int[length(index.lists[list_index]) for list_index in ranking]
        matching_sizes=metadata===nothing||filters===nothing ? copy(list_sizes) :
                       Int[
            count(
                index->matches_filter(metadata[index], filters[query_index]),
                index.lists[list_index],
            ) for list_index in ranking
        ]
        cumulative_sizes=cumsum(list_sizes)
        cumulative_matching=cumsum(matching_sizes)
        for nprobe in nprobes
            1<=nprobe<=length(index.lists)||throw(
                ArgumentError("nprobe must be between one and list count"),
            )
            selected=Set(ranking[1:nprobe])
            covered=count(in(selected), truth_lists)
            denominator=length(truth_lists)
            push!(
                rows,
                merge(
                    routing[query_index],
                    (
                        nprobe = nprobe,
                        truth_list_recall = iszero(denominator) ? 1.0 : covered/denominator,
                        candidates_visited = cumulative_sizes[nprobe],
                        matching_candidates = cumulative_matching[nprobe],
                        database_fraction = cumulative_sizes[nprobe]/sum(
                            length,
                            index.lists,
                        ),
                    ),
                ),
            )
        end
    end

    return rows
end
