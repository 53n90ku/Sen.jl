using Test
using Sen

@testset "sqared distance" begin
    a = Float32[1,2]
    b = Float32[4,6]

    @test squared_distance(a,b) == 25.0f0
    @test_throws DimensionMismatch squared_distance(a,Float32[1])
end

@testset "dot product IVF routing" begin
    centroids=Float32[10 1;0 1]
    index=IVFIndex(
        centroids,
        [Int[],Int[]],
        Float32[],
        :dot,
        fill(Float32(pi),2),
        fill(-1.0f0,2),
        zeros(Float32,2),
    )
    query=Float32[1,1]

    @test Sen.centroid_distances(index,query)==Float32[-10,-2]
    @test nearest_centroid(centroids,query;metric=:dot,)==1
    @test rank_ivf_lists(index,query;nprobe=1,)==[1]
end

@testset "ivf postfilter oversampling" begin
    vectors=Float32[
        1.0 0.9 0.8 0.7
        0.0 0.0 0.0 0.0
    ]

    metadata=[
        (allowed=false,),
        (allowed=false,),
        (allowed=true,),
        (allowed=true,),
    ]

    centroids=reshape(Float32[1.0,0.0],2,1)
    index=IVFIndex(centroids,[[1,2,3,4]])
    query=Float32[1.0,0.0]

    without_oversampling=search_ivf_postfilter(index,vectors,metadata,query;k=2,nprobe=1,metric=:dot,filter=(allowed=true,),oversample=1,)
    with_oversampling=search_ivf_postfilter(index,vectors,metadata,query;k=2,nprobe=1,metric=:dot,filter=(allowed=true,),oversample=2,)

    @test isempty(without_oversampling)
    @test length(with_oversampling)==2
    @test [result.index for result in with_oversampling]==[3,4]
    @test resolve_postfilter_oversample(10,0.05;candidate_multiplier=4.0,)==80
    @test resolve_postfilter_oversample(10,0.50;candidate_multiplier=4.0,)==10

    @test_throws ArgumentError search_ivf_postfilter(index,vectors,metadata,query;k=2,nprobe=1,filter=(allowed=true,),oversample=0,)
end

@testset "ivf prefilter search" begin
    vectors=Float32[
        1.0 0.9 0.8 0.7
        0.0 0.0 0.0 0.0
    ]
    metadata=[
        (allowed=false,),
        (allowed=false,),
        (allowed=true,),
        (allowed=true,),
    ]
    centroids=reshape(Float32[1.0,0.0],2,1)
    index=IVFIndex(centroids,[[1,2,3,4]])
    filter_index=build_bitset_index(metadata)
    query=Float32[1.0,0.0]

    results=search_ivf_prefilter(index,vectors,metadata,query;k=2,nprobe=1,metric=:dot,filter=(allowed=true,),filter_index=filter_index,)

    @test [result.index for result in results]==[3,4]
end

@testset "centroid training" begin
    vectors = Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.0 0.0 0.0
    ]
    centroids = train_centroids(vectors;nlists=2,iterations = 20,seed = 42,)

    @test size(centroids)==(2,2)

    left_distance = minimum(abs.(centroids[1,:].-(-0.95f0)),)
    right_distance = minimum(abs.(centroids[1,:].-0.95f0),)

    @test left_distance < 0.1f0
    @test right_distance < 0.1f0

    @test_throws ArgumentError train_centroids(vectors; nlists=5,)
end

@testset "build ivf index" begin
    vectors = Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.0 0.0 0.0
    ]
    index = build_ivf(vectors;nlists=2,iterations = 20,seed = 42,)

    @test size(index.centroids)==(2,2)
    @test length(index.lists)==2
    @test length(index.vector_norms)==4
    @test all(norm->norm>0,index.vector_norms)

    assigned_ids = sort(vcat(index.lists...))
    @test assigned_ids == [1,2,3,4]

    left_list = findfirst(list-> 1 in list ,index.lists)
    right_list = findfirst(list-> 3 in list, index.lists)

    @test left_list !== nothing
    @test right_list !==nothing
    @test 2 in index.lists[left_list]
    @test 4 in index.lists[right_list]
    @test left_list != right_list

    end

@testset "ivf search" begin
    vectors=Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.1 0.1 0.0
    ]
    metadata = [
        (name = "left-1",),
        (name = "left-2",),
        (name = "right-1",),
        (name = "right-2",),
    ]
    index = build_ivf(vectors; nlists=2,iterations=20,seed = 42,)
    query = Float32[-1,0]

    results = search_ivf(index,vectors,metadata,query;k=2,nprobe =1,)
    result_ids = [result.index for result in results]

    @test length(results)==2
    @test Set(result_ids)==Set([1,2])

    full_results = search_ivf(index,vectors,metadata,query;k=4,nprobe=2,)
    exact_results = search_exact(vectors,metadata,query;k=4,)

    full_ids = [result.index for result in full_results]
    exact_ids = [result.index for result in exact_results]

    @test test_recall_at_k(full_ids,exact_ids,4)==1.0
    @test_throws ArgumentError search_ivf(index,vectors,metadata,query;nprobe=0,)
    @test_throws ArgumentError search_ivf(index,vectors,metadata,query;nprobe=3,)

    end

@testset "ivf post filter search" begin
    vectors = Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.1 0.1 0.0
    ]
    metadata = [
        (name = "left-1",allowed = false),
        (name = "left-2",allowed = true),
        (name = "right-1",allowed = true),
        (name = "right-2",allowed = false),
    ]
    index = build_ivf(vectors;nlists=2,iterations=20,seed = 42,)
    query = Float32[-1,0]
    results = search_ivf_postfilter(index, vectors, metadata,query;k=2,nprobe = 1,filter = (allowed=true,),)

    @test length(results)==1
    @test results[1].metadata.name=="left-2"

    empty_results = search_ivf_postfilter(index,vectors,metadata,query;k=2,nprobe=1,filter =(name="missing",),)

    @test isempty(empty_results)
    end

@testset "filter aware ivf stats" begin
    vectors = Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.1 0.1 0.0
    ]
    metadata = [
        (side = "left",allowed = false),
        (side = "left",allowed = false),
        (side = "right",allowed = true),
        (side = "right",allowed = true),
    ]

    index = build_filter_aware_ivf(vectors,metadata;nlists=2,iterations =20,seed = 42,)

    left_list = findfirst(list->1 in list,index.ivf.lists,)
    right_list = findfirst(list->3 in list,index.ivf.lists,)

    @test left_list!==nothing
    @test right_list!==nothing

    @test estimate_list_filter_density(index,left_list,(allowed=true,),)==0.0
    @test estimate_list_filter_density(index,left_list,(side="left",),)==1.0
    @test estimate_list_filter_density(index,right_list,(allowed=true,),)==1.0

    query = Float32[-1,0]
    vector_only_ranking = rank_filter_aware_lists(index,query,(side="right",);nprobe =1,vector_weight=1.0,filter_weight=0.0,)

    @test vector_only_ranking==[left_list]

    filter_only_ranking = rank_filter_aware_lists(index,query,(side="right",);nprobe=1,vector_weight=0.0,filter_weight=1.0,)

    @test filter_only_ranking==[right_list]

    @test_throws ArgumentError rank_filter_aware_lists(index,query,(side="right",);nprobe=3,)

end

@testset "compound filter density" begin
    vectors=Float32[
        1.0 0.0
        0.0 1.0
    ]
    metadata=[
        (first=true,second=false,),
        (first=false,second=true,),
    ]
    ivf=IVFIndex(reshape(Float32[0.5,0.5],2,1),[[1,2]])
    index=build_filter_aware_ivf(ivf,metadata)

    @test estimate_list_filter_density(index,1,(first=true,))==0.5
    @test estimate_list_filter_density(index,1,(second=true,))==0.5
    @test estimate_list_filter_density(index,1,(first=true,second=true,))==0.0
end

@testset "filter aware ivf search" begin
    vectors = Float32[
        -1.0  -0.9   0.9   1.0
         0.0   0.1   0.1   0.0
    ]

    metadata = [
        (side = "left",),
        (side = "left",),
        (side = "right",),
        (side = "right",),
    ]

    index = build_filter_aware_ivf(vectors,metadata;nlists = 2,iterations = 20,seed = 42,)
    query = Float32[-1, 0]
    naive_results = search_ivf_postfilter(index.ivf,vectors,metadata,query; k = 2,nprobe = 1,filter = (side = "right",),)

    @test isempty(naive_results)

    aware_results = search_filter_aware_ivf(index,vectors,metadata,query;k = 2,nprobe = 1,filter = (side = "right",),vector_weight = 0.0,filter_weight = 1.0,)

    aware_ids = [result.index for result in aware_results]

    @test length(aware_results) == 2
    @test Set(aware_ids) == Set([3, 4])
    @test all(result.metadata.side == "right"
        for result in aware_results
    )
end

@testset "adaptive filter aware ivf" begin
    vectors=Float32[
        -1.0 -0.9 0.9 1.0
        0.0 0.1 0.1 0.0
    ]
    metadata=[
        (side="left",),
        (side="left",),
        (side="right",),
        (side="right",),
    ]
    index=build_filter_aware_ivf(vectors,metadata;nlists=2,iterations=20,seed=42,)
    query=Float32[-1.0,0.0]

    fixed=search_filter_aware_ivf(index,vectors,metadata,query;k=2,nprobe=1,filter=(side="right",),vector_weight=1.0,filter_weight=0.0,)
    adaptive=search_filter_aware_ivf(index,vectors,metadata,query;k=2,nprobe=1,filter=(side="right",),adaptive=true,max_nprobe=2,candidate_multiplier=1.0,vector_weight=1.0,filter_weight=0.0,)
    selection=Sen.select_filter_aware_candidates(index,metadata,query,(side="right",);k=2,nprobe=1,adaptive=true,max_nprobe=2,candidate_multiplier=1.0,vector_weight=1.0,filter_weight=0.0,)

    @test isempty(fixed)
    @test Set(result.index for result in adaptive)==Set([3,4])
    @test length(selection.selected_lists)==2
    @test length(selection.visited_indices)==2
    @test length(selection.candidate_indices)==2
end

@testset "spherical ivf geometry" begin
    dataset=generate_test_clusters(240,12,6;noise=0.04,seed=42,)
    index=build_ivf(dataset.vectors;nlists=6,iterations=15,seed=42,metric=:cosine,restarts=3,training_count=180,)
    query=@view dataset.centers[:,1]
    bounds=list_score_upper_bounds(index,query)

    @test index.metric===:cosine
    @test length(index.list_radii)==6
    @test all(list_index->isapprox(vector_norm(@view index.centroids[:,list_index]),1.0f0;atol=1.0f-5,),1:6)

    for list_index in eachindex(index.lists)
        centroid=@view index.centroids[:,list_index]

        for vector_index in index.lists[list_index]
            vector=@view dataset.vectors[:,vector_index]
            angle=acos(clamp(cosine_similarity(centroid,vector),-1.0f0,1.0f0))
            score=cosine_similarity(query,vector)
            @test angle<=index.list_radii[list_index]+1.0f-5
            @test score<=bounds[list_index]+1.0f-5
        end
    end

    @test_throws ArgumentError train_centroids(dataset.vectors;nlists=6,restarts=0,)
    @test_throws ArgumentError train_centroids(dataset.vectors;nlists=6,training_count=5,)
end

@testset "list filter postings" begin
    vectors=Float32[
        1.0 0.9 0.0 0.0
        0.0 0.1 1.0 0.9
    ]
    metadata=[
        (group=:a,allowed=true,),
        (group=:a,allowed=false,),
        (group=:b,allowed=true,),
        (group=:b,allowed=false,),
    ]
    ivf=IVFIndex(Float32[1.0 0.0;0.0 1.0],[[1,2],[3,4]])
    index=build_filter_aware_ivf(ivf,metadata)

    @test estimate_list_filter_count(index,1,(group=:a,))==2
    @test estimate_list_filter_count(index,1,(group=:a,allowed=true,))==1
    @test collect_filtered_list_candidates(index,[1,2],(allowed=true,))==[1,3]
end

@testset "safe bound search" begin
    dataset=generate_test_clusters(400,12,4;noise=0.02,seed=52,)
    metadata=fill((selected=true,),400)
    ivf=build_ivf(dataset.vectors;nlists=4,iterations=20,seed=52,metric=:cosine,restarts=4,)
    index=build_filter_aware_ivf(ivf,metadata)
    query=@view dataset.centers[:,1]
    truth=search_exact(dataset.vectors,metadata,query;k=10,filter=(selected=true,),)
    stats=search_filter_aware_bound_with_stats(index,dataset.vectors,metadata,query;filter=(selected=true,),k=10,minimum_nprobe=1,max_nprobe=4,)

    @test test_recall_at_k([result.index for result in stats.results],[result.index for result in truth],10)==1.0
    @test stats.exact
    @test stats.stopped_by_bound
    @test stats.probed_lists<4
    @test stats.visited==stats.scored
end
