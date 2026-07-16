using Test

function write_test_fvecs(path,vectors)
    open(path,"w") do io
        for index in axes(vectors,2)
            write(io,Int32(size(vectors,1)))
            write(io,Float32.(vectors[:,index]))
        end
    end
end

@testset "real dataset manifest" begin
    mktempdir() do path
        file_path=joinpath(path,"vectors.fvecs")
        write_test_fvecs(file_path,Float32[1 0;0 1])
        manifest=create_dataset_manifest(path;name="fixture",source="local",license="test",vector_count=2,dimension=2,query_count=1,filenames=["vectors.fvecs"],)
        manifest_path=save_dataset_manifest(joinpath(path,"manifest.toml"),manifest)
        loaded=load_dataset_manifest(manifest_path)

        @test loaded.preprocessing_hash==manifest.preprocessing_hash
        @test validate_dataset_manifest(loaded,path).vector_count==2

        open(file_path,"a") do io
            write(io,UInt8(0))
        end
        @test_throws ArgumentError validate_dataset_manifest(loaded,path)
    end
end

@testset "real vector formats" begin
    mktempdir() do path
        fvecs_path=joinpath(path,"vectors.fvecs")
        vectors=Float32[3 0 -1;4 2 0]
        write_test_fvecs(fvecs_path,vectors)

        @test fvecs_info(fvecs_path).count==3
        @test load_fvecs_indices(fvecs_path,[3,1];normalize=false,)==vectors[:,[3,1]]
        @test load_fvecs_indices(fvecs_path,[1])[:,1]≈Float32[0.6,0.8]

        truth_path=save_ivecs(joinpath(path,"truth.ivecs"),[[1,3],[2,1]])
        @test ivecs_info(truth_path).dimension==2
        @test load_ivecs_indices(truth_path,[2,1])==[[2,1],[1,3]]
    end
end

@testset "ranged vector conversion" begin
    mktempdir() do path
        source=joinpath(path,"chunk.fvecs")
        output=joinpath(path,"vectors.f32")
        vectors=Float32[3 0;4 2]
        write_test_fvecs(source,vectors)
        SenBench.append_normalized_fvecs!(output,source,2,2)
        mapped=load_f32_matrix(output,2,2)

        @test mapped[:,1]≈Float32[0.6,0.8]
        @test mapped[:,2]≈Float32[0.0,1.0]
        @test size(mapped)==(2,2)
    end
end

@testset "real metadata and splits" begin
    mktempdir() do path
        metadata_path=joinpath(path,"metadata.jsonl")
        query_path=joinpath(path,"queries.jsonl")
        open(metadata_path,"w") do io
            println(io,"{\"main_categories\":[\"cs\"],\"number_of_sub_categories\":1,\"has_comments\":true,\"number_of_versions\":2,\"number_of_authors\":3,\"license\":null}")
            println(io,"{\"main_categories\":[\"math\"],\"number_of_sub_categories\":2,\"has_comments\":false,\"number_of_versions\":1,\"number_of_authors\":1,\"license\":\"cc-by-4.0\"}")
        end
        open(query_path,"w") do io
            for label in [1,2,1,3,2,4,1,2]
                println(io,"{\"label\":$(label)}")
            end
        end

        metadata=load_arxiv_metadata(metadata_path)
        labels=load_arxiv_query_labels(query_path)
        split=stratified_query_split(labels;train_count=3,heldout_count=3,seed=42,)

        @test metadata[1].label==1
        @test metadata[1].license=="unspecified"
        @test labels==[1,2,1,3,2,4,1,2]
        @test isempty(intersect(Set(split.train_indices),Set(split.heldout_indices)))
        @test split==stratified_query_split(labels;train_count=3,heldout_count=3,seed=42,)
        @test dataset_selectivities(metadata,[(label=1,),(label=2,)])==[0.5,0.5]
    end
end

@testset "sealed real query partitions" begin
    labels=repeat(1:4,inner=4)
    partitions=stratified_query_partitions(labels;development_count=6,validation_count=4,confirmation_count=4,seed=42,)

    @test length(partitions.development_indices)==6
    @test length(partitions.validation_indices)==4
    @test length(partitions.confirmation_indices)==4
    @test isempty(intersect(Set(partitions.development_indices),Set(partitions.validation_indices)))
    @test isempty(intersect(Set(vcat(partitions.development_indices,partitions.validation_indices)),Set(partitions.confirmation_indices)))
    @test partitions==stratified_query_partitions(labels;development_count=6,validation_count=4,confirmation_count=4,seed=42,)

    balanced=stratified_query_partitions(labels;development_count=6,validation_count=4,confirmation_count=4,seed=42,strata=repeat(1:2,inner=8),)
    @test Set(repeat(1:2,inner=8)[balanced.validation_indices])==Set([1,2])
    @test Set(repeat(1:2,inner=8)[balanced.confirmation_indices])==Set([1,2])

    scarce_strata=vcat(fill(1,3),fill(2,13))
    jointly_balanced=balanced_query_partitions(labels;development_count=6,validation_count=4,confirmation_count=4,seed=42,strata=scarce_strata,)
    @test 1 in scarce_strata[jointly_balanced.development_indices]
    @test 1 in scarce_strata[jointly_balanced.validation_indices]
    @test 1 in scarce_strata[jointly_balanced.confirmation_indices]

    mktempdir() do path
        base=joinpath(path,"queries")
        saved=save_query_partitions(base,partitions;dataset_hash="dataset",query_count=length(labels),)
        public=load_query_partitions(base;dataset_hash="dataset",)

        @test isfile(saved.public)
        @test isfile(saved.sealed)
        @test isempty(public.confirmation_indices)
        @test query_partition_indices(public,:development)==partitions.development_indices
        @test_throws ArgumentError query_partition_indices(public,:confirmation)

        sealed=load_query_partitions(base;dataset_hash="dataset",allow_confirmation=true,)
        @test query_partition_indices(sealed,:confirmation;allow_confirmation=true,)==partitions.confirmation_indices
        @test_throws ArgumentError load_query_partitions(base;dataset_hash="changed",)
    end
end

@testset "IVF quality diagnostics" begin
    vectors=Float32[
        1.0 0.9 0.8 0.0 0.1 0.2
        0.0 0.1 0.2 1.0 0.9 0.8
    ]
    SenBench.normalize_columns!(vectors)
    index=build_ivf(vectors;nlists=2,iterations=4,restarts=2,seed=42,)
    diagnostics=ivf_index_diagnostics(index,vectors)
    queries=vectors[:,[1,4]]
    truth=[[1,2],[4,5]]
    metadata=NamedTuple[(label=index<=3 ? 1 : 2,) for index in 1:6]
    filters=[(label=1,),(label=2,)]
    routing=ivf_routing_diagnostics(index,queries,truth;metadata=metadata,filters=filters,)
    curve=ivf_probe_curve(index,queries,truth,[1,2];metadata=metadata,filters=filters,)
    aggregate=aggregate_probe_curve(curve)

    @test diagnostics.empty_lists==0
    @test diagnostics.vector_count==6
    @test diagnostics.mean_quantization_loss>=0
    @test length(routing)==2
    @test all(row->row.nprobe_100>=1,routing)
    @test length(curve)==4
    @test all(row->row.truth_list_recall==1.0,filter(row->row.nprobe==2,curve))
    @test any(row->row.bucket===:all&&row.nprobe==2&&row.recall==1.0,aggregate)
    @test probe_schedule(9)==[1,2,3,4,5,7,8,9]
    @test 384 in probe_schedule(1536)
end

@testset "resumable IVF quality sweep" begin
    vectors=Float32[
        1.0 0.9 0.8 0.7 -1.0 -0.9 -0.8 -0.7
        0.0 0.1 0.2 0.3 0.0 0.1 0.2 0.3
    ]
    SenBench.normalize_columns!(vectors)
    metadata=NamedTuple[(label=index<=4 ? 1 : 2,) for index in 1:8]
    train_queries=vectors[:,[1,2,5,6]]
    heldout_queries=vectors[:,[3,7]]
    train_filters=[(label=1,),(label=1,),(label=2,),(label=2,)]
    heldout_filters=[(label=1,),(label=2,)]
    train_truth=compute_groundtruth(vectors,metadata,train_queries;k=2,filters=train_filters,)
    heldout_truth=compute_groundtruth(vectors,metadata,heldout_queries;k=2,filters=heldout_filters,)
    manifest=DatasetManifest("fixture","local","main","test",8,2,6,:cosine,42,Dict("fixture"=>"abc"),Dict("fixture"=>1),"abc")
    dataset=ArxivFANNSDataset(manifest,"fixture",vectors,metadata,train_queries,train_filters,train_truth,[1,2,3,4],heldout_queries,heldout_filters,heldout_truth,[5,6])

    mktempdir() do path
        first_result=run_ivf_quality_configuration(dataset,path;nlists=2,training_count=8,iterations=3,restarts=1,)
        second_result=run_ivf_quality_configuration(dataset,path;nlists=2,training_count=8,iterations=3,restarts=1,)

        @test first_result.status===:complete
        @test second_result.status===:cached
        @test isfile(joinpath(first_result.path,"checkpoint.toml"))
        @test isfile(joinpath(first_result.path,"aggregate.tsv"))
        stored_index=SenBench.load_ivf_index(joinpath(first_result.path,"index"))
        validation=evaluate_hybrid_validation(dataset,stored_index;nprobe=2,k=2,repetitions=1,rare_threshold=0.1,)
        saved=save_hybrid_validation(joinpath(path,"validation"),validation)

        @test validation.recall_gate
        @test validation.hybrid.summary.average_recall==1.0
        @test validation.recall_interval.lower==1.0
        @test validation.recall_interval.upper==1.0
        @test isfile(saved.summary)
    end
end

@testset "groundtruth cache" begin
    mktempdir() do path
        cache_path=joinpath(path,"truth")
        rows=[[1,2],[2,3]]
        saved=save_groundtruth_cache(cache_path,rows;dataset_hash="abc",query_indices=[4,7],k=2,filter_name="em",)

        @test isfile(saved.data)
        @test load_groundtruth_cache(cache_path;dataset_hash="abc",query_indices=[4,7],k=2,filter_name="em",)==rows
        @test_throws ArgumentError load_groundtruth_cache(cache_path;dataset_hash="changed",query_indices=[4,7],k=2,filter_name="em",)
    end
end

@testset "blockwise exact groundtruth" begin
    vectors=Float32[
        1.0 0.9 0.0 -1.0 -0.9 0.0
        0.0 0.1 1.0 0.0 0.1 -1.0
    ]
    SenBench.normalize_columns!(vectors)
    metadata=NamedTuple[(label=index<=3 ? 1 : 2,) for index in 1:6]
    queries=Float32[1.0 -1.0 0.0;0.0 0.0 1.0]
    filters=[(label=1,),(label=2,),(label=1,)]
    expected=compute_groundtruth(vectors,metadata,queries;k=2,filters=filters,)
    actual=compute_blockwise_groundtruth(vectors,metadata,queries;k=2,filters=filters,block_size=2,)

    @test actual==expected
end

@testset "real benchmark protocol" begin
    vectors=Float32[
        1.0 0.9 0.8 0.7 -1.0 -0.9 -0.8 -0.7
        0.0 0.1 0.2 0.3 0.0 0.1 0.2 0.3
    ]
    SenBench.normalize_columns!(vectors)
    metadata=NamedTuple[(label=index<=4 ? 1 : 2,main_category="test",has_comments=false,version_count=1,author_count=1,license="test") for index in 1:8]
    train_queries=Float32[1.0 -1.0 0.8;0.0 0.0 0.2]
    heldout_queries=Float32[0.9 -0.9 0.7;0.1 0.1 0.3]
    SenBench.normalize_columns!(train_queries)
    SenBench.normalize_columns!(heldout_queries)
    train_filters=NamedTuple[(label=1,),(label=2,),(label=1,)]
    heldout_filters=NamedTuple[(label=1,),(label=2,),(label=1,)]
    train_truth=compute_groundtruth(vectors,metadata,train_queries;k=2,filters=train_filters,)
    heldout_truth=compute_groundtruth(vectors,metadata,heldout_queries;k=2,filters=heldout_filters,)
    manifest=DatasetManifest("fixture","local","main","test",8,2,6,:cosine,42,Dict("fixture"=>"abc"),Dict("fixture"=>1),"abc")
    dataset=ArxivFANNSDataset(manifest,"fixture",vectors,metadata,train_queries,train_filters,train_truth,[1,2,3],heldout_queries,heldout_filters,heldout_truth,[4,5,6])
    result=run_real_benchmark(dataset;name="fixture",nlists=2,nprobes=[1,2],k=2,target_recall=0.5,selection_margin=0.0,probe_safety_factor=1.0,repetitions=1,iterations=3,restarts=1,training_count=8,candidate_multiplier=2.0,)

    @test Set(keys(result.method_results))==Set([:exact,:ivf_prefilter,:ivf_postfilter,:filter_aware,:filter_aware_bound,:auto])
    @test result.method_results[:exact].summary.average_recall==1.0
    @test result.best_approximate_recall>=0.5
    @test length(result.raw_rows)==18
    @test all(row->row.dataset=="fixture",result.raw_rows)
    @test length(real_benchmark_summary_rows(result))==6

    mktempdir() do path
        saved=save_real_benchmark(path,result)
        @test all(isfile,[saved.raw,saved.summary,saved.dataset,saved.environment])
    end
end
