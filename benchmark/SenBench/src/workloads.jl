using Random
using Dates

const EXPRESSION_FAMILIES=(:eq,:in,:range_numeric,:range_date,:and,:or,:not)
const EXPRESSION_BUCKETS=(rare=0.01,medium=0.10,broad=0.50,)

struct ExpressionWorkloadCase
    name::String
    family::Symbol
    bucket::Symbol
    workload::Symbol
    filter::FilterExpr
    target_selectivity::Float64
    actual_selectivity::Float64
end

function generate_filter_order(vectors::AbstractMatrix;workload::Symbol=:random,seed::Int=42,skew::Float64=4.0,)
    dim,count=size(vectors)
    count>0||throw(ArgumentError("vectors cannot be empty"))
    workload in (:random,:correlated,:anticorrelated,:skewed)||throw(ArgumentError("workload must be random, correlated, anticorrelated or skewed"))
    skew>0||throw(ArgumentError("skew must be positive"))

    rng=MersenneTwister(seed)
    workload===:random&&return randperm(rng,count)

    direction=randn(rng,Float32,dim)
    direction./=sqrt(sum(abs2,direction))
    projections=Vector{Float64}(undef,count)

    for vector_index in 1:count
        vector=@view vectors[:,vector_index]
        projections[vector_index]=dot_similarity(direction,vector)
    end

    if workload===:correlated
        return collect(sortperm(projections;rev=true,))
    elseif workload===:anticorrelated
        return collect(sortperm(projections;rev=false,))
    end

    scores=[skew*projections[index]+randn(rng) for index in 1:count]
    return collect(sortperm(scores;rev=true,))
end

function generate_filter_metadata(vectors::AbstractMatrix,selectivity::Float64;workload::Symbol=:random,seed::Int=42,skew::Float64=4.0,)
    order=generate_filter_order(vectors;workload=workload,seed=seed,skew=skew,)
    return generate_selectivity_metadata(order,selectivity)
end

function expression_target_count(count::Int,selectivity::Float64)
    count>0||throw(ArgumentError("vector count must be positive"))
    0.0<selectivity<=1.0||throw(ArgumentError("selectivity must be between zero and one"))
    return clamp(round(Int,count*selectivity),1,count)
end

function generate_expression_metadata(vectors::AbstractMatrix;workload::Symbol=:random,seed::Int=42,skew::Float64=4.0,)
    _,count=size(vectors)
    order=generate_filter_order(vectors;workload=workload,seed=seed,skew=skew,)
    positions=Vector{Int}(undef,count)

    for(position,vector_index) in enumerate(order)
        positions[vector_index]=position
    end

    rare_count=expression_target_count(count,EXPRESSION_BUCKETS.rare)
    medium_count=expression_target_count(count,EXPRESSION_BUCKETS.medium)
    broad_count=expression_target_count(count,EXPRESSION_BUCKETS.broad)
    rare_half=rare_count÷2
    medium_half=medium_count÷2
    broad_half=broad_count÷2
    start_date=Date(2020,1,1)
    metadata=Vector{NamedTuple}(undef,count)

    for vector_index in 1:count
        position=positions[vector_index]
        segment=clamp(ceil(Int,100*position/count),1,100)
        metadata[vector_index]=(
            rank=position,
            created=start_date+Day(position-1),
            segment=segment,
            rare_match=position<=rare_count,
            rare_left=position<=rare_half,
            rare_right=rare_half<position<=rare_count,
            medium_match=position<=medium_count,
            medium_left=position<=medium_half,
            medium_right=medium_half<position<=medium_count,
            broad_match=position<=broad_count,
            broad_left=position<=broad_half,
            broad_right=broad_half<position<=broad_count,
        )
    end

    return metadata
end

function expression_bucket_selectivity(bucket::Symbol)
    bucket in keys(EXPRESSION_BUCKETS)||throw(ArgumentError("unknown expression selectivity bucket"))
    return Float64(getproperty(EXPRESSION_BUCKETS,bucket))
end

function expression_bucket_fields(bucket::Symbol)
    expression_bucket_selectivity(bucket)
    return(
        match=Symbol(bucket,:_match),
        left=Symbol(bucket,:_left),
        right=Symbol(bucket,:_right),
    )
end

function build_expression_filter(family::Symbol,bucket::Symbol,count::Int)
    family in EXPRESSION_FAMILIES||throw(ArgumentError("unknown expression family"))
    selectivity=expression_bucket_selectivity(bucket)
    target_count=expression_target_count(count,selectivity)
    fields=expression_bucket_fields(bucket)

    family===:eq&&return Eq(fields.match,true)
    family===:in&&return In(:segment,collect(1:clamp(round(Int,100*selectivity),1,100)))
    family===:range_numeric&&return Range(:rank,1,target_count)
    family===:range_date&&return Range(:created,Date(2020,1,1),Date(2020,1,1)+Day(target_count-1))
    family===:and&&return And(Eq(fields.match,true),Range(:rank,1,target_count))
    family===:or&&return Or(Eq(fields.left,true),Eq(fields.right,true))
    return Not(Eq(fields.match,false))
end

function build_expression_workload_case(metadata::AbstractVector,filter::FilterExpr;name::AbstractString,family::Symbol,bucket::Symbol,workload::Symbol,)
    workload in (:random,:correlated,:anticorrelated,:skewed,:natural)||throw(ArgumentError("unknown expression workload"))
    family in EXPRESSION_FAMILIES||throw(ArgumentError("unknown expression family"))
    target_selectivity=expression_bucket_selectivity(bucket)
    index=build_bitset_index(metadata)
    actual_selectivity=estimate_selectivity(index,filter)
    return ExpressionWorkloadCase(String(name),family,bucket,workload,filter,target_selectivity,actual_selectivity)
end

function generate_expression_workload(vectors::AbstractMatrix;workload::Symbol=:random,seed::Int=42,skew::Float64=4.0,families=EXPRESSION_FAMILIES,buckets=keys(EXPRESSION_BUCKETS),)
    metadata=generate_expression_metadata(vectors;workload=workload,seed=seed,skew=skew,)
    resolved_families=Symbol.(collect(families))
    resolved_buckets=Symbol.(collect(buckets))
    cases=ExpressionWorkloadCase[]

    for bucket in resolved_buckets
        for family in resolved_families
            filter=build_expression_filter(family,bucket,length(metadata))
            name="$(workload)-$(bucket)-$(family)"
            push!(cases,build_expression_workload_case(metadata,filter;name=name,family=family,bucket=bucket,workload=workload,))
        end
    end

    return(metadata=metadata,cases=cases,)
end
