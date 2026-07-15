function measure_vector_load(path::AbstractString,mmap::Bool;repetitions::Int,mmap_threshold_bytes::Int)
    GC.gc()
    first=@timed load_vector_store(path;mmap=mmap,mmap_threshold_bytes=mmap_threshold_bytes,)
    first_store=first.value
    mapped=is_mapped(first_store)
    dim=first_store.dim
    count=length(first_store)
    release_vector_store_mapping!(first_store)

    times_ns=Vector{Int}(undef,repetitions)
    allocated_bytes=Vector{Int}(undef,repetitions)

    for repetition in 1:repetitions
        measurement=@timed load_vector_store(path;mmap=mmap,mmap_threshold_bytes=mmap_threshold_bytes,)
        store=measurement.value
        times_ns[repetition]=round(Int,measurement.time*1_000_000_000)
        allocated_bytes[repetition]=measurement.bytes
        release_vector_store_mapping!(store)
    end

    return(
        mapped=mapped,
        dim=dim,
        count=count,
        first_ms=first.time*1000,
        warm=latency_summary(times_ns),
        mean_allocated_bytes=sum(allocated_bytes)/length(allocated_bytes),
        maximum_allocated_bytes=maximum(allocated_bytes),
    )
end

function run_vector_load_benchmark(path::AbstractString;repetitions::Int=5,mmap_threshold_bytes::Int=0,)
    repetitions>0||throw(ArgumentError("repetitions must be positive"))
    mmap_threshold_bytes>=0||throw(ArgumentError("mmap threshold bytes cannot be negative"))
    vector_path=joinpath(path,"vectors.bin")
    isfile(vector_path)||throw(ArgumentError("vector file does not exist"))

    copied=measure_vector_load(path,false;repetitions=repetitions,mmap_threshold_bytes=mmap_threshold_bytes,)
    mapped=measure_vector_load(path,true;repetitions=repetitions,mmap_threshold_bytes=mmap_threshold_bytes,)
    copied.dim==mapped.dim||error("load benchmark dimensions do not match")
    copied.count==mapped.count||error("load benchmark counts do not match")

    return(
        file_bytes=filesize(vector_path),
        dim=copied.dim,
        count=copied.count,
        copied=copied,
        mapped=mapped,
        warm_speedup=copied.warm.p50_ms/mapped.warm.p50_ms,
        allocation_reduction=1-mapped.mean_allocated_bytes/max(1,copied.mean_allocated_bytes),
    )
end
