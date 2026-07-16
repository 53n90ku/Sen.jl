function batch_queries_per_second(query_count::Int, mean_ms::Float64)
    query_count>0||throw(ArgumentError("query count must be positive"))
    mean_ms>=0||throw(ArgumentError("mean latency cannot be negative"))
    iszero(mean_ms)&&return Inf
    return query_count/(mean_ms/1000)
end

function run_batch_search_benchmark(
    db::VectorDB,
    queries::AbstractMatrix{<:Real};
    repetitions::Int = 10,
    workers::Int = Threads.nthreads(:default),
    parallel_threshold::Int = 4,
    kwargs...,
)
    repetitions>0||throw(ArgumentError("repetitions must be positive"))
    workers>0||throw(ArgumentError("workers must be positive"))
    query_count=size(queries, 2)
    query_count>0||throw(ArgumentError("batch benchmark requires atleast one query"))

    serial=measure_latency(
        ()->search(
            db,
            queries;
            parallel = false,
            workers = 1,
            parallel_threshold = parallel_threshold,
            kwargs...,
        );
        repetitions = repetitions,
    )
    parallel=measure_latency(
        ()->search(
            db,
            queries;
            parallel = true,
            workers = workers,
            parallel_threshold = parallel_threshold,
            kwargs...,
        );
        repetitions = repetitions,
    )
    serial_qps=batch_queries_per_second(query_count, serial.summary.mean_ms)
    parallel_qps=batch_queries_per_second(query_count, parallel.summary.mean_ms)

    return (
        query_count = query_count,
        workers = Sen.resolve_database_batch_workers(
            query_count;
            parallel = true,
            workers = workers,
            parallel_threshold = parallel_threshold,
        ),
        serial = (latency = serial.summary, queries_per_second = serial_qps),
        parallel = (latency = parallel.summary, queries_per_second = parallel_qps),
        throughput_speedup = parallel_qps/serial_qps,
    )
end
