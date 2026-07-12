function create_db(path::String; dim::Int, metric::Symbol=:cosine)
    dim>0|| throw(ArgumentError("dimension must be positive"))
    metric in (:cosine, :dot)|| throw(ArgumentError("metric must be :cosine or :dot"))

    return VectorDB(path, dim,metric)
end