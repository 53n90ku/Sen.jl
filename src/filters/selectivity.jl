function estimate_selectivity(index::MetadataIndex,filter::Union{NamedTuple,FilterExpr};metadata::Union{Nothing,AbstractVector}=nothing,)::Float64
    index.count==0&&return 0.0
    metadata===nothing||length(metadata)==index.count||throw(DimensionMismatch("metadata count doesnt match metadata index"))
    expression=normalize_filter(filter)

    matching_count=if supports_indexed_filter(index,expression)
        count(evaluate_filter(index,expression))
    else
        metadata===nothing&&throw(ArgumentError("filter selectivity requires metadata because this expression is not supported by the metadata index"))
        count(row->matches_filter(row,expression),metadata)
    end

    return matching_count/index.count
end
