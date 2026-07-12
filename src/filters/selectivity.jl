function  estimate_selectivity(index::BitsetIndex,filter::NamedTuple,)::Float64
    index.count==0&&return 0.0
    mask = evaluate_filter(index,filter)
    matching_count=count(identity,mask)
    return matching_count/index.count    
end