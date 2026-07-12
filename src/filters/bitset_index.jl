struct BitsetIndex
    masks::Dict{Tuple{Symbol,Any},BitVector}
    count::Int
end

function build_bitset_index(metadata::AbstractVector)
    count = length(metadata)
    masks=Dict{Tuple{Symbol,Any},BitVector}()

    for(index,row) in enumerate(metadata)
        for(field,value) in pairs(row)
            key = (field, value)
            mask = get!(masks,key) do 
                falses(count)
            end
            mask[index]=true
        end
    end
    return BitsetIndex(masks,count)
end

function evaluate_filter(index::BitsetIndex,filter::NamedTuple)
    result = trues(index.count)

    for(field,expected_value) in pairs(filter)
        key = (field,expected_value)
        mask = get(index.masks,key,nothing)

        mask===nothing&& return falses(index.count)
        result.&=mask
    end
    return result
end
            

