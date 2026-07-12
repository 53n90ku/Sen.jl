function matches_filter(metadata::NamedTuple,filter::NamedTuple)
    for(field,expected_value) in pairs(filter)
        hasproperty(metadata,field)||return false

        actual_value=getproperty(metadata,field)
        actual_value==expected_value||return false
    end
    return true
end