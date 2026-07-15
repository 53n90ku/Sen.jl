abstract type SearchStrategy end

struct ExactStrategy <: SearchStrategy
end

struct IVFStrategy <: SearchStrategy
end

struct PreFilterExactStrategy <: SearchStrategy
end

struct IVFPostFilterStrategy <: SearchStrategy
end

struct IVFPreFilterStrategy <: SearchStrategy
end

struct FilterAwareIVFStrategy <: SearchStrategy
end

struct BoundedFilterAwareIVFStrategy <: SearchStrategy
end
