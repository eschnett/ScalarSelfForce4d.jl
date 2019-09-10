"""
Efficient small vectors
"""
module Vecs

using LinearAlgebra



export Vec
struct Vec{D, T} <: DenseArray{T, 1}
    elts::NTuple{D, T}
end

# Vec{D}(x::T) where {D,T} = Vec{D,T}(ntuple(d->x, D))
# Vec{D,T}(x::T) where {D,T} = Vec{D,T}(ntuple(d->x, D))

# vec1(::Val{D}, x::T) where {D, T} = Vec{D,T}(ntuple(d -> x, D))



# Vec interacts with scalars

function Base.promote_rule(::Type{Vec{D,T}}, ::Type{T}) where {D, T<:Number}
    Vec{D,T}
end
function Base.convert(::Type{Vec{D,T}}, x::T) where {D, T<:Number}
    Vec{D,T}(ntuple(d -> x, D))
end



# Vec is a collection

function Base.eltype(x::Vec{D,T})::Type where {D, T}
    T
end
function Base.length(x::Vec{D,T})::Int where {D, T}
    D
end
function Base.ndims(x::Vec{D,T})::Int where {D, T}
    1
end
function Base.size(x::Vec{D,T})::Tuple{Int} where {D, T}
    (D,)
end
function Base.size(x::Vec{D,T}, d)::Int where {D, T}
    D
end

function Base.getindex(x::Vec{D,T}, d::Integer)::T where {D, T}
    getindex(x.elts, d)
end



# Vec are a vector space

function Base.zeros(::Type{Vec{D,T}})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(ntuple(d -> T(0), D))
end
# function Base.zero(::Type{Vec{D,T}})::Vec{D,T} where {D, T<:Number}
#     Vec{D,T}(ntuple(d -> T(0), D))
# end
# function Base.one(::Type{Vec{D,T}})::Vec{D,T} where {D, T<:Number}
#     Vec{D,T}(ntuple(d -> T(1), D))
# end

function Base.:+(x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(.+(x.elts))
end
function Base.:-(x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(.-(x.elts))
end
function Base.inv(x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(inv.(x.elts))
end

function Base.:+(x::Vec{D,T}, y::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(x.elts .+ y.elts)
end
function Base.:-(x::Vec{D,T}, y::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(x.elts .- y.elts)
end

function Base.:*(a::Number, x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(T(a) .* x.elts)
end
function Base.:*(x::Vec{D,T}, a::Number)::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(x.elts .* T(a))
end
function Base.:\(a::Number, x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(T(a) .\ x.elts)
end
function Base.:/(x::Vec{D,T}, a::Number)::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(x.elts ./ T(a))
end

const ArithOp = Union{typeof(+), typeof(-), typeof(*), typeof(/), typeof(\)}
function Base.broadcasted(op::ArithOp,
                          x::Vec{D,T}, y::Vec{D,T})::Vec{D,T} where
        {D, T<:Number}
    Vec{D,T}(ntuple(d -> op(x.elts[d], y.elts[d]), D))
end
function Base.broadcasted(op::ArithOp, x::Vec{D,T}, a::Number)::Vec{D,T} where
        {D, T<:Number}
    Vec{D,T}(ntuple(d -> op(x.elts[d], T(a)), D))
end
function Base.broadcasted(op::ArithOp, a::Number, x::Vec{D,T})::Vec{D,T} where
        {D, T<:Number}
    Vec{D,T}(ntuple(d -> op(T(a), x.elts[d]), D))
end

const CmpOp = Union{typeof(==), typeof(!=),
                    typeof(<), typeof(<=), typeof(>), typeof(>=)}
function Base.broadcasted(op::CmpOp,
                          x::Vec{D,T}, y::Vec{D,T})::Vec{D,Bool} where
        {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> op(x.elts[d], y.elts[d]), D))
end
function Base.broadcasted(op::CmpOp, x::Vec{D,T}, a::Number)::Vec{D,Bool} where
        {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> op(x.elts[d], T(a)), D))
end
function Base.broadcasted(op::CmpOp, a::Number, x::Vec{D,T})::Vec{D,Bool} where
        {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> op(T(a), x.elts[d]), D))
end

function Base.:~(x::Vec{D,Bool})::Vec{D,Bool} where {D}
    Vec{D,Bool}(.~(x.elts))
end
function Base.:&(x::Vec{D,Bool}, y::Vec{D,Bool})::Vec{D,Bool} where {D}
    Vec{D,Bool}(x.elts .& y.elts)
end
function Base.:|(x::Vec{D,Bool}, y::Vec{D,Bool})::Vec{D,Bool} where {D}
    Vec{D,Bool}(x.elts .& y.elts)
end
function Base.:!(x::Vec{D,Bool})::Vec{D,Bool} where {D}
    ~x
end
# function Base.:&&(x::Vec{D,Bool}, y::Vec{D,Bool})::Vec{D,Bool} where {D}
#     x & y
# end
# function Base.:||(x::Vec{D,Bool}, y::Vec{D,Bool})::Vec{D,Bool} where {D}
#     x | y
# end

function Base.all(x::Vec{D,Bool})::Bool where {D}
    all(x.elts)
end
function Base.any(x::Vec{D,Bool})::Bool where {D}
    any(x.elts)
end
function Base.max(x::Vec{D,T})::T where {D, T<:Number}
    max(x.elts)
end
function Base.min(x::Vec{D,T})::T where {D, T<:Number}
    min(x.elts)
end
function Base.prod(x::Vec{D,T})::T where {D, T<:Number}
    prod(x.elts)
end
function Base.sum(x::Vec{D,T})::T where {D, T<:Number}
    sum(x.elts)
end

function LinearAlgebra.norm(x::Vec{D,T}, p::Real=2) where {D, T<:Number}
    norm(x.elts, p)
end

end