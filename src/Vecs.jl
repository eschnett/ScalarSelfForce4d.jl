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
function Base.convert(::Type{Vec{D,T}}, x::T) where {D, T}
    Vec{D,T}(ntuple(d -> x, D))
end

function Base.promote_rule(::Type{Vec{D,T}}, ::Type{Vec{D,U}}) where {D, T, U}
    R = promote_type(T, U)
    Vec{D,R}
end
function Base.convert(::Type{Vec{D,R}}, x::Vec{D,T}) where {D, T, R}
    Vec{D,R}(ntuple(d -> R(x[d]), D))
end



# Vec is a collection

function Base.eltype(::Type{Vec{D,T}})::Type where {D, T}
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
function Base.ones(::Type{Vec{D,T}})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(ntuple(d -> T(1), D))
end
function Base.falses(::Type{Vec{D,Bool}})::Vec{D,Bool} where {D}
    zeros(Vec{D,Bool})
end
function Base.trues(::Type{Vec{D,Bool}})::Vec{D,Bool} where {D}
    ones(Vec{D,Bool})
end
# function Base.zero(::Type{Vec{D,T}})::Vec{D,T} where {D, T<:Number}
#     Vec{D,T}(ntuple(d -> T(0), D))
# end
# function Base.one(::Type{Vec{D,T}})::Vec{D,T} where {D, T<:Number}
#     Vec{D,T}(ntuple(d -> T(1), D))
# end

export unitvec
function unitvec(::Val{D}, dir::Int)::Vec{D,Bool} where {D}
    @assert 1 <= dir <= D
    Vec{D,Bool}(ntuple(d -> d == dir, D))
end
export unitvecs
function unitvecs(::Val{D})::NTuple{D, Vec{D, Bool}} where {D}
    ntuple(dir -> unitvec(Val(D), dir), D)
end



function Base.:+(x::Vec{D,T})::Vec{D,T} where {D, T}
    Vec{D,T}(.+(x.elts))
end
function Base.:-(x::Vec{D,T})::Vec{D,T} where {D, T}
    Vec{D,T}(.-(x.elts))
end
function Base.inv(x::Vec{D,T})::Vec{D,T} where {D, T}
    Vec{D,T}(inv.(x.elts))
end

function Base.:+(x::Vec{D,T}, y::Vec{D,T})::Vec{D,T} where {D, T}
    Vec{D,T}(x.elts .+ y.elts)
end
function Base.:-(x::Vec{D,T}, y::Vec{D,T})::Vec{D,T} where {D, T}
    Vec{D,T}(x.elts .- y.elts)
end
function Base.:+(x::Vec, y::Vec)::Vec
    +(promote(x, y)...)
end
function Base.:-(x::Vec, y::Vec)::Vec
    -(promote(x, y)...)
end

function Base.:*(a::Number, x::Vec{D,T})::Vec{D,T} where {D, T}
    Vec{D,T}(T(a) .* x.elts)
end
function Base.:*(x::Vec{D,T}, a::Number)::Vec{D,T} where {D, T}
    Vec{D,T}(x.elts .* T(a))
end
function Base.:\(a::Number, x::Vec{D,T})::Vec{D,T} where {D, T}
    Vec{D,T}(T(a) .\ x.elts)
end
function Base.:/(x::Vec{D,T}, a::Number)::Vec{D,T} where {D, T}
    Vec{D,T}(x.elts ./ T(a))
end

const UnaryOp = Union{typeof(ceil), typeof(floor), typeof(round)}
function Base.broadcasted(op::UnaryOp, x::Vec{D,T})::Vec{D,T} where
        {D, T<:Number}
    Vec{D,T}(ntuple(d -> op(x.elts[d]), D))
end

const ArithOp = Union{typeof(+), typeof(-), typeof(*), typeof(/), typeof(\),
                      typeof(min), typeof(max)}
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

function Base.iszero(x::Vec{D,T})::Bool where {D, T<:Number}
    all(iszero.(x))
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
    Vec{D,Bool}(x.elts .| y.elts)
end
function Base.xor(x::Vec{D,Bool}, y::Vec{D,Bool})::Vec{D,Bool} where {D}
    Vec{D,Bool}(xor.(x.elts, y.elts))
end

# function Base.:!(x::Vec{D,Bool})::Vec{D,Bool} where {D}
#     ~x
# end
# function Base.:&&(x::Vec{D,Bool}, y::Vec{D,Bool})::Vec{D,Bool} where {D}
#     x & y
# end
# function Base.:||(x::Vec{D,Bool}, y::Vec{D,Bool})::Vec{D,Bool} where {D}
#     x | y
# end

const BoolOp = Union{typeof(:&), typeof(:|), typeof(xor)}
function Base.broadcasted(op::BoolOp,
                          x::Vec{D,Bool}, y::Vec{D,Bool})::Vec{D,Bool} where {D}
    Vec{D,Bool}(ntuple(d -> op(x.elts[d], y.elts[d]), D))
end
function Base.broadcasted(op::BoolOp,
                          x::Vec{D,Bool}, a::Bool)::Vec{D,Bool} where {D}
    Vec{D,Bool}(ntuple(d -> op(x.elts[d], a), D))
end
function Base.broadcasted(op::BoolOp,
                          a::Bool, x::Vec{D,Bool})::Vec{D,Bool} where {D}
    Vec{D,Bool}(ntuple(d -> op(a, x.elts[d]), D))
end

function Base.all(x::Vec{D,Bool})::Bool where {D}
    all(x.elts)
end
function Base.any(x::Vec{D,Bool})::Bool where {D}
    any(x.elts)
end
function Base.count(x::Vec{D,Bool})::Int where {D}
    count(x.elts)
end
function Base.maximum(x::Vec{D,T})::T where {D, T<:Number}
    max(x.elts)
end
function Base.minimum(x::Vec{D,T})::T where {D, T<:Number}
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



function Base.strides(n::Vec{D,Int})::Vec{D,Int} where {D}
    Vec{D,Int}(ntuple(dir -> dir==1 ? 1 : prod(n[d] for d in 1:dir-1), D))
end

end
