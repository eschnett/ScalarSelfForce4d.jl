"""
Discretized operators
"""
module Ops

using LinearAlgebra
using SparseArrays

using ..Domains
using ..Funs
using ..Vecs



export Op
struct Op{D,T,U} <: AbstractArray{U, 2}
    dom::Domain{D,T}
    mat::SparseMatrixCSC{U,Int}
end



# Op interacts with scalars

# function Base.promote_rule(::Type{Op{D,T,U}}, ::Type{U}) where
#         {D, T<:Number, U}
#     Op{D,T,U}
# end
# function Base.convert(::Type{Op{D,T,U}}, x::U) where {D, T<:Number, U}
#     Op{D,T,U}(ntuple(d -> x, D))
# end



# Fun is a collection

function Base.eltype(A::Op{D,T,U})::Type where {D, T, U}
    eltype(A.mat)
end
function Base.length(A::Op{D,T,U})::Int where {D, T, U}
    length(A.mat)
end
function Base.ndims(A::Op{D,T,U})::Int where {D, T, U}
    ndims(A.mat)
end
function Base.size(A::Op{D,T,U})::NTuple{2,Int} where {D, T, U}
    size(A.mat)
end
function Base.size(A::Op{D,T,U}, d)::Int where {D, T, U}
    size(A.mat, d)
end

function Base.getindex(A::Op{D,T,U}, i::Vec{D,Int})::U where {D, T, U}
    getindex(A.mat, i.elts)
end
function Base.getindex(A::Op{D,T,U}, is...)::U where {D, T, U}
    getindex(A.mat, is...)
end



# Op is a vector space

function Base.zeros(::Type{Op{D,T,U}}, dom::Domain{D,T})::Op{D,T,U} where
        {D, T<:Number, U}
    len = prod(dom.n)
    mat = spzeros(U, len, len)
    Op{D,T,U}(dom, mat)
end

function Base.:+(A::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    Op{D,T,U}(A.dom, +A.mat)
end
function Base.:-(A::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    Op{D,T,U}(A.dom, -A.mat)
end

function Base.:+(A::Op{D,T,U}, B::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    @assert A.dom == B.dom
    Op{D,T,U}(A.dom, A.mat + B.mat)
end
function Base.:-(A::Op{D,T,U}, B::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    @assert A.dom == B.dom
    Op{D,T,U}(A.dom, A.mat - B.mat)
end

function Base.:*(a::Number, A::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    Op{D,T,U}(A.dom, U(a) * A.mat)
end
function Base.:*(A::Op{D,T,U}, a::Number)::Op{D,T,U} where {D, T<:Number, U}
    Op{D,T,U}(A.dom, A.mat * U(a))
end
function Base.:\(a::Number, A::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    Op{D,T,U}(A.dom, U(a) \ A.mat)
end
function Base.:/(A::Op{D,T,U}, a::Number)::Op{D,T,U} where {D, T<:Number, U}
    Op{D,T,U}(A.dom, A.mat / U(a))
end



function Base.:*(A::Op{D,T,U}, B::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    @assert A.dom == B.dom
    Op{D,T,U}(A.dom, A.mat * B.mat)
end



function Base.zero(::Type{Op{D,T,U}}, dom::Domain{D,T})::Op{D,T,U} where
        {D, T, U}
    zeros(Op{D,T,U}, dom)
end
function Base.one(::Type{Op{D,T,U}}, dom::Domain{D,T})::Op{D,T,U} where
        {D, T, U}
    n = dom.n

    str = Vec{D,Int}(ntuple(dir -> dir==1 ? 1 : prod(n[d] for d in 1:dir-1), D))
    len = prod(n)
    idx(i::Vec{D,Int}) = 1 + sum(i[d] * str[d] for d in 1:D)

    I = Int[]
    J = Int[]
    V = U[]
    function ins!(i, j, v)
        push!(I, idx(i))
        push!(J, idx(j))
        push!(V, v)
    end
    for ic in CartesianIndices(dom.n.elts)
        i = Vec(ic.I) .- 1
        ins!(i, i, U(1))
    end
    mat = sparse(I, J, V, len, len)
    Op{D,T,U}(dom, mat)
end

function Base.:*(op::Op{D,T,U}, rhs::Fun{D,T,U})::Fun{D,T,U} where {D, T, U}
    dom = rhs.dom
    @assert op.dom == dom

    res = reshape(op.mat * reshape(rhs.coeffs, :), dom.n.elts)
    Fun{D,T,U}(dom, res)
end

# function Base.:*(lhs::Fun{D,T,U}, op::Op{D,T,U})::Fun{D,T,U} where {D, T, U}
#     dom = rhs.dom
#     @assert op.dom == dom
# 
#     TODO: bra and ket
#     res = reshape(reshape(lhs.coeffs, :) * op.mat, dom.n.elts)
#     Fun{D,T,U}(dom, res)
# end

function Base.:\(op::Op{D,T,U}, rhs::Fun{D,T,U})::Fun{D,T,U} where {D, T, U}
    dom = rhs.dom
    @assert op.dom == dom

    M = op.mat
    if T <: BigFloat
        @info "Converting sparse to full matrix..."
        M = Matrix(M)
    else
        # do nothing
    end
    sol = reshape(M \ reshape(rhs.coeffs, :), dom.n.elts)
    Fun{D,T,U}(dom, sol)
end

# function Base.:\(op::Op{D,T,U}, rhs::Fun{D,T,U})::Fun{D,T,U} where {D, T, U}
#     dom = rhs.dom
#     @assert op.dom == dom
# 
#     len = prod(dom.n)
# 
#     bnd = boundary(U, dom)
#     proj = I(len) - bnd.mat
#     sol = reshape(op.mat \ (proj * reshape(rhs.coeffs, :)), dom.n.elts)
#     Fun{D,T,U}(dom, sol)
# end



# Note: These work for boundary conditions, but not for initial
# conditions. The matrices / RHS vectors have rows that are off by one
# for initial conditions.
export mix_op_bc
function mix_op_bc(bnd::Op{D,T,U},
                   iop::Op{D,T,U}, bop::Op{D,T,U})::Op{D,T,U} where
        {D, T<:Number, U}
    dom = bnd.dom
    @assert iop.dom == dom
    @assert bop.dom == dom

    id = one(Op{D,T,U}, dom)
    int = id - bnd
    int * iop + bnd * bop
end
function mix_op_bc(bnd::Op{D,T,U},
                   rhs::Fun{D,T,U}, bvals::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U}
    dom = bnd.dom
    @assert rhs.dom == dom
    @assert bvals.dom == dom

    id = one(Op{D,T,U}, dom)
    int = id - bnd
    int * rhs + bnd * bvals
end

end
