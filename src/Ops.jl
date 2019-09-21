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
struct Op{D,T,U} <: AbstractMatrix{U}
    domi::Domain{D,T}
    domj::Domain{D,T}
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

function Base.eltype(::Type{Op{D,T,U}})::Type where {D, T, U}
    U
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

function Base.zeros(::Type{Op{D,T,U}},
                    domi::Domain{D,T}, domj::Domain{D,T})::Op{D,T,U} where
        {D, T<:Number, U}
    leni = prod(domi.n)
    lenj = prod(domj.n)
    mat = spzeros(U, leni, lenj)
    Op{D,T,U}(domi, domj, mat)
end

function Base.:+(A::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    Op{D,T,U}(A.domi, A.domj, +A.mat)
end
function Base.:-(A::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    Op{D,T,U}(A.domi, A.domj, -A.mat)
end

function Base.:+(A::Op{D,T,U}, B::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    @assert A.domi == B.domi && A.domj == B.domj
    Op{D,T,U}(A.domi, A.domj, A.mat + B.mat)
end
function Base.:-(A::Op{D,T,U}, B::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    @assert A.domi == B.domi && A.domj == B.domj
    Op{D,T,U}(A.domi, A.domj, A.mat - B.mat)
end

function Base.:*(a::Number, A::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    Op{D,T,U}(A.domi, A.domj, U(a) * A.mat)
end
function Base.:*(A::Op{D,T,U}, a::Number)::Op{D,T,U} where {D, T<:Number, U}
    Op{D,T,U}(A.domi, A.domj, A.mat * U(a))
end
function Base.:\(a::Number, A::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    Op{D,T,U}(A.domi, A.domj, U(a) \ A.mat)
end
function Base.:/(A::Op{D,T,U}, a::Number)::Op{D,T,U} where {D, T<:Number, U}
    Op{D,T,U}(A.domi, A.domj, A.mat / U(a))
end

function Base.iszero(A::Op{D,T,U})::Bool where {D, T<:Number, U}
    iszero(A.mat)
end
function Base.:(==)(A::Op{D,T,U}, B::Op{D,T,U})::Bool where {D, T<:Number, U}
    iszero(A - B)
end



function Base.:*(A::Op{D,T,U}, B::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U}
    @assert A.domj == B.domi
    Op{D,T,U}(A.domi, B.domj, A.mat * B.mat)
end



function Base.zero(::Type{Op{D,T,U}},
                   domi::Domain{D,T}, domj::Domain{D,T})::Op{D,T,U} where
        {D, T, U}
    zeros(Op{D,T,U}, domi, domj)
end
function Base.one(::Type{Op{D,T,U}},
                  domi::Domain{D,T}, domj::Domain{D,T})::Op{D,T,U} where
        {D, T, U}
    ni = domi.n
    stri = strides(ni)
    leni = prod(ni)
    idxi(i::CartesianIndex{D}) = 1 + sum((i[d] - 1) * stri[d] for d in 1:D)
    nj = domj.n
    strj = strides(nj)
    lenj = prod(nj)
    idxj(j::CartesianIndex{D}) = 1 + sum((j[d] - 1) * strj[d] for d in 1:D)

    n = min.(ni, nj)
    I = Int[]
    J = Int[]
    V = U[]
    sizehint!(I, prod(n))
    sizehint!(J, prod(n))
    sizehint!(V, prod(n))
    function ins!(i, j, v)
        push!(I, idxi(i))
        push!(J, idxj(j))
        push!(V, v)
    end
    for i in CartesianIndices(n.elts)
        ins!(i, i, U(1))
    end
    mat = sparse(I, J, V, leni, lenj)
    Op{D,T,U}(domi, domj, mat)
end

function Base.:*(op::Op{D,T,U}, rhs::Fun{D,T,U})::Fun{D,T,U} where {D, T, U}
    @assert op.domj == rhs.dom

    res = reshape(op.mat * reshape(rhs.coeffs, :), op.domi.n.elts)
    Fun{D,T,U}(op.domi, res)
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
    @assert op.domi == rhs.dom

    M = op.mat
    if T <: Union{BigFloat, Rational}
        @info "Converting sparse to full matrix..."
        M = Matrix(M)
    else
        # do nothing
    end
    sol = reshape(M \ reshape(rhs.coeffs, :), op.domj.n.elts)
    Fun{D,T,U}(op.domj, sol)
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
    domi = bnd.domi
    domj = bnd.domj
    @assert iop.domi == domi && iop.domj == domj
    @assert bop.domi == domi && bop.domj == domj

    id = one(Op{D,T,U}, domi, domj)
    int = id - bnd
    int * iop + bnd * bop
end
function mix_op_bc(bnd::Op{D,T,U},
                   rhs::Fun{D,T,U}, bvals::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U}
    domi = bnd.domi
    domj = bnd.domj
    @assert rhs.domi == domi && rhs.domj == domj
    @assert bvals.domi == domi && bvals.domj == domj

    id = one(Op{D,T,U}, domi, domj)
    int = id - bnd
    int * rhs + bnd * bvals
end

end
