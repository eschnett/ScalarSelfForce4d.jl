"""
Basis functions
"""
module Bases

using LinearAlgebra

using ..Defs
using ..Domains
using ..Vecs



# Basis functions
export basis
function basis(dom::Domain{D,T}, d::Int, i::Int, ix::Int, x::T)::T where
        {D,T <: Number}
    @assert !dom.dual
    @assert !dom.staggered[d]
    @assert i >= 0 && i <= dom.n[d] - 1
    @assert ix >= 0 && ix <= dom.n[d] - 1
    if ix == i - 1
        x0 = coord(dom, d, i - 1)
        x1 = coord(dom, d, i)
        @assert x0 <= x <= x1   # TODO
        linear(x0, T(0), x1, T(1), x)
    elseif ix == i
        x1 = coord(dom, d, i)
        x0 = coord(dom, d, i + 1)
        @assert x1 <= x <= x0   # TODO
        linear(x0, T(0), x1, T(1), x)
    else
        T(0)
    end
end

# Staggered basis functions
export sbasis
function sbasis(dom::Domain{D,T}, d::Int, i::Int, ix::Int, x::T)::T where
        {D,T <: Number}
    @assert !dom.dual
    @assert dom.staggered[d]
    @assert i >= 0 && i <= dom.n[d] - 1
    @assert ix >= 0 && ix <= dom.n[d] - 1
    if ix == i
        dx = spacing(dom, d)
        1 / dx
    else
        @assert false
        T(0)
    end
end

# Dot product between basis functions
export dot_basis
function dot_basis(dom::Domain{D,T}, d::Int, i::Int, j::Int)::T where
        {D,T <: Number}
    @assert !dom.dual
    @assert !dom.staggered[d]
    @assert i >= 0 && i < dom.n[d]
    @assert j >= 0 && j < dom.n[d]
    dx = spacing(dom)[d]
    if j == i - 1
        return dx / 6
    elseif j == i
        if i == 0 || i == dom.n[d] - 1
            # This ignores the cell at infinity
            return dx / 3
        else
            return T(2) / 3 * dx
        end
    elseif j == i + 1
        return dx / 6
    else
        return T(0)
    end
end

export dot_sbasis
function dot_sbasis(dom::Domain{D,T}, d::Int, i::Int, j::Int)::T where
        {D,T <: Number}
    @assert !dom.dual
    @assert dom.staggered[d]
    @assert i >= 0 && i < dom.n[d]
    @assert j >= 0 && j < dom.n[d]
    dx = spacing(dom)[d]
    if j == i
        return 1 / dx
    else
        return T(0)
    end
end



export fbasis
function fbasis(dom::Domain{D,T}, d::Int, i::Int, ix::Int, x::T)::T where
        {D,T <: Number}
    if !dom.staggered[d]
        basis(dom, d, i, ix, x)
    else
        sbasis(dom, d, i, ix, x)
    end
end
function fbasis(dom::Domain{D,T}, i::Vec{D,Int}, ix::Vec{D,Int},
                x::Vec{D,T})::T where
        {D,T <: Number}
    prod(fbasis(dom, d, i[d], ix[d], x[d]) for d in 1:D)
end

export dot_fbasis
function dot_fbasis(dom::Domain{D,T}, d::Int, i::Int, j::Int)::T where
        {D,T <: Number}
    if !dom.staggered[d]
        dot_basis(dom, d, i, j)
    else
        dot_sbasis(dom, d, i, j)
    end
end
function dot_fbasis(dom::Domain{D,T}, d::Int)::AbstractMatrix{T} where
        {D,T <: Number}
    if !dom.staggered[d]
        dv = [dot_fbasis(dom, d, i, i) for i in 0:dom.n[d] - 1]
        ev = [dot_fbasis(dom, d, i, i + 1) for i in 0:dom.n[d] - 2]
        SymTridiagonal(dv, ev)
    else
        dv = [dot_fbasis(dom, d, i, i) for i in 0:dom.n[d] - 1]
        Diagonal(dv)
    end
end
function dot_fbasis(dom::Domain{D,T})::NTuple{D,AbstractMatrix{T}} where
        {D,T <: Number}
    ntuple(d->dot_fbasis(dom, d), D)
end



# # Integration weights for basis functions
export weight
function weight(dom::Domain{D,T}, d::Int, i::Int)::T where {D,T <: Number}
    @assert !dom.dual

    n = dom.n[d]
    @assert i >= 0 && i < n
    dx = spacing(dom)[d]

    if !dom.staggered[d]
        if i == 0
            return dx / 2
        elseif i < n - 1
            return dx
        else
            return dx / 2
        end
    else
        return T(1)
    end
end

export weights
function weights(dom::Domain{D,T}, d::Int)::Vector{T} where {D,T}
    T[weight(dom, d, i) for i in 0:dom.n[d] - 1]
end
function weights(dom::Domain{D,T})::NTuple{D,Vector{T}} where {D,T}
    ntuple(d->weights(dom, d), D)
end

end
