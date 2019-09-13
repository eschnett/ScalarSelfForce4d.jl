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
function basis(dom::Domain{D,T}, d::Int, i::Int, x::T)::T where {D, T<:Number}
    @assert i>=0 && i<dom.n[d]
    if dom.staggered[d]
        x0 = coord(dom, d, T(i) - T(1)/2)
        x1 = coord(dom, d, T(i) + T(1)/2)
        y0 = i==0 ? T(1) : T(1)/2
        y1 = i==dom.n[d]-1 ? T(1) : T(1)/2
        characteristic(x0, y0, x1, y1, x)
    else
        fm = linear(dom.xmin[d], T(1 - i), dom.xmax[d], T(1 + dom.n[d] - 1 - i),
                    x)
        fp = linear(dom.xmin[d], T(1 + i), dom.xmax[d], T(1 - dom.n[d] + 1 + i),
                    x)
        f0 = T(0)
        max(f0, min(fm, fp))
    end
end
@generated function basis(dom::Domain{D,T}, i::Vec{D,Int}, x::Vec{D,T})::T where
        {D, T<:Number}
    quote
        *($([:(basis(dom, $d, i[$d], x[$d])) for d in 1:D]...))
    end
end

# Like basis, but the CC basis functions are continued to avoid
# problems due to round-off
export basis1
function basis1(dom::Domain{D,T}, d::Int, i::Int, x::T)::T where {D, T<:Number}
    @assert i>=0 && i<dom.n[d]
    if dom.staggered[d]
        T(1)
    else
        fm = linear(dom.xmin[d], T(1 - i), dom.xmax[d], T(1 + dom.n[d] - 1 - i),
                    x)
        fp = linear(dom.xmin[d], T(1 + i), dom.xmax[d], T(1 - dom.n[d] + 1 + i),
                    x)
        f0 = T(0)
        max(f0, min(fm, fp))
    end
end
function basis1(dom::Domain{D,T}, i::Vec{D,Int}, x::Vec{D,T})::T where
        {D, T<:Number}
    prod(basis1(dom, d, i[d], x[d]) for d in 1:D)
end



# Dot product between basis functions
export dot_basis
function dot_basis(dom::Domain{D,T}, d::Int, i::Int, j::Int)::T where
        {D, T<:Number}
    n = dom.n[d]
    @assert i>=0 && i<n
    @assert j>=0 && j<n
    dx = spacing(dom)[d]
    if dom.staggered[d]
        if (j == i)
            return dx
        else
            return T(0)
        end
    else
        if j == i-1
            return dx/6
        elseif j == i
            if i == 0 || i == n-1
                return dx/3
            else
                return T(2)/3*dx
            end
        elseif j == i+1
            return dx/6
        else
            return T(0)
        end
    end
end



# Integration weights for basis functions (assuming a diagonal weight matrix)
export weight
function weight(dom::Domain{D,T}, d::Int, i::Int)::T where {D, T<:Number}
    n = dom.n[d]
    @assert i>=0 && i<n
    dx = spacing(dom)[d]
    if dom.staggered[d]
        return dx
    else
        if i == 0
            return dx/2
        elseif i < n-1
            return dx
        else
            return dx/2
        end
    end
end
function weight(dom::Domain{D,T}, i::Vec{D,Int})::T where {D, T<:Number}
    prod(weight(dom, i[d]) for d in 1:D)
end

export weights
function weights(dom::Domain{D,T}, d::Int)::AbstractMatrix{T} where {D,T}
    # We know the overlaps of the support of the basis functions
    if dom.staggered[d]
        dv = [dot_basis(dom, d, i, i) for i in 0:dom.n[d]-1]
        Diagonal(dv)
    else
        dv = [dot_basis(dom, d, i, i) for i in 0:dom.n[d]-1]
        ev = [dot_basis(dom, d, i, i+1) for i in 0:dom.n[d]-2]
        SymTridiagonal(dv, ev)
    end
end
function weights(dom::Domain{D,T}) where {D,T}
    ntuple(d -> weights(dom, d), D)
end



# Derivative of basis functions   dϕ^i/dϕ^j
function deriv_basis(dom::Domain{D,T}, d::Int, i::Int, j::Int)::T where
        {D, T<:Number}
    @assert !any(dom.staggered) # TODO
    dx = (dom.xmax[d] - dom.xmin[d]) / (fun.dom.n[d] - 1)
    if i == 0
        if j == i
            return -1/dx
        elseif j == i+1
            return 1/dx
        end
    elseif i < dom.n[d]
        if j == i-1
            return -T(1)/2/dx
        elseif j == i+1
            return T(1)/2/dx
        end
    else
        if j == i-1
            return -1/dx
        elseif j == i
            return 1/dx
        end
    end
    T(0)
end

end
