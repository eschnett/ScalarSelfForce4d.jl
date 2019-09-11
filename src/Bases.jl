"""
Basis functions
"""
module Bases

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
        characteristic(T, x0, x1, x)
    else
        fm = linear(dom.xmin[d], T(1 - i), dom.xmax[d], T(1 + dom.n[d] - 1 - i),
                    x)
        fp = linear(dom.xmin[d], T(1 + i), dom.xmax[d], T(1 - dom.n[d] + 1 + i),
                    x)
        f0 = T(0)
        max(f0, min(fm, fp))
    end
end
function basis(dom::Domain{D,T}, i::Vec{D,Int}, x::Vec{D,T})::T where
        {D, T<:Number}
    prod(basis(dom, d, i[d], x[d]) for d in 1:D)
end



# Dot product between basis functions
export dot_basis
function dot_basis(dom::Domain{D,T}, d::Int, i::Int, j::Int)::T where
        {D, T<:Number}
    n = dom.n[d]
    @assert i>=0 && i<n
    @assert j>=0 && j<n
    if dom.staggered[d]
        return T(i == j)
    else
        dx = (dom.xmax[d] - dom.xmin[d]) / (n - 1)
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
    if dom.staggered[d]
        dx = (dom.xmax[d] - dom.xmin[d]) / n
        return dx
    else
        dx = (dom.xmax[d] - dom.xmin[d]) / (n - 1)
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

end
