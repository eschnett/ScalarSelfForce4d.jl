"""
Discrete domain specification
"""
module Domains

using ..Defs
using ..Vecs



export Domain
struct Domain{D, T}
    # VC = false, CC = true
    staggered::Vec{D,Bool}
    # +1 = spacelike, -1 = timelike
    metric::Vec{D,Int}

    # Actual number of vertices or cells; for the same dx, n[VC] =
    # n[CC] + 1
    n::Vec{D,Int}

    # Domain boundary; vertices lie on the boundary, cell centres lie
    # dx/2 inwards
    xmin::Vec{D,T}
    xmax::Vec{D,T}
end

function (::Type{Domain{D,T}})(np::Int; lorentzian=false) where {D, T}
    if lorentzian
        metric = Vec(ntuple(d -> d==D ? -1 : 1, D))
    else
        metric = Vec(ntuple(d -> 1, D))
    end
    staggered = Vec(ntuple(d -> false, D))
    n = Vec(ntuple(d -> np, D))
    xmin = Vec{D,T}(ntuple(d -> metric[d]<0 ? 0 : -1, D))
    xmax = Vec{D,T}(ntuple(d -> 1, D))
    Domain{D,T}(staggered, metric, n, xmin, xmax)
end

export makestaggered
function makestaggered(dom::Domain{D,T})::Domain{D,T} where {D, T}
    @assert all(!dom.staggered)
    sdom = Domain{D,T}(!dom.staggered, dom.metric,
                       dom.n .- 1, dom.xmin, dom.xmax)
    sdom
end



# Coordinates of collocation points
export coord
function coord(dom::Domain{D,T}, d::Int, i::Number)::T where {D, T<:Number}
    if dom.staggered[d]
        j = T(i) + T(1)/2
        @assert 0 <= j <= dom.n[d]
        linear(T(0), dom.xmin[d], T(dom.n[d]), dom.xmax[d], j)
    else
        @assert 0 <= i <= dom.n[d] - 1
        linear(T(0), dom.xmin[d], T(dom.n[d]-1), dom.xmax[d], T(i))
    end
end

export coords
function coords(dom::Domain{D,T}, d::Int)::Vector{T} where {D, T<:Number}
    T[coord(dom, d, i) for i in 0:dom.n[d]-1]
end
function coords(dom::Domain{D,T})::NTuple{D, Vector{T}} where {D, T<:Number}
    ntuple(d -> coords(dom, d) for d in 1:D)
end

end
