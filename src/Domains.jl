"""
Discrete domain specification
"""
module Domains

using ..Defs
using ..Vecs



export Domain
struct Domain{D, T}
    staggered::Vec{D,Bool}
    # openlb::Vec{D,Bool}
    # openub::Vec{D,Bool}

    # +1 = spacelike, -1 = timelike
    metric::Vec{D,Int}

    # The actual number of vertices or cells also depends on whether
    # the lower or upper bounds are open or closed
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

export makeunstaggered
function makeunstaggered(dom::Domain{D,T})::Domain{D,T} where {D, T}
    Domain{D,T}(Vec(ntuple(d->false, D)), dom.metric,
                dom.n + dom.staggered, dom.xmin, dom.xmax)
end

export makestaggered
function makestaggered(sdom::Domain{D,T},
                       staggered::Vec{D,Bool})::Domain{D,T} where {D, T}
    dom = makeunstaggered(sdom)
    Domain{D,T}(staggered, dom.metric,
                dom.n - staggered, dom.xmin, dom.xmax)
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
function coord(dom::Domain{D,T}, i::Vec{D,<:Number})::Vec{D,T} where
        {D, T<:Number}
    Vec(ntuple(d -> coord(dom, d, i[d]), D))
end

export coords
function coords(dom::Domain{D,T}, d::Int)::Vector{T} where {D, T<:Number}
    T[coord(dom, d, i) for i in 0:dom.n[d]-1]
end
function coords(dom::Domain{D,T})::NTuple{D, Vector{T}} where {D, T<:Number}
    ntuple(d -> coords(dom, d) for d in 1:D)
end



export spacing
function spacing(dom::Domain{D,T})::Vec{D,T} where {D, T<:Number}
    Vec(ntuple(d -> ((dom.xmax[d] - dom.xmin[d]) /
                     (dom.n[d] - !dom.staggered[d])), D))
end

end
