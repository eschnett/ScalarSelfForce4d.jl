"""
Discrete domain specification
"""
module Domains

using ..Defs
using ..Vecs



export Domain
struct Domain{D, T}
    dual::Bool

    # Whether the respective direction holds vertices (false) or cells
    # (true)
    staggered::Vec{D,Bool}

    # Actual number of vertices or cells
    n::Vec{D,Int}

    # Metric: +1 = spacelike, -1 = timelike
    metric::Vec{D,Int}

    # Domain boundary; vertices lie on the boundary, cell centres lie
    # dx/2 inwards
    xmin::Vec{D,T}
    xmax::Vec{D,T}
end

function (::Type{Domain{D,T}})(np::Int; lorentzian=false) where {D, T}
    staggered = falses(Vec{D,Bool})
    n = Vec{D,Int}(ntuple(d -> np, D))
    if lorentzian
        metric = Vec{D,Int}(ntuple(d -> d==D ? -1 : 1, D))
    else
        metric = Vec{D,Int}(ntuple(d -> 1, D))
    end
    xmin = Vec{D,T}(ntuple(d -> metric[d] < 0 ? 0 : -1, D))
    xmax = Vec{D,T}(ntuple(d -> 1, D))
    Domain{D,T}(false, staggered, n, metric, xmin, xmax)
end

export makeprimal
function makeprimal(dom::Domain{D,T})::Domain{D,T} where {D, T}
    Domain{D,T}(false, xor.(dom.dual, dom.staggered), dom.n,
                dom.metric, dom.xmin, dom.xmax)
end

export makedual
function makedual(dom::Domain{D,T}, dual::Bool)::Domain{D,T} where {D, T}
    Domain{D,T}(dual, xor.(dual != dom.dual, dom.staggered), dom.n,
                dom.metric, dom.xmin, dom.xmax)
end

export makeunstaggered
function makeunstaggered(dom::Domain{D,T})::Domain{D,T} where {D, T}
    Domain{D,T}(dom.dual, falses(Vec{D,Bool}),
                !dom.dual
                ? dom.n + dom.staggered
                : dom.n - dom.staggered,
                dom.metric, dom.xmin, dom.xmax)
end

export makestaggered
function makestaggered(sdom::Domain{D,T},
                       staggered::Vec{D,Bool})::Domain{D,T} where {D, T}
    dom = makeunstaggered(sdom)
    Domain{D,T}(dom.dual, staggered,
                !dom.dual
                ? dom.n - staggered
                : dom.n + staggered,
                dom.metric, dom.xmin, dom.xmax)
end



# Coordinates of collocation points
export coord
function coord(dom::Domain{D,T}, d::Int, i::Number)::T where {D, T<:Number}
    @assert !dom.dual
    @assert 0 <= i <= dom.n[d] + dom.staggered[d] - 1
    linear(T(0), dom.xmin[d],
           T(dom.n[d] + dom.staggered[d] - 1), dom.xmax[d], T(i))
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
function spacing(dom::Domain{D,T}, d::Int)::T where {D, T<:Number}
    nc = dom.n[d] + (dom.staggered[d] != dom.dual) - 1
    (dom.xmax[d] - dom.xmin[d]) / nc
end
function spacing(dom::Domain{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(ntuple(d -> spacing(dom, d), D))
end

end
