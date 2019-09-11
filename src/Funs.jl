"""
Discretized functions
"""
module Funs

using LinearAlgebra

using ..Bases
using ..Defs
using ..Domains
using ..Quadrature
using ..Vecs



export Fun
struct Fun{D,T,U} <: DenseArray{T, D}
    dom::Domain{D,T}
    coeffs::Array{U,D}
end



# Fun interacts with scalars

# function Base.promote_rule(::Type{Fun{D,T,U}}, ::Type{U}) where
#         {D, T<:Number, U<:Number}
#     Fun{D,T,U}
# end
# function Base.convert(::Type{Fun{D,T,U}}, x::U) where {D, T<:Number, U<:Number}
#     Fun{D,T,U}(ntuple(d -> x, D))
# end



# Fun is a collection

function Base.eltype(f::Fun{D,T,U})::Type where {D, T, U}
    U
end
function Base.length(f::Fun{D,T,U})::Int where {D, T, U}
    prod(f.dom.n)
end
function Base.ndims(f::Fun{D,T,U})::Int where {D, T, U}
    D
end
function Base.size(f::Fun{D,T,U})::NTuple{D,Int} where {D, T, U}
    f.dom.n.elts
end
function Base.size(f::Fun{D,T,U}, d)::Int where {D, T, U}
    prod(f.dom.n)
end

function Base.getindex(f::Fun{D,T,U}, i::Vec{D,Int})::U where {D, T, U}
    getindex(f.coeffs, i.elts)
end
function Base.getindex(f::Fun{D,T,U}, is...)::U where {D, T, U}
    getindex(f.coeffs, is...)
end



# Fun is a vector space

function Base.zeros(::Type{Fun{D,T,U}}, dom::Domain{D,T})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    Fun{D,T,U}(dom, zeros(U, dom.n.elts))
end

function Base.:+(f::Fun{D,T,U})::Fun{D,T,U} where {D, T<:Number, U<:Number}
    Fun{D,T,U}(f.dom, +f.elts)
end
function Base.:-(f::Fun{D,T,U})::Fun{D,T,U} where {D, T<:Number, U<:Number}
    Fun{D,T,U}(f.dom, -f.elts)
end

function Base.:+(f::Fun{D,T,U}, g::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert f.dom == g.dom
    Fun{D,T,U}(f.dom, f.coeffs + g.coeffs)
end
function Base.:-(f::Fun{D,T,U}, g::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert f.dom == g.dom
    Fun{D,T,U}(f.dom, f.coeffs - g.coeffs)
end

function Base.:*(a::Number, f::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    Fun{D,T,U}(f.dom, U(a) * f.coeffs)
end
function Base.:*(f::Fun{D,T,U}, a::Number)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    Fun{D,T,U}(f.dom, f.coeffs * U(a))
end
function Base.:\(a::Number, f::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    Fun{D,T,U}(f.dom, U(a) \ f.coeffs)
end
function Base.:/(f::Fun{D,T,U}, a::Number)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    Fun{D,T,U}(f.dom, f.coeffs / U(a))
end

# function Base.:.+(f::Fun{D,T,U}, c::U)::Fun{D,T,U} where
#         {D, T<:Number, U<:Number}
#     Fun{D,T,U}(f.dom, f.coeffs .+ c)
# end
# function Base.:.-(f::Fun{D,T,U}, c::U)::Fun{D,T,U} where
#         {D, T<:Number, U<:Number}
#     Fun{D,T,U}(f.dom, f.coeffs .- c)
# end

# function Base.:*(f::Fun{D,T,U}, g::Fun{D,T,U})::U where
#         {D, T<:Number, U<:Number}
# TODO: bra and ket
# end

function Base.max(f::Fun{D,T,U})::T where {D, T<:Number, U<:Number}
    maximum(f.coeffs)
end
function Base.min(f::Fun{D,T,U})::T where {D, T<:Number, U<:Number}
    minimum(f.coeffs)
end
function Base.sum(f::Fun{D,T,U})::T where {D, T<:Number, U<:Number}
    Ws = ntuple(D) do d
        ws = [weight(dom, d, i) for i in 0:dom.n[d]-1]
        Diagonal(ws)
    end

    n = dom.n
    s = U(0)
    if D == 1
        for i1 in 1:n[1]
            s += Ws[1][i1] * f.coeffs[i1]
        end
    elseif D == 2
        for i2 in 1:n[2], i1 in 1:n[1]
            s += Ws[1][i1] * Ws[2][i2] * f.coeffs[i1,i2]
        end
    elseif D == 3
        for i3 in 1:n[3], i2 in 1:n[2], i1 in 1:n[1]
            s += Ws[1][i1] * Ws[2][i2] * Ws[3][i3] * f.coeffs[i1,i2,i3]
        end
    elseif D == 4
        for i4 in 1:n[4], i3 in 1:n[3], i2 in 1:n[2], i1 in 1:n[1]
            s += (Ws[1][i1] * Ws[2][i2] * Ws[3][i3] * Ws[4][i4] *
                  f.coeffs[i1,i2,i3,i4])
        end
    else
        @assert false
    end
end

function LinearAlgebra.norm(f::Fun{D,T,U}, p::Real=2) where
        {D, T<:Number, U<:Number}
    if p == Inf
        maximum(abs.(f.coeffs))
    else
        @assert false
    end
end



# Fun are a category

# TODO: composition

function fidentity(::Type{U}, dom::Domain{1,T})::Fun{1,T,U} where
        {T<:Number, U<:Number}
    if dom.staggered[1]
        dx = (dom.xmax[1] - dom.xmin[1]) / dom.n[1]
        cs = LinRange(U(dom.xmin[1] + dx[1]/2), U(dom.xmax[1] - dx[1]/2),
                      dom.n[1])
    else
        cs = LinRange(U(dom.xmin[1]), U(dom.xmax[1]), dom.n[1])
    end
    Fun{1,T,U}(dom, cs)
end

function fconst(dom::Domain{D,T}, f::U)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    cs = fill(f, dom.n.elts)
    Fun{D,T,U}(dom, cs)
end



# Evaluate a function
function (fun::Fun{D,T,U})(x::Vec{D,T})::U where {D, T<:Number, U<:Number}
    f = U(0)
    for ic in CartesianIndices(size(fun.coeffs))
        i = Vec(ic.I) .- 1
        f += fun.coeffs[ic] * basis(fun.dom, i, x)
    end
    f
end

# Create a discretized function by projecting onto the basis functions
export approximate
function approximate(fun, ::Type{U}, dom::Domain{D,T})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    if all(!dom.staggered)
        approximate_vc(fun, U, dom)
    elseif all(dom.staggered)
        approximate_cc(fun, U, dom)
    else
        @assert false
    end
end

function approximate_vc(fun, ::Type{U}, dom::Domain{D,T})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert all(!dom.staggered)

    str = Vec{D,Int}(ntuple(dir ->
                            dir==1 ? 1 : prod(dom.n[d] for d in 1:dir-1), D))
    len = prod(dom.n)
    idx(i::Vec{D,Int}) = 1 + sum(i[d] * str[d] for d in 1:D)
    function active(i::Vec{D,Int})::Bool
        mask === nothing && return true
        mask.mat[idx(i), idx(i)] != 0
    end

    fs = Array{U,D}(undef, dom.n.elts)
    for ic in CartesianIndices(size(fs))
        i = Vec(ic.I) .- 1
        f = U(0)
        function kernel(x0::NTuple{D})::U
            x = Vec{D,T}(x0)
            U(fun(x)) * U(basis(dom, i, x))
        end
        x0 = Vec{D,T}(ntuple(d -> linear(
            0, dom.xmin[d], dom.n[d]-1, dom.xmax[d],
            max(0, i[d]-1)), D))
        x1 = Vec{D,T}(ntuple(d -> linear(
            0, dom.xmin[d], dom.n[d]-1, dom.xmax[d],
            min(dom.n[d]-1, i[d]+1)), D))
        n = Vec{D,Int}(ntuple(d -> 8, D))
        f = quad(kernel, U, x0.elts, x1.elts, n.elts)
        fs[ic] = f
    end

    Ws = ntuple(D) do d
        # We know the overlaps of the support of the basis functions
        dv = [dot_basis(dom, d, i, i) for i in 0:dom.n[d]-1]
        ev = [dot_basis(dom, d, i, i+1) for i in 0:dom.n[d]-2]
        SymTridiagonal(dv, ev)
    end

    n = dom.n
    cs = fs
    if D == 1
        cs[:] = Ws[1] \ cs[:]
    elseif D == 2
        for i2 in 1:n[2]
            cs[:,i2] = Ws[1] \ cs[:,i2]
        end
        for i1 in 1:n[1]
            cs[i1,:] = Ws[2] \ cs[i1,:]
        end
    elseif D == 3
        for i3 in 1:n[3], i2 in 1:n[2]
            cs[:,i2,i3] = Ws[1] \ cs[:,i2,i3]
        end
        for i3 in 1:n[3], i1 in 1:n[1]
            cs[i1,:,i3] = Ws[2] \ cs[i1,:,i3]
        end
        for i2 in 1:n[2], i1 in 1:n[1]
            cs[i1,i2,:] = Ws[3] \ cs[i1,i2,:]
        end
    elseif D == 4
        for i4 in 1:n[4], i3 in 1:n[3], i2 in 1:n[2]
            cs[:,i2,i3,i4] = Ws[1] \ cs[:,i2,i3,i4]
        end
        for i4 in 1:n[4], i3 in 1:n[3], i1 in 1:n[1]
            cs[i1,:,i3,i4] = Ws[2] \ cs[i1,:,i3,i4]
        end
        for i4 in 1:n[4], i2 in 1:n[2], i1 in 1:n[1]
            cs[i1,i2,:,i4] = Ws[3] \ cs[i1,i2,:,i4]
        end
        for i3 in 1:n[3], i2 in 1:n[2], i1 in 1:n[1]
            cs[i1,i2,i3,:] = Ws[4] \ cs[i1,i2,i3,:]
        end
    else
        @assert false
    end
    return Fun{D,T,U}(dom, cs)
end

function approximate_cc(fun, ::Type{U}, dom::Domain{D,T})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert all(dom.staggered)

    str = Vec{D,Int}(ntuple(dir ->
                            dir==1 ? 1 : prod(dom.n[d] for d in 1:dir-1), D))
    len = prod(dom.n)
    idx(i::Vec{D,Int}) = 1 + sum(i[d] * str[d] for d in 1:D)
    function active(i::Vec{D,Int})::Bool
        mask === nothing && return true
        mask.mat[idx(i), idx(i)] != 0
    end

    fs = Array{U,D}(undef, dom.n.elts)
    for ic in CartesianIndices(size(fs))
        i = Vec(ic.I) .- 1
        f = U(0)
        function kernel(x0::NTuple{D})::U
            x = Vec{D,T}(x0)
            U(fun(x))
        end
        x0 = Vec{D,T}(ntuple(d -> linear(
            T(0), dom.xmin[d], T(dom.n[d]), dom.xmax[d], T(i[d])), D))
        x1 = Vec{D,T}(ntuple(d -> linear(
            T(-1), dom.xmin[d], T(dom.n[d]-1), dom.xmax[d], T(i[d])), D))
        n = Vec{D,Int}(ntuple(d -> 8, D))
        f = quad(kernel, U, x0.elts, x1.elts, n.elts)
        fs[ic] = f
    end

    return Fun{D,T,U}(dom, fs)
end



# Approximate a delta function
export approximate_delta
function approximate_delta(::Type{U}, dom::Domain{D,T},
                           x::Vec{D,T})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert !any(dom.staggered) # TODO

    fs = Array{U,D}(undef, dom.n.elts)
    for ic in CartesianIndices(size(fs))
        i = Vec(ic.I) .- 1
        fs[ic] = U(basis(dom, i, x))
    end

    Ws = ntuple(D) do d
        # We know the overlaps of the support of the basis functions
        dv = [dot_basis(dom, d, i, i) for i in 0:dom.n[d]-1]
        ev = [dot_basis(dom, d, i, i+1) for i in 0:dom.n[d]-2]
        SymTridiagonal(dv, ev)
    end

    n = dom.n
    cs = fs
    if D == 1
        cs[:] = Ws[1] \ cs[:]
    elseif D == 2
        for i2 in 1:n[2]
            cs[:,i2] = Ws[1] \ cs[:,i2]
        end
        for i1 in 1:n[1]
            cs[i1,:] = Ws[2] \ cs[i1,:]
        end
    elseif D == 3
        for i3 in 1:n[3], i2 in 1:n[2]
            cs[:,i2,i3] = Ws[1] \ cs[:,i2,i3]
        end
        for i3 in 1:n[3], i1 in 1:n[1]
            cs[i1,:,i3] = Ws[2] \ cs[i1,:,i3]
        end
        for i2 in 1:n[2], i1 in 1:n[1]
            cs[i1,i2,:] = Ws[3] \ cs[i1,i2,:]
        end
    elseif D == 4
        for i4 in 1:n[4], i3 in 1:n[3], i2 in 1:n[2]
            cs[:,i2,i3,i4] = Ws[1] \ cs[:,i2,i3,i4]
        end
        for i4 in 1:n[4], i3 in 1:n[3], i1 in 1:n[1]
            cs[i1,:,i3,i4] = Ws[2] \ cs[i1,:,i3,i4]
        end
        for i4 in 1:n[4], i2 in 1:n[2], i1 in 1:n[1]
            cs[i1,i2,:,i4] = Ws[3] \ cs[i1,i2,:,i4]
        end
        for i3 in 1:n[3], i2 in 1:n[2], i1 in 1:n[1]
            cs[i1,i2,i3,:] = Ws[4] \ cs[i1,i2,i3,:]
        end
    else
        @assert false
    end
    Fun{D,T,U}(dom, cs)
end

end
