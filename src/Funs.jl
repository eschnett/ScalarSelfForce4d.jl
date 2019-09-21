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
struct Fun{D,T,U} <: DenseArray{T,D}
    dom::Domain{D,T}
    coeffs::Array{U,D}
end



# Fun interacts with scalars

# function Base.promote_rule(::Type{Fun{D,T,U}}, ::Type{U}) where
#         {D, T<:Number, U}
#     Fun{D,T,U}
# end
# function Base.convert(::Type{Fun{D,T,U}}, x::U) where {D, T<:Number, U}
#     Fun{D,T,U}(ntuple(d -> x, D))
# end



# Fun is a collection

function Base.eltype(::Type{Fun{D,T,U}})::Type where {D,T,U}
    U
end
function Base.length(f::Fun{D,T,U})::Int where {D,T,U}
    prod(f.dom.n)
end
function Base.ndims(f::Fun{D,T,U})::Int where {D,T,U}
    D
end
function Base.size(f::Fun{D,T,U})::NTuple{D,Int} where {D,T,U}
    f.dom.n.elts
end
function Base.size(f::Fun{D,T,U}, d)::Int where {D,T,U}
    prod(f.dom.n)
end

function Base.getindex(f::Fun{D,T,U}, i::Vec{D,Int})::U where {D,T,U}
    getindex(f.coeffs, i.elts)
end
function Base.getindex(f::Fun{D,T,U}, is...)::U where {D,T,U}
    getindex(f.coeffs, is...)
end



# Fun is a vector space

function Base.zeros(::Type{Fun{D,T,U}}, dom::Domain{D,T})::Fun{D,T,U} where
        {D,T <: Number,U}
    Fun{D,T,U}(dom, zeros(U, dom.n.elts))
end

function Base.:+(f::Fun{D,T,U})::Fun{D,T,U} where {D,T <: Number,U}
    Fun{D,T,U}(f.dom, +f.coeffs)
end
function Base.:-(f::Fun{D,T,U})::Fun{D,T,U} where {D,T <: Number,U}
    Fun{D,T,U}(f.dom, -f.coeffs)
end

function Base.:+(f::Fun{D,T,U}, g::Fun{D,T,U})::Fun{D,T,U} where
        {D,T <: Number,U}
    @assert f.dom == g.dom
    Fun{D,T,U}(f.dom, f.coeffs + g.coeffs)
end
function Base.:-(f::Fun{D,T,U}, g::Fun{D,T,U})::Fun{D,T,U} where
        {D,T <: Number,U}
    @assert f.dom == g.dom
    Fun{D,T,U}(f.dom, f.coeffs - g.coeffs)
end

function Base.:*(a::Number, f::Fun{D,T,U})::Fun{D,T,U} where {D,T <: Number,U}
    Fun{D,T,U}(f.dom, U(a) * f.coeffs)
end
function Base.:*(f::Fun{D,T,U}, a::Number)::Fun{D,T,U} where {D,T <: Number,U}
    Fun{D,T,U}(f.dom, f.coeffs * U(a))
end
function Base.:\(a::Number, f::Fun{D,T,U})::Fun{D,T,U} where {D,T <: Number,U}
    Fun{D,T,U}(f.dom, U(a) \ f.coeffs)
end
function Base.:/(f::Fun{D,T,U}, a::Number)::Fun{D,T,U} where {D,T <: Number,U}
    Fun{D,T,U}(f.dom, f.coeffs / U(a))
end

# function Base.:.+(f::Fun{D,T,U}, c::U)::Fun{D,T,U} where {D, T<:Number, U}
#     Fun{D,T,U}(f.dom, f.coeffs .+ c)
# end
# function Base.:.-(f::Fun{D,T,U}, c::U)::Fun{D,T,U} where {D, T<:Number, U}
#     Fun{D,T,U}(f.dom, f.coeffs .- c)
# end

# function Base.:*(f::Fun{D,T,U}, g::Fun{D,T,U})::U where {D, T<:Number, U}
# TODO: bra and ket
# end

function Base.iszero(f::Fun{D,T,U})::Bool where {D,T <: Number,U}
    all(iszero(f.coeffs))
end
function Base.:(==)(f::Fun{D,T,U}, g::Fun{D,T,U})::Bool where {D,T <: Number,U}
    iszero(f - g)
end

function Base.max(f::Fun{D,T,U})::T where {D,T <: Number,U}
    maximum(f.coeffs)
end
function Base.min(f::Fun{D,T,U})::T where {D,T <: Number,U}
    minimum(f.coeffs)
end
function Base.sum(f::Fun{D,T,U})::T where {D,T <: Number,U}
    dom = f.dom
    Ws = weights(dom)

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
    s
end

function LinearAlgebra.norm(f::Fun{D,T,U}, p::Real = 2) where {D,T <: Number,U}
    if p == Inf
        maximum(abs.(f.coeffs))
    else
        @assert false
    end
end



# Fun are a category

function fidentity(dom::Domain{1,T})::Fun{1,T,T} where {T <: Number}
    if dom.staggered[1]
        dx = (dom.xmax[1] - dom.xmin[1]) / dom.n[1]
        cs = LinRange(dom.xmin[1] + dx[1] / 2, dom.xmax[1] - dx[1] / 2, dom.n[1])
    else
        cs = LinRange(dom.xmin[1], dom.xmax[1], dom.n[1])
    end
    Fun{1,T,T}(dom, cs)
end

function fidentity(::Type{U}, dom::Domain{1,T})::Fun{1,T,U} where {T <: Number,U}
    if dom.staggered[1]
        dx = (dom.xmax[1] - dom.xmin[1]) / dom.n[1]
        cs = LinRange(U(dom.xmin[1] + dx[1] / 2), U(dom.xmax[1] - dx[1] / 2),
                      dom.n[1])
    else
        cs = LinRange(U(dom.xmin[1]), U(dom.xmax[1]), dom.n[1])
    end
    Fun{1,T,U}(dom, cs)
end

# function fidentity(dom::Domain{D,T})::Fun{1,T,Vec{D,T}} where {D, T<:Number}
#     if dom.staggered[1]
#         dx = (dom.xmax[1] - dom.xmin[1]) / dom.n[1]
#         cs = LinRange(dom.xmin[1] + dx[1]/2, dom.xmax[1] - dx[1]/2, dom.n[1])
#     else
#         cs = LinRange(dom.xmin[1], dom.xmax[1], dom.n[1])
#     end
#     Fun{1,T,T}(dom, cs)
# end

# TODO: composition

function fconst(dom::Domain{D,T}, f::U)::Fun{D,T,U} where {D,T <: Number,U}
    cs = fill(f, dom.n.elts)
    Fun{D,T,U}(dom, cs)
end



# Evaluate a function
function (fun::Fun{D,T,U})(x::Vec{D,T})::U where {D,T <: Number,U}
    dom = fun.dom
    @assert !dom.dual
    ix = Vec{D,Int}(ntuple(D) do d
        q = linear(dom.xmin[d], T(0),
                   dom.xmax[d], T(dom.n[d] + dom.staggered[d] - 1), x[d])
        iq = floor(Int, q)
        max(0, min(dom.n[d] + dom.staggered[d] - 2, iq))
    end)

    f = U(0)
    for dic in CartesianIndices(ntuple(d->0:!dom.staggered[d], D))
        di = Vec(dic.I)
        i = ix + di
        if all((i .>= 0) & (i .< dom.n))
            ic = CartesianIndex((i .+ 1).elts)
            f += U(fbasis(dom, i, ix, x) * fun.coeffs[ic])
        end
    end
    f
end

# Create a discretized function by projecting onto the basis functions
#     f(x) = c^j b_j(x)
#     int b_i(x) f(x) = int b_i(x) c^j b_j(x)
#                     = c^j int b_i(x) b_j(x)
export approximate
function approximate(fun, dom::Domain{D,T})::Fun{D,T} where {D,T <: Number}
    U = typeof(fun(dom.xmin))

    fs = Array{U,D}(undef, dom.n.elts)
    for ic in CartesianIndices(size(fs))
        i = Vec(ic.I) .- 1
        f = U(0)
        # To ensure smooth kernels we integrate in two sub-regions for
        # non-staggered directions
        for dic in CartesianIndices(ntuple(d->0:!dom.staggered[d], D))
            di = Vec(dic.I)
            ix = i - di
            if all((ix .>= 0) & (ix .< dom.n + dom.staggered .- 1))
                function kernel(x0::NTuple{D})::U
                    x = Vec{D,T}(x0)
                    fun(x) * U(fbasis(dom, i, ix, x))
                end
                x0 = coord(dom, ix)
                x1 = coord(dom, ix .+ 1)
                n = convert(Vec{D,Int}, 4)
                f += quad(kernel, U, x0.elts, x1.elts, n.elts)
            end
        end
        fs[ic] = f
    end

    Ms = dot_fbasis(dom)

    n = dom.n
    cs = fs
    if D == 1
        cs[:] = Ms[1] \ cs[:]
    elseif D == 2
        for i2 in 1:n[2]
            cs[:,i2] = Ms[1] \ cs[:,i2]
        end
        for i1 in 1:n[1]
            cs[i1,:] = Ms[2] \ cs[i1,:]
        end
    elseif D == 3
        for i3 in 1:n[3], i2 in 1:n[2]
            cs[:,i2,i3] = Ms[1] \ cs[:,i2,i3]
        end
        for i3 in 1:n[3], i1 in 1:n[1]
            cs[i1,:,i3] = Ms[2] \ cs[i1,:,i3]
        end
        for i2 in 1:n[2], i1 in 1:n[1]
            cs[i1,i2,:] = Ms[3] \ cs[i1,i2,:]
        end
    elseif D == 4
        for i4 in 1:n[4], i3 in 1:n[3], i2 in 1:n[2]
            cs[:,i2,i3,i4] = Ms[1] \ cs[:,i2,i3,i4]
        end
        for i4 in 1:n[4], i3 in 1:n[3], i1 in 1:n[1]
            cs[i1,:,i3,i4] = Ms[2] \ cs[i1,:,i3,i4]
        end
        for i4 in 1:n[4], i2 in 1:n[2], i1 in 1:n[1]
            cs[i1,i2,:,i4] = Ms[3] \ cs[i1,i2,:,i4]
        end
        for i3 in 1:n[3], i2 in 1:n[2], i1 in 1:n[1]
            cs[i1,i2,i3,:] = Ms[4] \ cs[i1,i2,i3,:]
        end
    else
        @assert false
    end
    return Fun{D,T,U}(dom, cs)
end



# Approximate a delta function
export approximate_delta
function approximate_delta(dom::Domain{D,T}, x::Vec{D,T})::Fun{D,T,T} where
        {D,T <: Number}
    @assert !any(dom.staggered) # TODO

    fs = zeros(T, dom.n.elts)
    ix = Vec{D,Int}(ntuple(D) do d
        q = linear(dom.xmin[d], T(0), dom.xmax[d], T(dom.n[d] - 1), x[d])
        iq = floor(Int, q)
        max(0, min(dom.n[d] - 2, iq))
    end)
    for dic in CartesianIndices(ntuple(d->0:1, D))
        di = Vec(dic.I)
        i = ix - di
        if all((i .>= 0) & (i .< dom.n))
            ic = CartesianIndex((i .+ 1).elts)
            fs[ic] = fbasis(dom, i, ix, x)
        end
    end

    Ms = dot_fbasis(dom)

    n = dom.n
    cs = fs
    if D == 1
        cs[:] = Ms[1] \ cs[:]
    elseif D == 2
        for i2 in 1:n[2]
            cs[:,i2] = Ms[1] \ cs[:,i2]
        end
        for i1 in 1:n[1]
            cs[i1,:] = Ms[2] \ cs[i1,:]
        end
    elseif D == 3
        for i3 in 1:n[3], i2 in 1:n[2]
            cs[:,i2,i3] = Ms[1] \ cs[:,i2,i3]
        end
        for i3 in 1:n[3], i1 in 1:n[1]
            cs[i1,:,i3] = Ms[2] \ cs[i1,:,i3]
        end
        for i2 in 1:n[2], i1 in 1:n[1]
            cs[i1,i2,:] = Ms[3] \ cs[i1,i2,:]
        end
    elseif D == 4
        for i4 in 1:n[4], i3 in 1:n[3], i2 in 1:n[2]
            cs[:,i2,i3,i4] = Ms[1] \ cs[:,i2,i3,i4]
        end
        for i4 in 1:n[4], i3 in 1:n[3], i1 in 1:n[1]
            cs[i1,:,i3,i4] = Ms[2] \ cs[i1,:,i3,i4]
        end
        for i4 in 1:n[4], i2 in 1:n[2], i1 in 1:n[1]
            cs[i1,i2,:,i4] = Ms[3] \ cs[i1,i2,:,i4]
        end
        for i3 in 1:n[3], i2 in 1:n[2], i1 in 1:n[1]
            cs[i1,i2,i3,:] = Ms[4] \ cs[i1,i2,i3,:]
        end
    else
        @assert false
    end
    Fun{D,T,T}(dom, cs)
end

end
