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
#         {D, T<:Number, U}
#     Fun{D,T,U}
# end
# function Base.convert(::Type{Fun{D,T,U}}, x::U) where {D, T<:Number, U}
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
        {D, T<:Number, U}
    Fun{D,T,U}(dom, zeros(U, dom.n.elts))
end

function Base.:+(f::Fun{D,T,U})::Fun{D,T,U} where {D, T<:Number, U}
    Fun{D,T,U}(f.dom, +f.elts)
end
function Base.:-(f::Fun{D,T,U})::Fun{D,T,U} where {D, T<:Number, U}
    Fun{D,T,U}(f.dom, -f.elts)
end

function Base.:+(f::Fun{D,T,U}, g::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U}
    @assert f.dom == g.dom
    Fun{D,T,U}(f.dom, f.coeffs + g.coeffs)
end
function Base.:-(f::Fun{D,T,U}, g::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U}
    @assert f.dom == g.dom
    Fun{D,T,U}(f.dom, f.coeffs - g.coeffs)
end

function Base.:*(a::Number, f::Fun{D,T,U})::Fun{D,T,U} where {D, T<:Number, U}
    Fun{D,T,U}(f.dom, U(a) * f.coeffs)
end
function Base.:*(f::Fun{D,T,U}, a::Number)::Fun{D,T,U} where {D, T<:Number, U}
    Fun{D,T,U}(f.dom, f.coeffs * U(a))
end
function Base.:\(a::Number, f::Fun{D,T,U})::Fun{D,T,U} where {D, T<:Number, U}
    Fun{D,T,U}(f.dom, U(a) \ f.coeffs)
end
function Base.:/(f::Fun{D,T,U}, a::Number)::Fun{D,T,U} where {D, T<:Number, U}
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

function Base.max(f::Fun{D,T,U})::T where {D, T<:Number, U}
    maximum(f.coeffs)
end
function Base.min(f::Fun{D,T,U})::T where {D, T<:Number, U}
    minimum(f.coeffs)
end
function Base.sum(f::Fun{D,T,U})::T where {D, T<:Number, U}
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

function LinearAlgebra.norm(f::Fun{D,T,U}, p::Real=2) where {D, T<:Number, U}
    if p == Inf
        maximum(abs.(f.coeffs))
    else
        @assert false
    end
end



# Fun are a category

# TODO: composition

function fidentity(dom::Domain{1,T})::Fun{1,T,T} where {T<:Number}
    if dom.staggered[1]
        dx = (dom.xmax[1] - dom.xmin[1]) / dom.n[1]
        cs = LinRange(dom.xmin[1] + dx[1]/2, dom.xmax[1] - dx[1]/2, dom.n[1])
    else
        cs = LinRange(dom.xmin[1], dom.xmax[1], dom.n[1])
    end
    Fun{1,T,T}(dom, cs)
end

function fidentity(::Type{U}, dom::Domain{1,T})::Fun{1,T,U} where {T<:Number, U}
    if dom.staggered[1]
        dx = (dom.xmax[1] - dom.xmin[1]) / dom.n[1]
        cs = LinRange(U(dom.xmin[1] + dx[1]/2), U(dom.xmax[1] - dx[1]/2),
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

function fconst(dom::Domain{D,T}, f::U)::Fun{D,T,U} where {D, T<:Number, U}
    cs = fill(f, dom.n.elts)
    Fun{D,T,U}(dom, cs)
end



# Evaluate a function
function (fun::Fun{D,T,U})(x::Vec{D,T})::U where {D, T<:Number, U}
    # f = U(0)
    # for ic in CartesianIndices(size(fun.coeffs))
    #     i = Vec(ic.I) .- 1
    #     f += U(basis(fun.dom, i, x) * fun.coeffs[ic])
    # end
    # f

    dom = fun.dom
    num_regions = 2 .- convert(Vec{D,Int}, dom.staggered)
    q = Vec{D,T}(ntuple(d -> linear(dom.xmin[d], T(0),
                                    dom.xmax[d], T(dom.n[d]+dom.staggered[d]-1),
                                    x[d]), D))
    # using round instead of floor to avoid bias
    i = Vec{D,Int}(ntuple(d -> max(0, min(dom.n[d] - num_regions[d],
                                          round(Int, q[d] - T(1)/2))), D))

    f = U(0)
    for rc in CartesianIndices(num_regions.elts)
        r = Vec(rc.I) .- 1
        j = i + r
        jc = CartesianIndex(j.elts .+ 1)
        f += U(basis1(dom, j, x) * fun.coeffs[jc])
    end
    f
end

# Create a discretized function by projecting onto the basis functions
export approximate
function approximate(fun, dom::Domain{D,T})::Fun{D,T} where {D, T<:Number}
    U = typeof(fun(dom.xmin))

    # To ensure smooth kernels, VC direction are integrated in 2
    # steps, CC regions in 1 step
    num_regions = 2 .- convert(Vec{D,Int}, dom.staggered)
    offset = (T(1)/2) * convert(Vec{D,T}, dom.staggered)

    fs = Array{U,D}(undef, dom.n.elts)
    for ic in CartesianIndices(size(fs))
        i = Vec(ic.I) .- 1
        f = U(0)
        function kernel(x0::NTuple{D})::U
            x = Vec{D,T}(x0)
            fun(x) * U(basis(dom, i, x))
        end
        for rc in CartesianIndices(num_regions.elts)
            r = Vec(rc.I) .- 1
            j = i - r
            if all(dom.staggered .| ((j .>= 0) .& (j .< dom.n .- 1)))
                x0 = coord(dom, j - offset)
                x1 = coord(dom, j - offset .+ 1)
                n = convert(Vec{D,Int}, 8)
                f += quad(kernel, U, x0.elts, x1.elts, n.elts)
            end
        end
        fs[ic] = f
    end

    Ws = weights(dom)

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



# Approximate a delta function
export approximate_delta
function approximate_delta(dom::Domain{D,T}, x::Vec{D,T})::Fun{D,T,T} where
        {D, T<:Number}
    @assert !any(dom.staggered) # TODO

    fs = Array{T,D}(undef, dom.n.elts)
    for ic in CartesianIndices(size(fs))
        i = Vec(ic.I) .- 1
        fs[ic] = basis(dom, i, x)
    end

    Ws = weights(dom)

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
    Fun{D,T,T}(dom, cs)
end



export solve_dAlembert_Dirichlet
function solve_dAlembert_Dirichlet(pot::Fun{D,T,U},
                                   bvals::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U}
    dom = pot.dom
    @assert bvals.dom == dom
    @assert !any(dom.staggered)

    n = dom.n
    dx = spacing(dom)
    dx2 = dx .* dx

    # TODO: use linear Cartesian index, calculate di

    sol = similar(pot.coeffs)
    if D == 4
        # Initial and boundary conditions
        sol[1,:,:,:] = bvals.coeffs[1,:,:,:]
        sol[end,:,:,:] = bvals.coeffs[end,:,:,:]
        sol[:,1,:,:] = bvals.coeffs[:,1,:,:]
        sol[:,end,:,:] = bvals.coeffs[:,end,:,:]
        sol[:,:,1,:] = bvals.coeffs[:,:,1,:]
        sol[:,:,end,:] = bvals.coeffs[:,:,end,:]
        sol[:,:,:,1] = bvals.coeffs[:,:,:,1]
        sol[:,:,:,2] = bvals.coeffs[:,:,:,2]
        # d'Alembert operator
        for i4=2:n[4]-1
            for i3=2:n[3]-1, i2=2:n[2]-1, i1=2:n[1]-1
                sol[i1,i2,i3,i4+1] =
                    (- sol[i1,i2,i3,i4-1] + 2*sol[i1,i2,i3,i4]
                     + dx2[4] * (
                         + (sol[i1-1,i2,i3,i4] - 2*sol[i1,i2,i3,i4] + sol[i1+1,i2,i3,i4]) / dx2[1]
                         + (sol[i1,i2-1,i3,i4] - 2*sol[i1,i2,i3,i4] + sol[i1,i2+1,i3,i4]) / dx2[2]
                         + (sol[i1,i2,i3-1,i4] - 2*sol[i1,i2,i3,i4] + sol[i1,i2,i3+1,i4]) / dx2[3]
                         - pot.coeffs[i1,i2,i3,i4]))
            end
        end
    else
        @assert false
    end

    Fun{D,T,U}(dom, sol)
end

end
