module ScalarSelfForce4d

using LinearAlgebra
using Reexport
using SparseArrays



include("Defs.jl")
include("Quadrature.jl")
include("Vecs.jl")
include("Domains.jl")


@reexport using .Defs
@reexport using .Domains
@reexport using .Quadrature
@reexport using .Vecs



################################################################################



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

# Integration weighs for basis functions (assuming a diagonal weight matrix)
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



################################################################################

# Discretized functions

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

function fconst(dom::Domain{D,T}, f::U)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    cs = fill(f, dom.n.elts)
    Fun{D,T,U}(dom, cs)
end

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



################################################################################

# Derivatives

# Derivative of basis functions   dϕ^i/dϕ^j
function deriv_basis(dom::Domain{D,T}, d::Int, i::Int, j::Int)::T where
        {D, T<:Number}
    @assert !any(dom.staggered) # TODO
    dx = (fun.dom.xmax[d] - fun.dom.xmin[d]) / (fun.dom.n[d] - 1)
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



export deriv
function deriv(dom::Domain{D,T}, d::Int)::Tridiagonal{T} where {D, T<:Number}
    @assert !any(dom.staggered) # TODO
    # We know the overlaps of the support of the basis functions
    n = dom.n[d] - 1
    dlv = [deriv_basis(dom, d, i, i-1) for i in 1:n]
    dv = [deriv_basis(dom, d, i, i) for i in 0:n]
    duv = [deriv_basis(dom, d, i, i+1) for i in 0:n-1]
    Tridiagonal(dlv, dv, duv)
end



function deriv(fun::Fun{D,T,U}, dir::Int)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert 1 <= dir <= D
    @assert !any(fun.dom.staggered) # TODO
    dx = (fun.dom.xmax[dir] - fun.dom.xmin[dir]) / (fun.dom.n[dir] - 1)
    cs = fun.coeffs
    dcs = similar(cs)
    n = size(dcs, dir)

    # TODO: use linear Cartesian index, calculate di

    inner_indices = CartesianIndices(ntuple(d -> size(dcs,d), dir - 1))
    outer_indices = CartesianIndices(ntuple(d -> size(dcs,dir+d), D - dir))

    for oi in outer_indices
        for ii in inner_indices
            dcs[ii,1,oi] = (cs[ii,2,oi] - cs[ii,1,oi]) / dx
        end
        for i in 2:n-1
            for ii in inner_indices
                dcs[ii,i,oi] = (cs[ii,i+1,oi] - cs[ii,i-1,oi]) / 2dx
            end
        end
        for ii in inner_indices
            dcs[ii,n,oi] = (cs[ii,n,oi] - cs[ii,n-1,oi]) / dx
        end
    end

    Fun{D,T,U}(fun.dom, dcs)
end

export deriv2
function deriv2(fun::Fun{D,T,U}, dir::Int)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert 1 <= dir <= D
    @assert !any(fun.dom.staggered) # TODO
    dx2 = ((fun.dom.xmax[dir] - fun.dom.xmin[dir]) / (fun.dom.n[dir] - 1)) ^ 2
    cs = fun.coeffs
    dcs = similar(cs)
    n = size(dcs, dir)

    inner_indices = CartesianIndices(ntuple(d -> size(dcs,d), dir - 1))
    outer_indices = CartesianIndices(ntuple(d -> size(dcs,dir+d), D - dir))

    for oi in outer_indices
        for ii in inner_indices
            dcs[ii,1,oi] = (cs[ii,1,oi] - 2*cs[ii,2,oi] + cs[ii,3,oi]) / dx2
        end
        for i in 2:n-1
            for ii in inner_indices
                dcs[ii,i,oi] =
                    (cs[ii,i-1,oi] - 2*cs[ii,i,oi] + cs[ii,i+1,oi]) / dx2
            end
        end
        for ii in inner_indices
            dcs[ii,n,oi] = (cs[ii,n-2,oi] - 2*cs[ii,n-1,oi] + cs[ii,n,oi]) / dx2
        end
    end

    Fun{D,T,U}(fun.dom, dcs)
end

function deriv2(fun::Fun{D,T,U}, dir1::Int, dir2::Int)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert 1 <= dir1 <= D
    @assert 1 <= dir2 <= D
    if dir1 == dir2
        deriv2(fun, dir1)
    else
        deriv(deriv(fun, dir1), dir2)
    end
end



################################################################################

# Operators

export Op
struct Op{D,T,U} <: AbstractArray{U, 2}
    dom::Domain{D,T}
    mat::SparseMatrixCSC{U,Int}
end



# Op interacts with scalars

# function Base.promote_rule(::Type{Op{D,T,U}}, ::Type{U}) where
#         {D, T<:Number, U<:Number}
#     Op{D,T,U}
# end
# function Base.convert(::Type{Op{D,T,U}}, x::U) where {D, T<:Number, U<:Number}
#     Op{D,T,U}(ntuple(d -> x, D))
# end

# Fun is a collection

function Base.eltype(A::Op{D,T,U})::Type where {D, T, U}
    eltype(A.mat)
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

function Base.zeros(::Type{Op{D,T,U}}, dom::Domain{D,T})::Op{D,T,U} where
        {D, T, U<:Number}
    len = prod(dom.n)
    mat = spzeros(U, len, len)
    Op{D,T,U}(dom, mat)
end
convert

function Base.:+(A::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U<:Number}
    Op{D,T,U}(A.dom, +A.mat)
end
function Base.:-(A::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U<:Number}
    Op{D,T,U}(A.dom, -A.mat)
end

function Base.:+(A::Op{D,T,U}, B::Op{D,T,U})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert A.dom == B.dom
    Op{D,T,U}(A.dom, A.mat + B.mat)
end
function Base.:-(A::Op{D,T,U}, B::Op{D,T,U})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert A.dom == B.dom
    Op{D,T,U}(A.dom, A.mat - B.mat)
end

function Base.:*(a::Number, A::Op{D,T,U})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    Op{D,T,U}(A.dom, U(a) * A.mat)
end
function Base.:*(A::Op{D,T,U}, a::Number)::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    Op{D,T,U}(A.dom, A.mat * U(a))
end
function Base.:\(a::Number, A::Op{D,T,U})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    Op{D,T,U}(A.dom, U(a) \ A.mat)
end
function Base.:/(A::Op{D,T,U}, a::Number)::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    Op{D,T,U}(A.dom, A.mat / U(a))
end



function Base.:*(A::Op{D,T,U}, B::Op{D,T,U})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert A.dom == B.dom
    Op{D,T,U}(A.dom, A.mat * B.mat)
end



function Base.zero(::Type{Op{D,T,U}}, dom::Domain{D,T})::Op{D,T,U} where
        {D, T, U<:Number}
    zeros(Op{D,T,U}, dom)
end
function Base.one(::Type{Op{D,T,U}}, dom::Domain{D,T})::Op{D,T,U} where
        {D, T, U<:Number}
    n = dom.n

    str = Vec{D,Int}(ntuple(dir -> dir==1 ? 1 : prod(n[d] for d in 1:dir-1), D))
    len = prod(n)
    idx(i::Vec{D,Int}) = 1 + sum(i[d] * str[d] for d in 1:D)

    I = Int[]
    J = Int[]
    V = U[]
    function ins!(i, j, v)
        push!(I, idx(i))
        push!(J, idx(j))
        push!(V, v)
    end
    for ic in CartesianIndices(dom.n.elts)
        i = Vec(ic.I) .- 1
        ins!(i, i, U(1))
    end
    mat = sparse(I, J, V, len, len)
    Op{D,T,U}(dom, mat)
end

function Base.:*(op::Op{D,T,U}, rhs::Fun{D,T,U})::Fun{D,T,U} where
        {D, T, U<:Number}
    dom = rhs.dom
    @assert op.dom == dom

    res = reshape(op.mat * reshape(rhs.coeffs, :), dom.n.elts)
    Fun{D,T,U}(dom, res)
end

# function Base.:*(lhs::Fun{D,T,U}, op::Op{D,T,U})::Fun{D,T,U} where
#         {D, T, U<:Number}
#     dom = rhs.dom
#     @assert op.dom == dom
# 
#     TODO: bra and ket
#     res = reshape(reshape(lhs.coeffs, :) * op.mat, dom.n.elts)
#     Fun{D,T,U}(dom, res)
# end

function Base.:\(op::Op{D,T,U}, rhs::Fun{D,T,U})::Fun{D,T,U} where
        {D, T, U<:Number}
    dom = rhs.dom
    @assert op.dom == dom

    M = op.mat
    if T <: Union{Float32, Float64}
        # do nothing
    else
        @info "Converting sparse to full matrix..."
        M = Matrix(M)
    end
    sol = reshape(M \ reshape(rhs.coeffs, :), dom.n.elts)
    Fun{D,T,U}(dom, sol)
end

# function Base.:\(op::Op{D,T,U}, rhs::Fun{D,T,U})::Fun{D,T,U} where
#         {D, T, U<:Number}
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
        {D, T<:Number, U<:Number}
    dom = bnd.dom
    @assert iop.dom == dom
    @assert bop.dom == dom

    id = one(Op{D,T,U}, dom)
    int = id - bnd
    int * iop + bnd * bop
end
function mix_op_bc(bnd::Op{D,T,U},
                   rhs::Fun{D,T,U}, bvals::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    dom = bnd.dom
    @assert rhs.dom == dom
    @assert bvals.dom == dom

    id = one(Op{D,T,U}, dom)
    int = id - bnd
    int * rhs + bnd * bvals
end



export boundary
function boundary(::Type{U}, dom::Domain{D,T})::Op{D,T,U} where {D, T, U<:Number}
    n = dom.n

    str = Vec{D,Int}(ntuple(dir -> dir==1 ? 1 : prod(n[d] for d in 1:dir-1), D))
    len = prod(n)
    idx(i::Vec{D,Int}) = 1 + sum(i[d] * str[d] for d in 1:D)

    I = Int[]
    J = Int[]
    V = U[]
    function ins!(i, j, v)
        push!(I, idx(i))
        push!(J, idx(j))
        push!(V, v)
    end
    for ic in CartesianIndices(dom.n.elts)
        i = Vec(ic.I) .- 1
        if any(i .== 0) || any(i .== n .- 1)
            ins!(i, i, U(1))
        end
    end
    mat = sparse(I, J, V, len, len)
    Op{D,T,U}(dom, mat)
end

export dirichlet
# TODO: Is this correct?
const dirichlet = boundary

export laplace
function laplace(::Type{U}, dom::Domain{D,T})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert !any(dom.staggered) # TODO
    n = dom.n
    dx2 = Vec(ntuple(d -> ((dom.xmax[d] - dom.xmin[d]) / (n[d] - 1)) ^ 2, D))

    str = Vec{D,Int}(ntuple(dir -> dir==1 ? 1 : prod(n[d] for d in 1:dir-1), D))
    len = prod(n)
    idx(i::Vec{D,Int}) = 1 + sum(i[d] * str[d] for d in 1:D)

    I = Int[]
    J = Int[]
    V = U[]
    function ins!(i, j, v)
        @assert all(0 .<= i .< n)
        @assert all(0 .<= j .< n)
        push!(I, idx(i))
        push!(J, idx(j))
        push!(V, v)
    end
    for ic in CartesianIndices(dom.n.elts)
        i = Vec(ic.I) .- 1
        for dir in 1:D
            di = Vec(ntuple(d -> d==dir ? 1 : 0, D))
            if i[dir] == 0
                j = i + di
            elseif i[dir] == n[dir] - 1
                j = i - di
            else
                j = i
            end
            ins!(i, j - di, 1 / U(dx2[dir]))
            ins!(i, j, -2 / U(dx2[dir]))
            ins!(i, j + di, 1 / U(dx2[dir]))
        end
    end
    mat = sparse(I, J, V, len, len)
    Op{D,T,U}(dom, mat)
end



export boundaryIV
function boundaryIV(::Type{U}, dom::Domain{D,T})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert !any(dom.staggered) # TODO
    n = dom.n

    str = Vec{D,Int}(ntuple(dir -> dir==1 ? 1 : prod(n[d] for d in 1:dir-1), D))
    len = prod(n)
    idx(i::Vec{D,Int}) = 1 + sum(i[d] * str[d] for d in 1:D)

    I = Int[]
    J = Int[]
    V = U[]
    function ins!(i, j, v)
        push!(I, idx(i))
        push!(J, idx(j))
        push!(V, v)
    end
    for ic in CartesianIndices(dom.n.elts)
        i = Vec(ic.I) .- 1
        isbnd = false
        for d in 1:D
            if d < D
                isbnd |= i[d] == 0 || i[d] == n[d] - 1
            else
                isbnd |= i[d] <= 1
            end
        end
        if isbnd
            ins!(i, i, U(1))
        end
    end
    mat = sparse(I, J, V, len, len)
    Op{D,T,U}(dom, mat)
end

export dirichletIV
# TODO: Is this correct?
const dirichletIV = boundaryIV

export dAlembert
function dAlembert(::Type{U}, dom::Domain{D,T})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert !any(dom.staggered) # TODO
    n = dom.n
    dx2 = Vec(ntuple(d -> ((dom.xmax[d] - dom.xmin[d]) / (n[d] - 1)) ^ 2, D))

    str = Vec{D,Int}(ntuple(dir -> dir==1 ? 1 : prod(n[d] for d in 1:dir-1), D))
    len = prod(n)
    idx(i::Vec{D,Int}) = 1 + sum(i[d] * str[d] for d in 1:D)

    I = Int[]
    J = Int[]
    V = U[]
    function ins!(i, j, v)
        @assert all(0 .<= i .< n)
        @assert all(0 .<= j .< n)
        push!(I, idx(i))
        push!(J, idx(j))
        push!(V, v)
    end
    for ic in CartesianIndices(dom.n.elts)
        i = Vec(ic.I) .- 1
        for dir in 1:D
            s = bitsign(dir == D)
            di = Vec(ntuple(d -> d==dir ? 1 : 0, D))
            if dir < D
                if i[dir] == 0
                    j = i + di
                elseif i[dir] == n[dir] - 1
                    j = i - di
                else
                    j = i
                end
            else
                if i[dir] == 0
                    j = i + di
                elseif i[dir] == 1
                    j = i
                else
                    j = i - di
                end
            end
            ins!(i, j - di, s / U(dx2[dir]))
            ins!(i, j, -2s / U(dx2[dir]))
            ins!(i, j + di, s / U(dx2[dir]))
        end
    end
    mat = sparse(I, J, V, len, len)
    Op{D,T,U}(dom, mat)
end

export solve_dAlembert_Dirichlet
function solve_dAlembert_Dirichlet(pot::Fun{D,T,U},
                                   bvals::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    dom = pot.dom
    @assert bvals.dom == dom
    @assert !any(dom.staggered) # TODO

    n = dom.n
    dx2 = Vec(ntuple(d -> ((dom.xmax[d] - dom.xmin[d]) / (n[d] - 1)) ^ 2, D))

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



################################################################################

# Discrete differential forms

# Derivative of a 0-form
function deriv0(u0::Fun{D,T,U})::NTuple{D,Fun{D,T,U}} where {D,T,U}
    if D == 2
        dom0 = u0.dom
        s0 = dom0.staggered
        n0 = dom0.n
        di = ntuple(dir -> CartesianIndex(ntuple(d -> d==dir, D)), D)
        dx = ntuple(d -> (dom0.xmax[d] - dom0.xmin[d]) / (n0[d] - 1), D)
        @assert s0 == Vec((false, false))
        s1x = Vec((true, false))
        s1t = Vec((false, true))
        n1x = Vec((n0[1]-s1x[1], n0[2]-s1x[2]))
        n1t = Vec((n0[1]-s1t[1], n0[2]-s1t[2]))
        dom1x = Domain{D, T}(s1x, dom0.metric, n1x, dom0.xmin, dom0.xmax)
        dom1t = Domain{D, T}(s1t, dom0.metric, n1t, dom0.xmin, dom0.xmax)
        cs0 = u0.coeffs
        dcs1x = Array{U}(undef, n1x.elts)
        for i in CartesianIndices(size(dcs1x))
            dcs1x[i] = (cs0[i + di[1]] - cs0[i]) / dx[1]
        end
        dcs1t = Array{U}(undef, n1t.elts)
        for i in CartesianIndices(size(dcs1t))
            dcs1t[i] = (cs0[i + di[2]] - cs0[i]) / dx[2]
        end
        return (Fun{D,T,U}(dom1x, dcs1x), Fun{D,T,U}(dom1t, dcs1t))
    else
        @assert false
    end
end

# Hodge star of a 1-form
function star1(u1::NTuple{D, Fun{D,T,U}})::NTuple{D, Fun{D,T,U}} where {D,T,U}
    if D == 2
        u1x, u1t = u1
        dom1x = u1x.dom
        dom1t = u1t.dom
        s1x = dom1x.staggered
        s1t = dom1t.staggered
        @assert s1x == Vec((true, false))
        @assert s1t == Vec((false, true))
        n1x = dom1x.n
        n1t = dom1t.n
        n = Vec((n1x[1] + s1x[1], n1x[2] + s1x[2]))
        di = ntuple(dir -> CartesianIndex(ntuple(d -> d==dir, D)), D)
        @assert n1x == Vec((n[1] - s1x[1], n[2] - s1x[2]))
        @assert n1t == Vec((n[1] - s1t[1], n[2] - s1t[2]))
        cs1x = u1x.coeffs
        cs1t = u1t.coeffs
        scs1x = Array{U}(undef, n1x.elts)
        for i in CartesianIndices(size(scs1x))
            s = U(0)
            c = U(0)
            if i[2] > 1
                s += cs1t[i-di[2]] + cs1t[i-di[2]+di[1]]
                c += 2
            end
            if i[2] < n[2]
                s += cs1t[i] + cs1t[i+di[1]]
                c += 2
            end
            scs1x[i] = - s / c
        end
        scs1t = Array{U}(undef, n1t.elts) 
        for i in CartesianIndices(size(scs1t))
            s = U(0)
            c = U(0)
            if i[1] > 1
                s += cs1x[i-di[1]] + cs1x[i-di[1]+di[2]]
                c += 2
            end
            if i[1] < n[1]
                s += cs1x[i] + cs1x[i+di[2]]
                c += 2
            end
            scs1t[i] = + s / c
        end
        return (Fun{D,T,U}(dom1x, scs1x), Fun{D,T,U}(dom1t, scs1t))
   else
        @assert false
    end
end

# Wedge of two 1-forms
function wedge11(u1::NTuple{D, Fun{D,T,U}},
                 v1::NTuple{D, Fun{D,T,U}})::Fun{D,T,U} where {D,T,U}
    if D == 2
        u1x, u1t = u1
        v1x, v1t = v1
        @assert u1x.dom.staggered == Vec((true, false))
        @assert u1t.dom.staggered == Vec((false, true))
        @assert v1x.dom.staggered == Vec((true, false))
        @assert v1t.dom.staggered == Vec((false, true))
        n = Vec((u1x.dom.n[1] + u1x.dom.staggered[1],
                 u1x.dom.n[2] + u1x.dom.staggered[2]))
        @assert u1x.dom.n == Vec((n[1] - u1x.dom.staggered[1],
                                  n[2] - u1x.dom.staggered[2]))
        @assert u1t.dom.n == Vec((n[1] - u1t.dom.staggered[1],
                                  n[2] - u1t.dom.staggered[2]))
        @assert v1x.dom.n == Vec((n[1] - v1x.dom.staggered[1],
                                  n[2] - v1x.dom.staggered[2]))
        @assert v1t.dom.n == Vec((n[1] - v1t.dom.staggered[1],
                                  n[2] - v1t.dom.staggered[2]))
        di = ntuple(dir -> CartesianIndex(ntuple(d -> d==dir, D)), D)
        ucs1x = u1x.coeffs
        ucs1t = u1t.coeffs
        vcs1x = u1x.coeffs
        vcs1t = u1t.coeffs
        s2 = Vec((true, true))
        n2 = Vec((n[1] - s2[1], n[2] - s2[2]))
        dom2 = Domain{D,T}(s2, u1x.dom.metric, n2, u1x.dom.xmin, u1x.dom.xmax)
        wcs2 = Array{U}(undef, n2.elts)
        for i in CartesianIndices(size(wcs2))
            wcs2[i] = (+ (+ ucs1t[i] * vcs1x[i]
                          + ucs1t[i+di[1]] * vcs1x[i]
                          + ucs1t[i+di[1]] * vcs1x[i+di[2]]
                          + ucs1t[i] * vcs1x[i+di[2]])
                       - (+ ucs1x[i] * vcs1t[i]
                          + ucs1x[i+di[2]] * vcs1t[i]
                          + ucs1x[i+di[2]] * vcs1t[i+di[1]]
                          + ucs1x[i] * vcs1t[i+di[1]])) / 8
        end
        return Fun{D,T,U}(dom2, wcs2)
    else
        @assert false
    end
end



################################################################################

# Scalar wave equation

export scalarwave_energy
function scalarwave_energy(phi::Fun{D,T,T})::Fun{D,T,T} where {D,T<:Number}
    @assert all(!phi.dom.staggered)

    dphi = deriv0(phi)
    sdphi = star1(dphi)
    eps = wedge11(dphi, sdphi)

    eps
end

function scalarwave_energy1(phi::Fun{D,T,T})::Fun{D,T,T} where {D,T<:Number}
    @assert all(!phi.dom.staggered)
    sdom = makestaggered(phi.dom)

    n = sdom.n
    dx = Vec(ntuple(d -> (sdom.xmax[d] - sdom.xmin[d]) / n[d], D))
    di = ntuple(dir -> Vec(ntuple(d -> Int(d==dir), D)), D)

    eps = Array{T}(undef, n.elts)
    if D == 4
        for ic in CartesianIndices(size(eps))
            i = Vec(ic.I)
            s = T(0)
            # x
            for c in 0:1, b in 0:1, a in 0:1
                im = i +         a*di[2] + b*di[3] + c*di[4];
                ip = i + di[1] + a*di[2] + b*di[3] + c*di[4];
                s += ((+ phi.coeffs[CartesianIndex(ip.elts)]
                       - phi.coeffs[CartesianIndex(im.elts)]) / dx[1]) ^2 / 8
            end
            # y
            for c in 0:1, b in 0:1, a in 0:1
                im = i +         a*di[1] + b*di[3] + c*di[4];
                ip = i + di[2] + a*di[1] + b*di[3] + c*di[4];
                s += ((+ phi.coeffs[CartesianIndex(ip.elts)]
                       - phi.coeffs[CartesianIndex(im.elts)]) / dx[2]) ^2 / 8
            end
            # z
            for c in 0:1, b in 0:1, a in 0:1
                im = i +         a*di[1] + b*di[2] + c*di[4];
                ip = i + di[3] + a*di[1] + b*di[2] + c*di[4];
                s += ((+ phi.coeffs[CartesianIndex(ip.elts)]
                       - phi.coeffs[CartesianIndex(im.elts)]) / dx[3]) ^2 / 8
            end
            # t
            for c in 0:1, b in 0:1, a in 0:1
                im = i +         a*di[1] + b*di[2] + c*di[3];
                ip = i + di[4] + a*di[1] + b*di[2] + c*di[3];
                s += ((+ phi.coeffs[CartesianIndex(ip.elts)]
                       - phi.coeffs[CartesianIndex(im.elts)]) / dx[4]) ^2 / 8
            end
            eps[ic] = s / 2
        end
    else
        @assert false
    end

    Fun{D,T,T}(sdom, eps)
end

# Energy conservation:
#
# Equations of motion, second order in time:
# phi[i,j+1] = 2 phi[i,j] - phi[i,j-1] + (phi[i-1,j] - 2 phi[i,j] + phi[i+1,j])
#            = - phi[i,j-1] + phi[i-1,j] + phi[i+1,j]
# 
# Equations of motion, first order in time:
# psi[i,j]   = phi[i,j+1] - phi[i,j]
#            = phi[i-1,j] + phi[i+1,j] - phi[i,j] - phi[i,j-1]
#
# phi[i,j+1] = phi[i,j] + psi[i,j]
# psi[i,j+1] = phi[i-1,j+1] + phi[i+1,j+1] - phi[i,j+1] - phi[i,j]
#            = phi[i-1,j+1] + phi[i+1,j+1] - phi[i,j+1] - phi[i,j+1] + psi[i,j]
#            = psi[i,j] + phi[i-1,j+1] - 2 phi[i,j+1] + phi[i+1,j+1]
# 
# Energy density:
# 1/2 eps[i,j] = (phi[i+1,j] - phi[i-1,j])^2 + (phi[i,j+1] - phi[i,j-1])^2
#              = + (phi[i+1,j] - phi[i-1,j])^2
#                + (2 phi[i,j] + 2 psi[i,j] - phi[i-1,j] - phi[i+1,j])^2
#              = 4 phi,x[i,j]^2
#                + 4 psi[i,j]^2 + phi,xx[i,j]^2 + 2 psi[i,j] phi,xx[i,j]
# 2 eps[i,j] = + psi[i,j]^2
#              + phi,x[i,j]^2 + 1/4 phi,xx[i,j]^2 + 1/2 psi[i,j] phi,xx[i,j]

# Discrete differential forms:
#    dphi = [phi[i,j+1] - phi[i,j], phi[i+1,j] - phi[i,j]]
#
#    *dphi = 1/4 [+ (dphi_x[i-1,j] + dphi_x[i,j] + dphi_x[i-1,j+1] + dphi_x[i,j+1]),
#                 - (dphi_t[i,j-1] + dphi_t[i+1,j-1] + dphi_t[i,j] + dphi_t[i+1,j])]
#
#    dphi ∧ *dphi = 1/8 (+ dphi_t[i,j] *dphi_x[i,j]
#                        + dphi_t[i+1,j] *dphi_x[i,j]
#                        + dphi_t[i,j] *dphi_x[i+1,j]
#                        + dphi_t[i+1,j] *dphi_x[i+1,j]
#                        - dphi_x[i,j] *dphi_t[i,j]
#                        - dphi_x[i,j+1] *dphi_t[i,j]
#                        - dphi_x[i,j] *dphi_t[i+1,j]
#                        - dphi_x[i,j+1] *dphi_t[i+1,j])
#                 = 1/8 (

#    L = 1/2 [(phi[i+1,j] - phi[i,j])^2 - (phi[i,j+1] - phi[i,j])^2]
# 
# Lagrangian:
# 


################################################################################

# Particles

# Equations of motion for a point particle (arXiv:1102.0259, (17.1) - (17.8)
#
# Variables:
#    phi(x^a)
#    m_0, q, m(tau), z^a(tau), u^a(tau)
# Properties:
#    u^2    = -1
#    m(tau) = m_0 - q phi(z^a(tau))
#    p^a    = m u^a
# Equations of motion:
#    a^a       = q/m (eta^ab - u^a u^b) (d_b phi)(z^a)
#    dz^a/dtau = u^a
#    du^a/dtau =? a^a
#    dm/dtau   = -q u^a (d_a phi)(z^a)
# Action:
#    dtau = sqrt[- eta_ab dz^a/dlambda dz^b/dlambda] dlambda
#    S = + 1/2 Int eta^ab (d_a phi) (d_b phi)
#        - 4 pi q Int phi(x^a) delta(x^a - z^a(tau)) dx^4 dtau
#        + 4 pi m_0 Int dtau
#    S = + 1/2 Int eta^ab (d_a phi) (d_b phi)
#        - 4 pi q Int phi(x^a) delta3(x^a - z^a(t)) dtau/dt dx^4
#        + 4 pi m_0 Int dtau/dt dt
# Generalized coordinates:
#    phi(x^a)
#    z^a
# Momenta:
#    psi(x^a) = d_t phi(x^a) = n^b d_b phi(x^a)
#    p_a(tau) = 4 pi m dt/dtau u_a(tau)
#    n^b d_b phi(x^a) + 4 pi m dtau/dt u_a(tau)
#    n^b d_b phi(x^a) + 4 pi m delta(x^4) dtau/dt u_a(t)
# Hamiltonian:
#    H = dphi/dt psi + u^a p_a (dtau/dt)^2 - L
#    H1 = n^a (d_a phi) n^b (d_b phi) - 1/2 eta^ab (d_a phi) (d_b phi)
#       = (n^a n^b - 1/2 eta^ab) (d_a phi) (d_b phi)
#    H2 = 4 pi q phi(x^a) delta(x^a - z^a(tau))
#    H3 = u^a p_a (dtau/dt)^2 - 4 pi m_0 dtau/dt
#       = 4 pi m u^a u_a dtau/dt - 4 pi m_0 dtau/dt
#       = 4 pi (m_0 - q phi(z)) dtau/dt - 4 pi m_0 dtau/dt
#       = - 4 pi q phi(z) dtau/dt



export Particle
struct Particle{D,T}
    dom::Domain{D,T}
    mass::T
    charge::T
    pos::Vec{D,T}
    vel::Vec{D,T}
end

# TODO: Particle is a vector space

export particle_density
function particle_density(p::Particle{D,T})::Fun{D,T,T} where {D,T}
    p.charge * approximate_delta(T, p.dom, p.pos)
end

export particle_acceleration
function particle_acceleration(p::Particle{D,T},
                               pot::Fun{D,T,T})::Vec{D,T} where {D,T}
    dom = p.dom
    @assert pot.dom == dom

    rho = particle_density(p)

    grad_pot = ntuple(d -> deriv(pot, d), D)

    acc = ntuple(d -> sum(rho .* grad_pot[d]), D)
    Vec{D,T}(acc)
end

end
