module ScalarSelf4d

using HCubature
using LinearAlgebra
using QuadGK
using StaticArrays
using TensorOperations



################################################################################

# Miscallaneous utilities

# Linear interpolation
export linear
function linear(x0::T, y0::U, x1::T, y1::U, x::T)::U where
        {T<:Number, U<:Number}
    U(x - x1) / U(x0 - x1) * y0 + U(x - x0) / U(x1 - x0) * y1
end



################################################################################

# Efficient small vectors

export Vec
struct Vec{D, T} <: Number      # DenseArray{T, 1}
    elts::NTuple{D, T}
end

# Vec{D}(x::T) where {D,T} = Vec{D,T}(ntuple(d->x, D))
# Vec{D,T}(x::T) where {D,T} = Vec{D,T}(ntuple(d->x, D))

vec1(::Val{D}, x::T) where {D, T} = Vec{D,T}(ntuple(d -> x, D))

function Base.size(x::Vec{D,T}, d)::Int where {D, T}
    D
end
function Base.size(x::Vec{D,T})::Tuple{Int} where {D, T}
    (D,)
end
function Base.length(x::Vec{D,T})::Int where {D, T}
    D
end

function Base.getindex(x::Vec{D,T}, d::Integer)::T where {D, T}
    getindex(x.elts, d)
end

function Base.zero(::Type{Vec{D,T}})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(ntuple(d -> T(0), D))
end

function Base.:+(x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(.+(x.elts))
end
function Base.:-(x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(.-(x.elts))
end

function Base.:+(x::Vec{D,T}, y::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(x.elts .+ y.elts)
end
function Base.:-(x::Vec{D,T}, y::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(x.elts .- y.elts)
end

function Base.:+(x::Vec{D,T}, y::T)::Vec{D,T} where {D, T<:Number}
    x + Vec(ntuple(d -> y, D))
end
function Base.:-(x::Vec{D,T}, y::T)::Vec{D,T} where {D, T<:Number}
    x - Vec(ntuple(d -> y, D))
end

function Base.:*(a::T, x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(a .* x.elts)
end
function Base.:*(x::Vec{D,T}, a::T)::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(x.elts .* a)
end
function Base.:\(x::Vec{D,T}, a::T)::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(a .\ x.elts)
end
function Base.:/(x::Vec{D,T}, a::T)::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(x.elts ./ a)
end

function LinearAlgebra.norm(x::Vec{D,T}, p::Real=2) where {D, T<:Number}
    norm(x.elts, p)
end

function Base.:(==)(x::Vec{D,T}, y::Vec{D,T})::Vec{D,Bool} where {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> x.elts[d] == y.elts[d], D))
end
function Base.:!=(x::Vec{D,T}, y::Vec{D,T})::Vec{D,Bool} where {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> x.elts[d] != y.elts[d], D))
end
function Base.:<(x::Vec{D,T}, y::Vec{D,T})::Vec{D,Bool} where {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> x.elts[d] < y.elts[d], D))
end
function Base.:<=(x::Vec{D,T}, y::Vec{D,T})::Vec{D,Bool} where {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> x.elts[d] <= y.elts[d], D))
end
function Base.:>(x::Vec{D,T}, y::Vec{D,T})::Vec{D,Bool} where {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> x.elts[d] > y.elts[d], D))
end
function Base.:>=(x::Vec{D,T}, y::Vec{D,T})::Vec{D,Bool} where {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> x.elts[d] >= y.elts[d], D))
end

function Base.:(==)(x::Vec{D,T}, y::T)::Vec{D,Bool} where {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> x.elts[d] == y, D))
end
function Base.:!=(x::Vec{D,T}, y::T)::Vec{D,Bool} where {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> x.elts[d] != y, D))
end
function Base.:<(x::Vec{D,T}, y::T)::Vec{D,Bool} where {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> x.elts[d] < y, D))
end
function Base.:<=(x::Vec{D,T}, y::T)::Vec{D,Bool} where {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> x.elts[d] <= y, D))
end
function Base.:>(x::Vec{D,T}, y::T)::Vec{D,Bool} where {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> x.elts[d] > y, D))
end
function Base.:>=(x::Vec{D,T}, y::T)::Vec{D,Bool} where {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> x.elts[d] >= y, D))
end

function Base.:~(x::Vec{D,Bool})::Vec{D,Bool} where {D}
    Vec{D,Bool}(.~(x.elts))
end
function Base.:&(x::Vec{D,Bool}, y::Vec{D,Bool})::Vec{D,Bool} where {D}
    Vec{D,Bool}(x.elts .& y.elts)
end
function Base.:|(x::Vec{D,Bool}, y::Vec{D,Bool})::Vec{D,Bool} where {D}
    Vec{D,Bool}(x.elts .& y.elts)
end
function Base.:!(x::Vec{D,Bool})::Vec{D,Bool} where {D}
    ~x
end
# function Base.:&&(x::Vec{D,Bool}, y::Vec{D,Bool})::Vec{D,Bool} where {D}
#     x & y
# end
# function Base.:||(x::Vec{D,Bool}, y::Vec{D,Bool})::Vec{D,Bool} where {D}
#     x | y
# end

function Base.all(x::Vec{D,Bool})::Bool where {D}
    all(x.elts)
end
function Base.any(x::Vec{D,Bool})::Bool where {D}
    any(x.elts)
end
function Base.max(x::Vec{D,T})::T where {D, T<:Number}
    max(x.elts)
end
function Base.min(x::Vec{D,T})::T where {D, T<:Number}
    min(x.elts)
end
function Base.prod(x::Vec{D,T})::T where {D, T<:Number}
    prod(x.elts)
end
function Base.sum(x::Vec{D,T})::T where {D, T<:Number}
    sum(x.elts)
end



################################################################################

# Run-time parameters

export Par
struct Par{D,T<:Number}
    n::Vec{D,Int}
    xmin::Vec{D,T}
    xmax::Vec{D,T}
end

function (::Type{Par{D,T}})(n::Int) where {D, T}
    Par{D,T}(Vec{D,Int}(ntuple(d -> n, D)),
             Vec{D,T}(ntuple(d -> d<4 ? -1 : 0, D)),
             Vec{D,T}(ntuple(d -> 1, D)))
end

function (::Type{Par{T}})(n::Int) where {T}
    Par{4,T}(n)
end



################################################################################

# Basis functions and collocation points

# Coordinates of collocation points
export coords
function coords(par::Par{D,T}, d::Int)::Vector{T} where {D, T<:Number}
    T[linear(1, par.xmin[d], par.n[d], par.xmax[d], i) for i in 1:par.n[d]]
end
function coords(par::Par{D,T})::NTuple{D, Vector{T}} where {D, T<:Number}
    ntuple(d -> coords(par, d) for d in 1:D)
end

# Basis functions
export basis
function basis(par::Par{D,T}, d::Int, i::Int, x::T)::T where {D, T<:Number}
    @assert i>=0 && i<par.n[d]
    fm = linear(par.xmin[d], T(1 - i), par.xmax[d], T(1 + par.n[d] - 1 - i), x)
    fp = linear(par.xmin[d], T(1 + i), par.xmax[d], T(1 - par.n[d] + 1 + i), x)
    f0 = T(0)
    max(f0, min(fm, fp))
end
function basis(par::Par{D,T}, i::Vec{D,Int}, x::Vec{D,T})::T where
        {D, T<:Number}
    prod(basis(par, d, i[d], x[d]) for d in 1:D)
end

# Dot product between basis functions
function dot_basis(par::Par{D,T}, d::Int, i::Int, j::Int)::T where
        {D, T<:Number}
    n = par.n[d]
    @assert i>=0 && i<n
    @assert j>=0 && j<n
    dx = (par.xmax[d] - par.xmin[d]) / (n - 1)
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



################################################################################

# Discretized functions

export Fun
struct Fun{D,T,U}
    par::Par{D,T}
    coeffs::Array{U,D}
end

# Evaluate a function
function (fun::Fun{D,T,U})(x::Vec{D,T})::U where {D,T,U}
    f = U(0)
    for ic in CartesianIndices(size(fun.coeffs))
        i = Vec(ic.I) - 1
        f += fun.coeffs[ic] * basis(fun.par, i, x)
    end
    f
end

# Create a discretized function by projecting onto the basis functions
export approximate
function approximate(fun, ::Type{U}, par::Par{D,T})::Fun{D,U} where
        {D, T<:Number, U<:Number}
    fs = Array{U,D}(undef, par.n.elts)
    for ic in CartesianIndices(size(fs))
        i = Vec(ic.I) - 1
        f = U(0)
        # Integrate piecewise
        for jc in CartesianIndices(ntuple(d -> 0:1, D))
            ij = i + Vec(jc.I) - 1
            if all(ij >= 0) && all(ij < par.n - 1)
                x0 = Vec{D,T}(ntuple(d -> linear(
                    0, par.xmin[d], par.n[d]-1, par.xmax[d], ij[d]), D))
                x1 = Vec{D,T}(ntuple(d -> linear(
                    0, par.xmin[d], par.n[d]-1, par.xmax[d], ij[d]+1), D))
                function kernel(x0::SVector{D,T})::U
                    x = Vec(Tuple(x0))
                    U(fun(x)) * U(basis(par, i, x))
                end
                r, e = hcubature(kernel,
                                 SVector{D,T}(x0.elts), SVector{D,T}(x1.elts))
                f += r
            end
        end
        fs[ic] = f
    end

    Winvs = ntuple(D) do d
        # We know the overlaps of the support of the basis functions
        dv = [dot_basis(par, d, i, i) for i in 0:par.n[d]-1]
        ev = [dot_basis(par, d, i, i+1) for i in 0:par.n[d]-2]
        W = SymTridiagonal(dv, ev)
        inv(W)
    end

    if D == 1
        Winv1 = Winvs[1]
        @tensor begin
            cs[i1] := Winv1[i1,j1] * fs[j1]
        end
    elseif D == 2
        Winv1 = Winvs[1]
        Winv2 = Winvs[2]
        @tensor begin
            cs[i1,i2] := Winv1[i1,j1] * Winv2[i2,j2] * fs[j1,j2]
        end
    elseif D == 3
        Winv1 = Winvs[1]
        Winv2 = Winvs[2]
        Winv3 = Winvs[3]
        @tensor begin
            cs[i1,i2,i3] :=
                Winv1[i1,j1] * Winv2[i2,j2] * Winv3[i3,j3] * fs[j1,j2,j3]
        end
    elseif D == 4
        Winv1 = Winvs[1]
        Winv2 = Winvs[2]
        Winv3 = Winvs[3]
        Winv4 = Winvs[4]
        @tensor begin
            cs[i1,i2,i3,i4] :=
                (Winv1[i1,j1] * Winv2[i2,j2] * Winv3[i3,j3] * Winv4[i4,j4] *
                fs[j1,j2,j3,j4])
        end
    else
        @assert false
    end
    return Fun{D,T,U}(par, cs)
end

# Approximate a delta function
export approximate_delta
function approximate_delta(::Type{U}, par::Par{D,T}, x::T)::Vector{U} where
        {D, T<:Number, U<:Number}
    fs = Array{U,D}(undef, par.n.elts)
    for ic in CartesianIndices(size(fs))
        i = Vec(ic.I) - 1
        fs[ic] = U(basis(par, i, x))
    end

    Winvs = ntuple(D) do d
        # We know the overlaps of the support of the basis functions
        dv = [dot_basis(par, d, i, i) for i in 0:par.n[d]-1]
        ev = [dot_basis(par, d, i, i+1) for i in 0:par.n[d]-2]
        W = SymTridiagonal(dv, ev)
        inv(W)
    end

    if D == 1
        Winv1 = Winvs[1]
        @tensor begin
            cs[i1] := Winv1[i1,j1] * fs[j1]
        end
    elseif D == 2
        Winv1 = Winvs[1]
        Winv2 = Winvs[2]
        @tensor begin
            cs[i1,i2] := Winv1[i1,j1] * Winv2[i2,j2] * fs[j1,j2]
        end
    elseif D == 3
        Winv1 = Winvs[1]
        Winv2 = Winvs[2]
        Winv3 = Winvs[3]
        @tensor begin
            cs[i1,i2,i3] :=
                Winv1[i1,j1] * Winv2[i2,j2] * Winv3[i3,j3] * fs[j1,j2,j3]
        end
    elseif D == 4
        Winv1 = Winvs[1]
        Winv2 = Winvs[2]
        Winv3 = Winvs[3]
        Winv4 = Winvs[4]
        @tensor begin
            cs[i1,i2,i3,i4] :=
                (Winv1[i1,j1] * Winv2[i2,j2] * Winv3[i3,j3] * Winv4[i4,j4] *
                fs[j1,j2,j3,j4])
        end
    else
        @assert false
    end
    return Fun{D,T,U}(par, cs)
end



################################################################################

# Derivatives

# Derivative of basis functions   dϕ^i/dϕ^j
function deriv_basis(par::Par{D,T}, d::Int, i::Int, j::Int)::T where
        {D, T<:Number}
    dx = (fun.par.xmax[d] - fun.par.xmin[d]) / (fun.par.n[d] - 1)
    if i == 0
        if j == i
            return -1/dx
        elseif j == i+1
            return 1/dx
        end
    elseif i < par.n[d]
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
function deriv(par::Par{D,T}, d::Int)::Tridiagonal{T} where {D, T<:Number}
    # We know the overlaps of the support of the basis functions
    n = par.n[d] - 1
    dlv = [deriv_basis(par, d, i, i-1) for i in 1:n]
    dv = [deriv_basis(par, d, i, i) for i in 0:n]
    duv = [deriv_basis(par, d, i, i+1) for i in 0:n-1]
    Tridiagonal(dlv, dv, duv)
end



function deriv(fun::Fun{D,T,U}, dir::Int)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert 1 <= dir <= D
    dx = (fun.par.xmax[dir] - fun.par.xmin[dir]) / (fun.par.n[dir] - 1)
    cs = fun.coeffs
    dcs = similar(cs)
    n = size(dcs, dir)

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

    Fun{D,T,U}(fun.par, dcs)
end

export deriv2
function deriv2(fun::Fun{D,T,U}, dir::Int)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert 1 <= dir <= D
    dx2 = ((fun.par.xmax[dir] - fun.par.xmin[dir]) / (fun.par.n[dir] - 1)) ^ 2
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

    Fun{D,T,U}(fun.par, dcs)
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

end
