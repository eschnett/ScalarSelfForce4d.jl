module ScalarSelf4d

using HCubature
using LinearAlgebra
using QuadGK
using TensorOperations
# using StaticArrays



# Linear interpolation 
export linear
function linear(x0::T, y0::U, x1::T, y1::U, x::T)::U where
        {T<:Number, U<:Number}
    U(x - x1) / U(x0 - x1) * y0 + U(x - x0) / U(x1 - x0) * y1
end



export Vec
struct Vec{D, T} <: DenseArray{T, 1}
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

function Base.getindex(x::Vec{D,T}, d)::T where {D, T}
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



export Par
struct Par{D,T<:Number}
    n::Int
    xmin::Vec{D,T}
    xmax::Vec{D,T}
end

function (::Type{Par{D,T}})(n::Int) where {D, T}
    Par{D,T}(n, Vec{D,T}(ntuple(d -> d<4 ? -1 : 0, D)),
                Vec{D,T}(ntuple(d -> 1, D)))
end

function (::Type{Par{T}})(n::Int) where {T}
    Par{4,T}(n)
end



export coords

function coords(par::Par{D,T}, d::Int)::Vector{T} where {D, T<:Number}
    T[linear(1, par.xmin[d], par.n, par.xmax[d], i) for i in 1:par.n]
end

function coords(par::Par{D,T})::NTuple{D, Vector{T}} where {D, T<:Number}
    ntuple(d -> coords(par, d) for d in 1:D)
end



export basis

function basis(par::Par{D,T}, d::Int, i::Int, x::T)::T where {D, T<:Number}
    @assert i>=0 && i<par.n
    fm = linear(par.xmin[d], T(1 - i), par.xmax[d], T(1 + par.n - 1 - i), x)
    fp = linear(par.xmin[d], T(1 + i), par.xmax[d], T(1 - (par.n - 1 - i)), x)
    f0 = T(0)
    max(f0, min(fm, fp))
end

function basis(par::Par{D,T}, i::Vec{D,Int}, x::Vec{D,T})::T where
        {D, T<:Number}
    prod(basis(par, d, i[d], x[d]) for d in 1:D)
end



export value

function value(par::Par{D,T}, d::Int, cs::Vector{U}, x::T)::U where
        {D, T<:Number, U<:Number}
    f = U(0)
    for i in 0:par.n-1
        f += cs[i+1] * basis(par, d, i, x)
    end
    f
end

function value(par::Par{D,T}, cs::Array{U,D}, x::Vec{D,T})::U where
        {D, T<:Number, U<:Number}
    nc = CartesianIndex(ntuple(d -> par.n, D))
    f = U(0)
    for ic in CartesianIndices(nc)
        i = Vec(ic.I) - vec1(Val(D), 1)
        f += cs[ic] * basis(par, i, x)
    end
    f
end



function dot_basis(par::Par{D,T}, d::Int, i::Int, j::Int)::T where
        {D, T<:Number}
    @assert i>=0 && i<par.n
    @assert j>=0 && j<par.n
    dx = (par.xmax[d] - par.xmin[d]) / (par.n - 1)
    if j == i-1
        return dx/6
    elseif j == i
        if i == 0 || i == par.n-1
            return dx/3
        else
            return 2*dx/3
        end
    elseif j == i+1
        return dx/6
    else
        return T(0)
    end
end



export project_basis

function project_basis(fun, ::Type{U}, par::Par{D,T}, d::Int)::Vector{U} where
        {D, T<:Number, U<:Number}
    dv = [dot_basis(par, d, i, i) for i in 0:par.n-1]
    ev = [dot_basis(par, d, i, i+1) for i in 0:par.n-2]
    W = SymTridiagonal(dv, ev)

    fs = Vector{U}(undef, par.n)
    for i in 0:par.n-1
        x0 = linear(0, par.xmin[d], par.n-1, par.xmax[d], i - 1)
        xc = linear(0, par.xmin[d], par.n-1, par.xmax[d], i)
        x1 = linear(0, par.xmin[d], par.n-1, par.xmax[d], i + 1)
        f = U(0)
        if i - 1 >= 0
            r, e = quadgk(x -> fun(x) * basis(par, d, i, x), x0, xc)
            f += r
        end
        if i + 1 < par.n
            r, e = quadgk(x -> fun(x) * basis(par, d, i, x), xc, x1)
            f += r
        end
        fs[i+1] = f
    end
    W \ fs
end

function project_basis(fun, ::Type{U}, par::Par{D,T})::Array{U,D} where
        {D, T<:Number, U<:Number}
    Winvs = ntuple(D) do d
        dv = [dot_basis(par, d, i, i) for i in 0:par.n-1]
        ev = [dot_basis(par, d, i, i+1) for i in 0:par.n-2]
        W = SymTridiagonal(dv, ev)
        inv(W)
    end

    nc = CartesianIndex(ntuple(d -> par.n, D))
    fs = Array{U,D}(undef, nc.I)
    for ic in CartesianIndices(nc)
        f = U(0)
        for jc in CartesianIndices(ntuple(d -> 0:1, D))
            i = Vec(ic.I) - vec1(Val(D), 1)
            ij = i + Vec(jc.I) - vec1(Val(D), 1)
            if all(ij >= vec1(Val(D), 0)) && all(ij < vec1(Val(D), par.n - 1))
                x0 = Vec{D,T}(ntuple(d -> linear(0, par.xmin[d],
                                           par.n-1, par.xmax[d], ij[d]), D))
                x1 = Vec{D,T}(ntuple(d -> linear(0, par.xmin[d],
                                           par.n-1, par.xmax[d], ij[d]+1), D))
                function g(x0)
                    x = Vec{D,T}(Tuple(x0))
                    U(fun(x)) * U(basis(par, i, x))
                end
                r, e = hcubature(g, x0, x1)
                f += r
            end
        end
        fs[ic] = f
    end

    if D == 1
        Winv1 = Winvs[1]
        @tensor begin
            cs[i1] := Winv1[i1,j1] * fs[j1]
        end
        return cs
    elseif D == 2
        Winv1 = Winvs[1]
        Winv2 = Winvs[2]
        @tensor begin
            cs[i1,i2] := Winv1[i1,j1] * Winv2[i2,j2] * fs[j1,j2]
        end
        return cs
    elseif D == 3
        Winv1 = Winvs[1]
        Winv2 = Winvs[2]
        Winv3 = Winvs[3]
        @tensor begin
            cs[i1,i2,i3] :=
                Winv1[i1,j1] * Winv2[i2,j2] * Winv2[i3,j3] * fs[j1,j2,j3]
        end
        return cs
    else
        @assert false
    end
end



export deriv, deriv2

function deriv(par::Par{D,T}, d::Int, f::AbstractArray{T,1},
        df::AbstractArray{T,1})::Nothing where {D, T<:Number}
    dx = (par.xmax[d] - par.xmin[d]) / (par.n - 1)
    df[1] = (f[2] - f[1]) / dx
    for i in 2:par.n-1
        df[i] = (f[i+1] - f[i-1]) / (2*dx)
    end
    df[par.n] = (f[par.n] - f[par.n - 1]) / dx
end
function deriv(par::Par{D,T}, f::AbstractArray{T,1}, d::Int) where
        {D, T<:Number}
    df = similar(f)
    deriv(par, d, f, df)
    df
end

function deriv2(par::Par{D,T}, d::Int, f::AbstractArray{T,1},
        df::AbstractArray{T,1})::Nothing where {D, T<:Number}
    dx = (par.xmax[d] - par.xmin[d]) / (par.n - 1)
    ddf[1] = (f[3] - 2*f[2] + f[1]) / dx^2
    for i in 2:par.n-1
        df[i] = (f[i+1] - 2*f[i] + f[i-1]) / dx^2
    end
    ddf[par.n] = (f[par.n] - 2*f[par.n-1] + f[par.n-2]) / dx^2
end
function deriv2(par::Par{D,T}, d::Int, f::AbstractArray{T,1}) where
        {D, T<:Number}
    ddf = similar(f)
    deriv2(par, d, f, ddf)
    ddf
end

end
