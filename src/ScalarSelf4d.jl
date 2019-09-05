module ScalarSelf4d

using HCubature
using LinearAlgebra
using QuadGK
using SparseArrays
using StaticArrays
using TensorOperations



################################################################################

# Miscallaneous utilities

# Inverse of signbit
export bitsign
function bitsign(b::Bool)::Int
    b ? -1 : +1
end

# Linear interpolation
export linear
function linear(x0::T, y0::U, x1::T, y1::U, x::T)::U where
        {T<:Number, U<:Number}
    U(x - x1) / U(x0 - x1) * y0 + U(x - x0) / U(x1 - x0) * y1
end



################################################################################

# Efficient small vectors

export Vec
struct Vec{D, T} <: DenseArray{T, 1}
    elts::NTuple{D, T}
end

# Vec{D}(x::T) where {D,T} = Vec{D,T}(ntuple(d->x, D))
# Vec{D,T}(x::T) where {D,T} = Vec{D,T}(ntuple(d->x, D))

# vec1(::Val{D}, x::T) where {D, T} = Vec{D,T}(ntuple(d -> x, D))



# Vec interacts with scalars

function Base.promote_rule(::Type{Vec{D,T}}, ::Type{T}) where {D, T<:Number}
    Vec{D,T}
end
function Base.convert(::Type{Vec{D,T}}, x::T) where {D, T<:Number}
    Vec{D,T}(ntuple(d -> x, D))
end



# Vec is a collection

function Base.eltype(x::Vec{D,T})::Type where {D, T}
    T
end
function Base.length(x::Vec{D,T})::Int where {D, T}
    D
end
function Base.ndims(x::Vec{D,T})::Int where {D, T}
    1
end
function Base.size(x::Vec{D,T})::Tuple{Int} where {D, T}
    (D,)
end
function Base.size(x::Vec{D,T}, d)::Int where {D, T}
    D
end

function Base.getindex(x::Vec{D,T}, d::Integer)::T where {D, T}
    getindex(x.elts, d)
end



# Vec are a vector space

function Base.zero(::Type{Vec{D,T}})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(ntuple(d -> T(0), D))
end
function Base.one(::Type{Vec{D,T}})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(ntuple(d -> T(1), D))
end

function Base.:+(x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(.+(x.elts))
end
function Base.:-(x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(.-(x.elts))
end
function Base.inv(x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(inv.(x.elts))
end

function Base.:+(x::Vec{D,T}, y::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(x.elts .+ y.elts)
end
function Base.:-(x::Vec{D,T}, y::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(x.elts .- y.elts)
end

function Base.:*(a::Number, x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(T(a) .* x.elts)
end
function Base.:*(x::Vec{D,T}, a::Number)::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(x.elts .* T(a))
end
function Base.:\(a::Number, x::Vec{D,T})::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(T(a) .\ x.elts)
end
function Base.:/(x::Vec{D,T}, a::Number)::Vec{D,T} where {D, T<:Number}
    Vec{D,T}(x.elts ./ T(a))
end

const ArithOp = Union{typeof(+), typeof(-), typeof(*), typeof(/), typeof(\)}
function Base.broadcasted(op::ArithOp,
                          x::Vec{D,T}, y::Vec{D,T})::Vec{D,T} where
        {D, T<:Number}
    Vec{D,T}(ntuple(d -> op(x.elts[d], y.elts[d]), D))
end
function Base.broadcasted(op::ArithOp, x::Vec{D,T}, a::Number)::Vec{D,T} where
        {D, T<:Number}
    Vec{D,T}(ntuple(d -> op(x.elts[d], T(a)), D))
end
function Base.broadcasted(op::ArithOp, a::Number, x::Vec{D,T})::Vec{D,T} where
        {D, T<:Number}
    Vec{D,T}(ntuple(d -> op(T(a), x.elts[d]), D))
end

function LinearAlgebra.norm(x::Vec{D,T}, p::Real=2) where {D, T<:Number}
    norm(x.elts, p)
end

const CmpOp = Union{typeof(==), typeof(!=),
                    typeof(<), typeof(<=), typeof(>), typeof(>=)}
function Base.broadcasted(op::CmpOp,
                          x::Vec{D,T}, y::Vec{D,T})::Vec{D,Bool} where
        {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> op(x.elts[d], y.elts[d]), D))
end
function Base.broadcasted(op::CmpOp, x::Vec{D,T}, a::Number)::Vec{D,Bool} where
        {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> op(x.elts[d], T(a)), D))
end
function Base.broadcasted(op::CmpOp, a::Number, x::Vec{D,T})::Vec{D,Bool} where
        {D, T<:Number}
    Vec{D,Bool}(ntuple(d -> op(T(a), x.elts[d]), D))
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

# Integration weighs for basis functions (assuming a diagonal weight matrix)
export weight
function weight(par::Par{D,T}, i::Int)::T where {D, T<:Number}
    n = par.n[d]
    @assert i>=0 && i<n
    dx = (par.xmax[d] - par.xmin[d]) / (n - 1)
    if i == 0
        return dx/2
    elseif i < n-1
        return dx
    else
        return dx/2
    end
end
function weight(par::Par{D,T}, i::Vec{D,Int})::T where {D, T<:Number}
    prod(weight(par, i[d]) for d in 1:D)
end



################################################################################

# Discretized functions

export Fun
struct Fun{D,T,U} <: DenseArray{T, D}
    par::Par{D,T}
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
    prod(f.par.n)
end
function Base.ndims(f::Fun{D,T,U})::Int where {D, T, U}
    D
end
function Base.size(f::Fun{D,T,U})::NTuple{D,Int} where {D, T, U}
    f.par.n.elts
end
function Base.size(f::Fun{D,T,U}, d)::Int where {D, T, U}
    prod(f.par.n)
end

function Base.getindex(f::Fun{D,T,U}, i::Vec{D,Int})::U where {D, T, U}
    getindex(f.coeffs, i.elts)
end
function Base.getindex(f::Fun{D,T,U}, is...)::U where {D, T, U}
    getindex(f.coeffs, is...)
end



# Fun is a vector space

function Base.zero(::Type{Fun{D,T,U}}, par::Par{D,T})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    Fun{D,T,U}(par, zeros(U, par.n.elts))
end

function Base.:+(f::Fun{D,T,U})::Fun{D,T,U} where {D, T<:Number, U<:Number}
    Fun{D,T,U}(f.par, +f.elts)
end
function Base.:-(f::Fun{D,T,U})::Fun{D,T,U} where {D, T<:Number, U<:Number}
    Fun{D,T,U}(f.par, -f.elts)
end

function Base.:+(f::Fun{D,T,U}, g::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert f.par == g.par
    Fun{D,T,U}(f.par, f.coeffs + g.coeffs)
end
function Base.:-(f::Fun{D,T,U}, g::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert f.par == g.par
    Fun{D,T,U}(f.par, f.coeffs - g.coeffs)
end

function Base.:*(a::Number, f::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    Fun{D,T,U}(f.par, U(a) * f.coeffs)
end
function Base.:*(f::Fun{D,T,U}, a::Number)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    Fun{D,T,U}(f.par, f.coeffs * U(a))
end
function Base.:\(a::Number, f::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    Fun{D,T,U}(f.par, U(a) \ f.coeffs)
end
function Base.:/(f::Fun{D,T,U}, a::Number)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    Fun{D,T,U}(f.par, f.coeffs / U(a))
end

# function Base.:.+(f::Fun{D,T,U}, c::U)::Fun{D,T,U} where
#         {D, T<:Number, U<:Number}
#     Fun{D,T,U}(f.par, f.coeffs .+ c)
# end
# function Base.:.-(f::Fun{D,T,U}, c::U)::Fun{D,T,U} where
#         {D, T<:Number, U<:Number}
#     Fun{D,T,U}(f.par, f.coeffs .- c)
# end

# function Base.:*(f::Fun{D,T,U}, g::Fun{D,T,U})::U where
#         {D, T<:Number, U<:Number}
# TODO: bra and ket
# end

# function LinearAlgebra.norm(f::Fun{D,T,U}, p::Real=2) where
#         {D, T<:Number, U<:Number}
#     TODO
# end



# Fun are a category

# TODO: composition

function fconst(par::Par{D,T}, f::U)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    cs = fill(f, par.n.elts)
    Fun{D,T,U}(par, cs)
end

function fidentity(::Type{U}, par::Par{1,T})::Fun{1,T,U} where
        {T<:Number, U<:Number}
    cs = LinRange(U(par.xmin[1]), U(par.xmax[1]), par.n[1])
    Fun{1,T,U}(par, cs)
end



# Evaluate a function
function (fun::Fun{D,T,U})(x::Vec{D,T})::U where {D, T<:Number, U<:Number}
    f = U(0)
    for ic in CartesianIndices(size(fun.coeffs))
        i = Vec(ic.I) .- 1
        f += fun.coeffs[ic] * basis(fun.par, i, x)
    end
    f
end

# Create a discretized function by projecting onto the basis functions
export approximate
function approximate(fun, ::Type{U}, par::Par{D,T};
                     # mask::Union{Nothing, Op{D,T,U}}=nothing,
                     mask=nothing,
                     rtol::Real=0)::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    S = eltype(U)
    if rtol == 0
        rtol = sqrt(S(max(eps(T), eps(S))))
    else
        rtol = S(rtol)
    end

    str = Vec{D,Int}(ntuple(dir ->
                            dir==1 ? 1 : prod(par.n[d] for d in 1:dir-1), D))
    len = prod(par.n)
    idx(i::Vec{D,Int}) = 1 + sum(i[d] * str[d] for d in 1:D)
    function active(i::Vec{D,Int})::Bool
        mask === nothing && return true
        mask.mat[idx(i), idx(i)] != 0
    end

    fs = Array{U,D}(undef, par.n.elts)
    t0 = Libc.time()
    for ic in CartesianIndices(size(fs))
        i = Vec(ic.I) .- 1
        t1 = Libc.time()
        if t1 >= t0 + 10
            pct = round(idx(i) / len * 100, digits=1)
            @info "approximate $i ($pct%)"
            t0 = t1
        end
        f = U(0)
        if active(i)
            # Integrate piecewise
            for jc in CartesianIndices(ntuple(d -> 0:1, D))
                ij = i + Vec(jc.I) .- 1
                if all(ij .>= 0) && all(ij .< par.n .- 1)
                    x0 = Vec{D,T}(ntuple(d -> linear(
                        0, par.xmin[d], par.n[d]-1, par.xmax[d], ij[d]), D))
                    x1 = Vec{D,T}(ntuple(d -> linear(
                        0, par.xmin[d], par.n[d]-1, par.xmax[d], ij[d]+1), D))
                    function kernel(x0)::U
                        x = Vec{D,T}(Tuple(x0))
                        U(fun(x)) * U(basis(par, i, x))
                    end
                    r, e = hcubature(kernel,
                                     SVector{D,S}(SVector{D,T}(x0.elts)),
                                     SVector{D,S}(SVector{D,T}(x1.elts)),
                                     rtol=rtol)
                    f += r
                end
            end
        end
        fs[ic] = f
    end

    @show "ap.1"
    Winvs = ntuple(D) do d
        # We know the overlaps of the support of the basis functions
        dv = [dot_basis(par, d, i, i) for i in 0:par.n[d]-1]
        ev = [dot_basis(par, d, i, i+1) for i in 0:par.n[d]-2]
        W = SymTridiagonal(dv, ev)
        inv(W)
    end
    @show "ap.2"

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
        @show "ap.3"
        @tensor begin
            cs[i1,i2,i3,i4] :=
                (Winv1[i1,j1] * Winv2[i2,j2] * Winv3[i3,j3] * Winv4[i4,j4] *
                fs[j1,j2,j3,j4])
        end
        @show "ap.4"
    else
        @assert false
    end
    return Fun{D,T,U}(par, cs)
end

# Approximate a delta function
export approximate_delta
function approximate_delta(::Type{U}, par::Par{D,T},
                           x::Vec{D,T})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    fs = Array{U,D}(undef, par.n.elts)
    for ic in CartesianIndices(size(fs))
        i = Vec(ic.I) .- 1
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
    Fun{D,T,U}(par, cs)
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



################################################################################

# Operators

export Op
struct Op{D,T,U} <: AbstractArray{U, 2}
    par::Par{D,T}
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

function Base.zero(::Type{Op{D,T,U}}, par::Par{D,T})::Op{D,T,U} where
        {D, T, U<:Number}
    len = prod(par.n)
    mat = spzeros(U, len, len)
    Op{D,T,U}(par, mat)
end
function Base.one(::Type{Op{D,T,U}}, par::Par{D,T})::Op{D,T,U} where
        {D, T, U<:Number}
    n = par.n

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
    for ic in CartesianIndices(par.n.elts)
        i = Vec(ic.I) .- 1
        ins!(i, i, U(1))
    end
    mat = sparse(I, J, V, len, len)
    Op{D,T,U}(par, mat)
end

function Base.:+(A::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U<:Number}
    Op{D,T,U}(A.par, +A.mat)
end
function Base.:-(A::Op{D,T,U})::Op{D,T,U} where {D, T<:Number, U<:Number}
    Op{D,T,U}(A.par, -A.mat)
end

function Base.:+(A::Op{D,T,U}, B::Op{D,T,U})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert A.par == B.par
    Op{D,T,U}(A.par, A.mat + B.mat)
end
function Base.:-(A::Op{D,T,U}, B::Op{D,T,U})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert A.par == B.par
    Op{D,T,U}(A.par, A.mat - B.mat)
end

function Base.:*(a::Number, A::Op{D,T,U})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    Op{D,T,U}(A.par, U(a) * A.mat)
end
function Base.:*(A::Op{D,T,U}, a::Number)::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    Op{D,T,U}(A.par, A.mat * U(a))
end
function Base.:\(a::Number, A::Op{D,T,U})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    Op{D,T,U}(A.par, U(a) \ A.mat)
end
function Base.:/(A::Op{D,T,U}, a::Number)::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    Op{D,T,U}(A.par, A.mat / U(a))
end



function Base.:*(A::Op{D,T,U}, B::Op{D,T,U})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    @assert A.par == B.par
    Op{D,T,U}(A.par, A.mat * B.mat)
end



function Base.:*(op::Op{D,T,U}, rhs::Fun{D,T,U})::Fun{D,T,U} where
        {D, T, U<:Number}
    par = rhs.par
    @assert op.par == par

    res = reshape(op.mat * reshape(rhs.coeffs, :), par.n.elts)
    Fun{D,T,U}(par, res)
end

# function Base.:*(lhs::Fun{D,T,U}, op::Op{D,T,U})::Fun{D,T,U} where
#         {D, T, U<:Number}
#     par = rhs.par
#     @assert op.par == par
# 
#     TODO: bra and ket
#     res = reshape(reshape(lhs.coeffs, :) * op.mat, par.n.elts)
#     Fun{D,T,U}(par, res)
# end

function Base.:\(op::Op{D,T,U}, rhs::Fun{D,T,U})::Fun{D,T,U} where
        {D, T, U<:Number}
    par = rhs.par
    @assert op.par == par

    M = op.mat
    if T <: Union{Float32, Float64}
        # do nothing
    else
        @info "Converting sparse to full matrix..."
        M = Matrix(M)
    end
    sol = reshape(M \ reshape(rhs.coeffs, :), par.n.elts)
    Fun{D,T,U}(par, sol)
end

# function Base.:\(op::Op{D,T,U}, rhs::Fun{D,T,U})::Fun{D,T,U} where
#         {D, T, U<:Number}
#     par = rhs.par
#     @assert op.par == par
# 
#     len = prod(par.n)
# 
#     bnd = boundary(U, par)
#     proj = I(len) - bnd.mat
#     sol = reshape(op.mat \ (proj * reshape(rhs.coeffs, :)), par.n.elts)
#     Fun{D,T,U}(par, sol)
# end



export boundary
function boundary(::Type{U}, par::Par{D,T})::Op{D,T,U} where {D, T, U<:Number}
    n = par.n

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
    for ic in CartesianIndices(par.n.elts)
        i = Vec(ic.I) .- 1
        if any(i .== 0) || any(i .== n .- 1)
            ins!(i, i, U(1))
        end
    end
    mat = sparse(I, J, V, len, len)
    Op{D,T,U}(par, mat)
end

export dirichlet
# TODO: Is this correct?
const dirichlet = boundary

export laplace
function laplace(::Type{U}, par::Par{D,T})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    n = par.n
    dx2 = Vec(ntuple(d -> ((par.xmax[d] - par.xmin[d]) / (n[d] - 1)) ^ 2, D))

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
    for ic in CartesianIndices(par.n.elts)
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
    Op{D,T,U}(par, mat)
end



export boundaryIV
function boundaryIV(::Type{U}, par::Par{D,T})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    n = par.n

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
    for ic in CartesianIndices(par.n.elts)
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
    Op{D,T,U}(par, mat)
end

export dirichletIV
# TODO: Is this correct?
const dirichletIV = boundaryIV

export dAlembert
function dAlembert(::Type{U}, par::Par{D,T})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    n = par.n
    dx2 = Vec(ntuple(d -> ((par.xmax[d] - par.xmin[d]) / (n[d] - 1)) ^ 2, D))

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
    for ic in CartesianIndices(par.n.elts)
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
    Op{D,T,U}(par, mat)
end



export mix_op_bc
function mix_op_bc(bnd::Op{D,T,U},
                   iop::Op{D,T,U}, bop::Op{D,T,U})::Op{D,T,U} where
        {D, T<:Number, U<:Number}
    par = bnd.par
    @assert iop.par == par
    @assert bop.par == par

    id = one(Op{D,T,U}, par)
    int = id - bnd
    int * iop + bnd * bop
end
function mix_op_bc(bnd::Op{D,T,U},
                   rhs::Fun{D,T,U}, bvals::Fun{D,T,U})::Fun{D,T,U} where
        {D, T<:Number, U<:Number}
    par = bnd.par
    @assert rhs.par == par
    @assert bvals.par == par

    id = one(Op{D,T,U}, par)
    int = id - bnd
    int * rhs + bnd * bvals
end

end
