"""
Numerical quadrature
"""
module Quadrature

using ..Defs



struct Quad{T}
    points::Vector{T}
    weights::Vector{T}
end



function symmetrize(xs::Vector{T})::Vector{T} where {T}
    (xs + reverse(xs)) ./ 2
end

function antisymmetrize(xs::Vector{T})::Vector{T} where {T}
    (xs - reverse(xs)) ./ 2
end

function makequad(::Type{T}, n::Int)::Quad{T} where {T <: AbstractFloat}
    @assert n > 0
    half = T(1) / 2
    # Choose points according to cos pi (i/n)
    xs = T[cospi((n - i - half) / n) for i in 0:n - 1]
    xs = antisymmetrize(xs)
    @assert all(-1 .< xs .< 1)
    # Choose weights so that Chebyshev polynomials are exact
    chebvals = Array{T}(undef, n, n)
    chebints = Array{T}(undef, n)
    for i in 0:n - 1
        chebints[1 + i] = isodd(i) ? 0 : T(1) / (1 - i^2)
        for j in 0:n - 1
            # x = xs[1+j]
            # chebvals[1+i, 1+j] = cos(i * acos(x))
            # chebvals[1+i, 1+j] = cos(i * acos(xs[1+j]))
            # acos(x) = pi * (n - j - half) / n
            chebvals[1 + i, 1 + j] = cospi(i * (n - j - half) / n)
        end
    end
    ws = chebvals \ chebints
    ws = symmetrize(ws)
    Quad{T}(xs, ws)
end

function makequad(::Type{T}, n::Int)::Quad{T} where {T <: Rational}
    @assert n > 0
    # Choose points according to cos pi (i/n)
    I = typeof(T(0).num)
    cospi1(x) = rationalize(I, cospi(x); tol = 1 / n^4)::T
    xs = T[cospi1((n - i - 1 / 2) / n) for i in 0:n - 1]
    xs = antisymmetrize(xs)
    @assert all(-1 .< xs .< 1)
    # Choose weights so that polynomials x^i are exact
    polyvals = Array{T}(undef, n, n)
    polyints = Array{T}(undef, n)
    for i in 0:n - 1
        polyints[1 + i] = isodd(i) ? 0 : 1 // (i + 1)
        for j in 0:n - 1
            polyvals[1 + i, 1 + j] = xs[1 + j]^i
        end
    end
    ws = polyvals \ polyints
    @assert ws == reverse(ws)
    Quad{T}(xs, ws)
end



const quads = Dict{Tuple{Type,Int},Quad}()

function getquad(::Type{T}, n::Int)::Quad{T} where {T}
    q = get(quads, (T, n), nothing)
    if q === nothing
        q = makequad(T, n)
        @assert length(q.points) == n
        @assert length(q.weights) == n
        quads[(T, n)] = q
    end
    q::Quad{T}
end



export quad

@generated function quad(f, ::Type{U}, xmin::NTuple{D,T}, xmax::NTuple{D,T},
                         n::NTuple{D,Int})::U where {D,T,U}
    quote
        q = tuple($([:(getquad(T, n[$d])) for d in 1:D]...))
        s = zero(U)
        @inbounds for i in CartesianIndices(n)
            x = tuple($(
                [:(linear(T(-1), xmin[$d], T(1), xmax[$d],
                          q[$d].points[i[$d]])) for d in 1:D]...))
            w = *($([:(q[$d].weights[i[$d]]) for d in 1:D]...))
            s += U(w * U(f(x)))
        end
        w = *($([:(xmax[$d] - xmin[$d]) for d in 1:D]...))
        U(w * s)
    end
end

end
