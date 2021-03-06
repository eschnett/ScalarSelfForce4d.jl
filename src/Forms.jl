"""
Differential forms
"""
module Forms

using Random
using Combinatorics
using LinearAlgebra
using SparseArrays

using ..Bases
using ..Defs
using ..Domains
using ..Funs
using ..Ops
using ..Vecs



export Form
struct Form{D,R,Dual,T,U}
    dom::Domain{D,T}
    comps::Dict{Vec{R,Int},Fun{D,T,U}}
    # infval::Union{Nothing, U}   # value at infinity for R == (!Dual ? R : 0)

    function Form(dict::Dict{Vec{R,Int},Fun{D,T,U}}) where {D,R,T,U}
        # infval::Union{Nothing, U}=nothing)
        @assert D isa Int && 0 <= D
        @assert R isa Int && 0 <= R <= D
        # @assert (infval isa U) == (R == (!Dual ? R : 0))

        i0 = Vec{R,Int}(ntuple(d->d, R))
        f0 = dict[i0]
        dom = makeunstaggered(f0.dom)

        count = 0
        for i in CartesianIndices(ntuple(d->D, R))
            !all(i[d + 1] > i[d] for d in 1:R - 1) && continue
            count += 1
            f = dict[Vec{R,Int}(i.I)]
            @assert f.dom.dual == dom.dual
            @assert all(f.dom.staggered[d] == (d in i.I) for d in 1:D)
            if !dom.dual
                @assert all(f.dom.n + f.dom.staggered .== dom.n)
            else
                @assert all(f.dom.n - f.dom.staggered .== dom.n)
            end
            @assert f.dom.metric == dom.metric
            @assert f.dom.xmin == dom.xmin
            @assert f.dom.xmax == dom.xmax
        end
        @assert length(dict) == count

        new{D,R,dom.dual,T,U}(dom, copy(dict))
        # infval)
    end
end

function Form(dict::Dict{NTuple{R,Int},Fun{D,T,U}}) where {D,R,T,U}
    # infval::Union{Nothing, U}=nothing)
    Form(Dict(Vec{R,Int}(k) => v for (k, v) in dict))
    # infval)
end



function Base.eltype(::Type{Form{D,R,Dual,T,U}})::Type where {D,R,Dual,T,U}
    U
end

function Base.getindex(form::Form{D,R,Dual,T,U},
                       i::Vec{R,Int})::Fun{D,T,U} where {D,R,Dual,T,U}
    form.comps[i]
end
function Base.getindex(form::Form{D,R,Dual,T,U},
                       i::NTuple{R,Int})::Fun{D,T,U} where {D,R,Dual,T,U}
    form.comps[Vec{R,Int}(i)]
end
# function Base.getindex(form::Form{D,R,Dual,T,U}, ::Val{Inf})::U where
#         {D,R,Dual,T,U}
#     @assert R == (!Dual ? R : 0)
#     form.infval::U
# end



export staggered2idx
function staggered2idx(staggered::Vec{D,Bool})::Vector{Int} where {D}
    idx = map(ib->ib[1], filter(ib->ib[2],
                                collect(enumerate(staggered.elts))))
    @assert idx2staggered(Val(D), idx) == staggered
    idx
end
export idx2staggered
function idx2staggered(::Val{D}, idx::AbstractVector{Int})::Vec{D,Bool} where
        {D}
    staggered = falses(D)
    for i in idx
        @assert !staggered[i]
        staggered[i] = true
    end
    Vec{D,Bool}(Tuple(staggered))
end



function Random.rand(rng::AbstractRNG, S, ::Type{Form{D,R,Dual,T,U}}, dom::Domain{D,T}) where {D, R, Dual, T, U}
    icomps = Dict{Vec{R,Int}, Fun{D,T,U}}()
    for staggeredc in CartesianIndices(ntuple(d->0:1, D))
        staggered = Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
        if count(staggered) == R
            idx = Vec{R,Int}(Tuple(staggered2idx(staggered)))
            fdom = makestaggered(makedual(dom, Dual), staggered)
            fvals = rand(rng, S, fdom.n.elts)
            fun = Fun{D,T,U}(fdom, fvals)
            icomps[idx] = fun
        end
    end
    Form(icomps)
end

function Random.rand(rng::AbstractRNG, ::Type{Form{D,R,Dual,T,U}}, dom::Domain{D,T}) where {D, R, Dual, T, U}
    rand(rng, U, Form{D,R,Dual,T,U}, dom)
end

function Random.rand(S, ::Type{Form{D,R,Dual,T,U}}, dom::Domain{D,T}) where {D, R, Dual, T, U}
    rand(Random.GLOBAL_RNG, S, Form{D,R,Dual,T,U}, dom)
end

function Random.rand(::Type{Form{D,R,Dual,T,U}}, dom::Domain{D,T}) where {D, R, Dual, T, U}
    rand(Random.GLOBAL_RNG, Form{D,R,Dual,T,U}, dom)
end



function Base.zeros(::Type{Form{D,R,Dual,T,U}}, dom::Domain{D,T})::Form{D,R,Dual,T,U} where {D,R,Dual,T,U}
    dom = makedual(dom, Dual)
    comps = Dict{Vec{R,Int},Fun{D,T,U}}()
    for staggeredc in CartesianIndices(ntuple(d->0:1, D))
        staggered = Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
        if count(staggered) == R
            idx = Vec{R,Int}(Tuple(staggered2idx(staggered)))
            fdom = makestaggered(dom, staggered)
            fvals = zeros(U, fdom.n.elts)
            fun = Fun{D,T,U}(fdom, fvals)
            comps[idx] = fun
        end
    end
    Form(comps)
end

function Base.:+(f::Form{D,R,Dual,T,U})::Form{D,R,Dual,T,U} where {D,R,Dual,T,U}
    Form(Dict(i => +fi for (i, fi) in f.comps))
    # f.infval isa Nothing ? nothing : +f.infval)
end
function Base.:-(f::Form{D,R,Dual,T,U})::Form{D,R,Dual,T,U} where {D,R,Dual,T,U}
    Form(Dict(i => -fi for (i, fi) in f.comps))
    # f.infval isa Nothing ? nothing : -f.infval)
end

function Base.:+(f::Form{D,R,Dual,T,U},
                 g::Form{D,R,Dual,T,U})::Form{D,R,Dual,T,U} where {D,R,Dual,T,U}
    # @assert f.dom == g.dom
    @assert keys(f.comps) == keys(g.comps)
    Form(Dict(i => fi + g.comps[i] for (i, fi) in f.comps))
    # f.infval isa Nothing ? nothing : f.infval + g.infval)
end
function Base.:-(f::Form{D,R,Dual,T,U},
                 g::Form{D,R,Dual,T,U})::Form{D,R,Dual,T,U} where {D,R,Dual,T,U}
    # @assert f.dom == g.dom
    @assert keys(f.comps) == keys(g.comps)
    Form(Dict(i => fi - g.comps[i] for (i, fi) in f.comps))
    # f.infval isa Nothing ? nothing : f.infval - g.infval)
end

function Base.:*(a::U, f::Form{D,R,Dual,T,U})::Form{D,R,Dual,T,U} where {D,R,Dual,T,U}
    Form(Dict(i => a * fi for (i, fi) in f.comps))
    # f.infval isa Nothing ? nothing : a * f.infval)
end
function Base.:*(f::Form{D,R,Dual,T,U}, a::U)::Form{D,R,Dual,T,U} where {D,R,Dual,T,U}
    Form(Dict(i => fi * a for (i, fi) in f.comps))
    # f.infval isa Nothing ? nothing : f.infval * a)
end
function Base.:\(a::U, f::Form{D,R,Dual,T,U})::Form{D,R,Dual,T,U} where {D,R,Dual,T,U}
    Form(Dict(i => a \ fi for (i, fi) in f.comps))
    # f.infval isa Nothing ? nothing : a \ f.infval)
end
function Base.:/(f::Form{D,R,Dual,T,U}, a::U)::Form{D,R,Dual,T,U} where {D,R,Dual,T,U}
    Form(Dict(i => fi / a for (i, fi) in f.comps))
    # f.infval isa Nothing ? nothing : f.infval / a)
end

function Base.iszero(f::Form{D,R,Dual,T,U}, )::Bool where {D,R,Dual,T,U}
    for (i, fi) in f.comps
        iszero(fi) || return false
    end
    return true
end
function Base.:(==)(f::Form{D,R,Dual,T,U},
                    g::Form{D,R,Dual,T,U})::Bool where {D,R,Dual,T,U}
    # @assert f.dom == g.dom
    # @assert keys(f.comps) == keys(g.comps)
    # # @assert (f.infval isa Nothing) == (g.invfal isa Nothing)
    # all(fi == g.comps[i] for (i,fi) in f.comps)
    # # && f.infval == g.infval
    iszero(f - g)
end



export FOp
struct FOp{D,RI,DualI,RJ,DualJ,T,U}
    comps::Dict{Vec{RI,Int},Dict{Vec{RJ,Int},Op{D,T,U}}}
end

function Base.eltype(::Type{FOp{D,RI,DualI,RJ,DualJ,T,U}})::Type where
        {D,RI,DualI,RJ,DualJ,T,U}
    U
end

function Base.zeros(::Type{FOp{D,RI,DualI,RJ,DualJ,T,U}})::FOp{D,RI,DualI,RJ,DualJ,T,U} where {D,RI,DualI,RJ,DualJ,T,U}
    FOp{D,RI,DualI,RJ,DualJ,T,U}(Dict{Vec{RI,Int},Dict{Vec{RJ,Int},Op{D,T,U}}}())
end

function map1(fun, A::FOp{D,RI,DualI,RJ,DualJ,T,U})::FOp{D,RI,DualI,RJ,DualJ,T,U} where {D,RI,DualI,RJ,DualJ,T,U}
    FOp{D,RI,DualI,RJ,DualJ,T,U}(Dict(i => Dict(j => fun(Aj) for (j, Aj) in Ai) for (i, Ai) in A.comps))
end
function map2(fun,
              A::FOp{D,RI,DualI,RJ,DualJ,T,U},
              B::FOp{D,RI,DualI,RJ,DualJ,T,U})::FOp{D,RI,DualI,RJ,DualJ,T,U} where {D,RI,DualI,RJ,DualJ,T,U}
    idict = Dict{Vec{RI,Int},Dict{Vec{RJ,Int},Op{D,T,U}}}()
    for X in (A, B)
        for (i, Xi) in X.comps
            for (j, Xij) in Xi
                jdict = get!(idict, i) do
                    Dict{Vec{RJ,Int},Op{D,T,U}}()
                end
                prev = get(jdict, j, missing)
                next = prev === missing ? Xij : fun(prev, Xij)
                jdict[j] = next
            end
        end
    end
    FOp{D,RI,DualI,RJ,DualJ,T,U}(idict)
end

function Base.:+(A::FOp{D,RI,DualI,RJ,DualJ,T,U})::FOp{D,RI,DualI,RJ,DualJ,T,U} where {D,RI,DualI,RJ,DualJ,T,U}
    map1(+, A)
end
function Base.:-(A::FOp{D,RI,DualI,RJ,DualJ,T,U})::FOp{D,RI,DualI,RJ,DualJ,T,U} where {D,RI,DualI,RJ,DualJ,T,U}
    map1(-, A)
end

function Base.:+(A::FOp{D,RI,DualI,RJ,DualJ,T,U},
                 B::FOp{D,RI,DualI,RJ,DualJ,T,U})::FOp{D,RI,DualI,RJ,DualJ,T,U} where {D,RI,DualI,RJ,DualJ,T,U}
    map2(+, A, B)
end
function Base.:-(A::FOp{D,RI,DualI,RJ,DualJ,T,U},
                 B::FOp{D,RI,DualI,RJ,DualJ,T,U})::FOp{D,RI,DualI,RJ,DualJ,T,U} where {D,RI,DualI,RJ,DualJ,T,U}
    map2(-, A, B)
end

function Base.:*(a::U, A::FOp{D,RI,DualI,RJ,DualJ,T,U})::FOp{D,RI,DualI,RJ,DualJ,T,U} where {D,RI,DualI,RJ,DualJ,T,U}
    map1(x->a * x, A)
end
function Base.:*(A::FOp{D,RI,DualI,RJ,DualJ,T,U}, a::U)::FOp{D,RI,DualI,RJ,DualJ,T,U} where {D,RI,DualI,RJ,DualJ,T,U}
    map1(x->x * a, A)
end
function Base.:\(a::U, A::FOp{D,RI,DualI,RJ,DualJ,T,U})::FOp{D,RI,DualI,RJ,DualJ,T,U} where {D,RI,DualI,RJ,DualJ,T,U}
    map1(x->a \ x, A)
end
function Base.:/(A::FOp{D,RI,DualI,RJ,DualJ,T,U}, a::U)::FOp{D,RI,DualI,RJ,DualJ,T,U} where {D,RI,DualI,RJ,DualJ,T,U}
    map1(x->x / a, A)
end

function Base.iszero(A::FOp{D,RI,DualI,RJ,DualJ,T,U})::Bool where {D,RI,DualI,RJ,DualJ,T,U}
    for (i, Ai) in A.comps
        for (j, Aij) in Ai
            iszero(Aij) || return false
        end
    end
    return true
end

function Base.:(==)(A::FOp{D,RI,DualI,RJ,DualJ,T,U},
                    B::FOp{D,RI,DualI,RJ,DualJ,T,U})::Bool where {D,RI,DualI,RJ,DualJ,T,U}
    iszero(A - B)
end



function Base.zero(::Type{FOp{D,RI,DualI,RJ,DualJ,T,U}})::FOp{D,RI,DualI,RJ,DualJ,T,U} where {D,RI,DualI,RJ,DualJ,T,U}
    zeros(FOp{D,RI,DualI,RJ,DualJ,T,U})
end
function Base.one(::Type{FOp{D,R,Dual,R,Dual,T,U}},
                  dom::Domain{D,T})::FOp{D,R,Dual,R,Dual,T,U} where
        {D,R,Dual,T,U}
    comps = Dict{Vec{R,Int},Dict{Vec{R,Int},Op{D,T,U}}}()
    for staggeredc in CartesianIndices(ntuple(d->0:1, D))
        staggered = Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
        if count(staggered) == R
            idx = Vec{R,Int}(Tuple(staggered2idx(staggered)))
            fdom = makestaggered(makedual(dom, Dual), staggered)
            op = one(Op{D,T,U}, fdom, fdom)
            comps[idx] = Dict(idx => op)
        end
    end
    FOp{D,R,Dual,R,Dual,T,U}(comps)
end

function Ops.shift(::Type{FOp{D,R,Dual,R,Dual,T,U}}, dom::Domain{D,T},
                   di::Vec{D,Int})::FOp{D,R,Dual,R,Dual,T,U} where
        {D,R,Dual,T,U}
    comps = Dict{Vec{R,Int},Dict{Vec{R,Int},Op{D,T,U}}}()
    for staggeredc in CartesianIndices(ntuple(d->0:1, D))
        staggered = Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
        if count(staggered) == R
            idx = Vec{R,Int}(Tuple(staggered2idx(staggered)))
            fdom = makestaggered(makedual(dom, Dual), staggered)
            op = shift(Op{D,T,U}, fdom, fdom, di)
            comps[idx] = Dict(idx => op)
        end
    end
    FOp{D,R,Dual,R,Dual,T,U}(comps)
end

function Base.:*(A::FOp{D,RI,DualI,RJ,DualJ,T,U},
                 f::Form{D,RJ,DualJ,T,U})::Form{D,RI,DualI,T,U} where
        {D,RI,DualI,RJ,DualJ,T,U}
    idict = Dict{Vec{RI,Int},Fun{D,T,U}}()
    for (i, Ai) in A.comps
        for (j, Aij) in Ai
            fj = get(f.comps, j, missing)
            if fj !== missing
                Af = Aij * fj
                prev = get(idict, i, missing)
                next = prev === missing ? Af : prev + Af
                idict[i] = next
            end
        end
    end
    Form(idict)
end
function Base.:*(A::FOp{D,RI,DualI,RK,DualK,T,U},
                 B::FOp{D,RK,DualK,RJ,DualJ,T,U})::FOp{D,RI,DualI,RJ,DualJ,T,U} where
        {D,RI,DualI,RJ,DualJ,RK,DualK,T,U}
    idict = Dict{Vec{RI,Int},Dict{Vec{RJ,Int},Op{D,T,U}}}()
    for (i, Ai) in A.comps
        for (k, Aik) in Ai
            Bk = get(B.comps, k, missing)
            if Bk !== missing
                for (j, Bkj) in Bk
                    AB = Aik * Bkj
                    jdict = get!(idict, i) do
                        Dict{Vec{RJ,Int},Op{D,T,U}}()
                    end
                    prev = get(jdict, j, missing)
                    next = prev === missing ? AB : prev + AB
                    jdict[j] = next
                end
            end
        end
    end
    FOp{D,RI,DualI,RJ,DualJ,T,U}(idict)
end

function Base.:\(A::FOp{D,RI,DualI,RJ,DualJ,T,U},
                 f::Form{D,RI,DualI,T,U})::Form{D,RJ,DualJ,T,U} where
        {D,RI,DualI,RJ,DualJ,T,U}
    @assert RI == 0 && RJ == 0
    Form(Dict(() => A.comps[Vec{0,Int}(())][Vec{0,Int}(())] \ f[()]))
end



function diff(f::Fun{D,T,U}, dir::Int)::Fun{D,T,U} where {D,T <: Number,U}
    @assert 1 <= dir <= D

    dom = f.dom
    @assert !dom.staggered[dir]
    rdom = makestaggered(dom, dom.staggered | unitvec(Val(D), dir))
    @assert rdom.staggered[dir]

    di = CartesianIndex(ntuple(d->d == dir, D))

    cs = f.coeffs
    rcs = Array{U}(undef, rdom.n.elts)
    if !dom.dual
        for i in CartesianIndices(size(rcs))
            rcs[i] = cs[i + di] - cs[i]
        end
    else
        for i in CartesianIndices(size(rcs))
            if i[dir] == 1
                # # Assume value at infinity is zero
                # rcs[i] = cs[i]
                # # Use off-centred stencils (why?)
                # rcs[i] = cs[i+di] - cs[i]
                # Zero works best
                rcs[i] = 0
            elseif i[dir] == size(rcs, dir)
                # rcs[i] = - cs[i-di]
                # rcs[i] = cs[i-di] - cs[i-2di]
                rcs[i] = 0
            else
                rcs[i] = cs[i] - cs[i - di]
            end
        end
    end

    Fun{D,T,U}(rdom, rcs)
end

function diff(dom::Domain{D,T}, dir::Int)::Op{D,T,T} where {D,T <: Number}
    @assert 1 <= dir <= D

    @assert !dom.staggered[dir]
    rdom = makestaggered(dom, dom.staggered | unitvec(Val(D), dir))
    @assert rdom.staggered[dir]

    di = CartesianIndex(ntuple(d->d == dir, D))

    ni = rdom.n
    nj = dom.n
    stri = strides(ni)
    idxi(i::CartesianIndex{D}) = 1 + sum((i[d] - 1) * stri[d] for d in 1:D)
    strj = strides(nj)
    idxj(i::CartesianIndex{D}) = 1 + sum((i[d] - 1) * strj[d] for d in 1:D)
    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, 2 * prod(ni))
    sizehint!(J, 2 * prod(ni))
    sizehint!(V, 2 * prod(ni))
    function ins!(i, j, v)
        @assert all(0 .< i.I .<= ni)
        @assert all(0 .< j.I .<= nj)
        push!(I, idxi(i))
        push!(J, idxj(j))
        push!(V, v)
    end

    if !dom.dual
        for i in CartesianIndices(ni.elts)
            # rcs[i] = cs[i+di] - cs[i]
            ins!(i, i, -1)
            ins!(i, i + di, 1)
        end
    else
        for i in CartesianIndices(ni.elts)
            if i[dir] == 1
                # # Assume value at infinity is zero
                # rcs[i] = cs[i]
                # # Use off-centred stencils (why?)
                # rcs[i] = cs[i+di] - cs[i]
                # Zero works best
                # rcs[i] = 0
            elseif i[dir] == ni[dir]
                # rcs[i] = - cs[i-di]
                # rcs[i] = cs[i-di] - cs[i-2di]
                # rcs[i] = 0
            else
                # rcs[i] = cs[i] - cs[i-di]
                ins!(i, i - di, -1)
                ins!(i, i, 1)
            end
        end
    end
    mat = sparse(I, J, V, prod(ni), prod(nj))

    Op{D,T,T}(rdom, dom, mat)
end



export integral
function integral(f::Form{D,R,false,T,U})::U where {D, R, T, U}
    sum(sum(fi.coeffs) for (i,fi) in f.comps)
end

function LinearAlgebra.dot(f::Form, g::Form)::Form
    integral(wedge(f, star(g)))
end



export boundary

@generated function boundary(f::Form{D,R,false,T,U})::Form{D,R,false,T,U} where
        {D, R, T, U}
    @assert 0 <= R < D
    quote
        dom = f.dom
        rd = Dict{Vec{R,Int}, Fun{D,T,U}}()
        $([begin
            staggered = Vec{D,Bool}(staggeredc.I)
            if count(staggered) == R
                idx = Vec{R,Int}(Tuple(staggered2idx(staggered)))
                quote
                    fi = f[$idx]
                    ridom = fi.dom
                    rc = Array{U}(undef, ridom.n.elts)
                    for i in CartesianIndices(size(rc))
                        rci = U(0)
                        $([begin
                            if !staggered[d]
                                s = bitsign(count(d .> idx))
                                quote
                                    if i[$d] == 1
                                        rci -= $s * fi[i]
                                    elseif i[$d] == ridom.n[$d]
                                        rci += $s * fi[i]
                                    end
                                end
                            end
                        end for d in 1:D]...)
                        rc[i] = rci
                    end
                    rd[$idx] = Fun(ridom, rc)
                end
            end
        end for staggeredc in CartesianIndices(ntuple(d -> 0:1, D))]...)
        Form(rd)
    end
end

function boundary(::Val{0}, ::Val{false},
                  dom::Domain{D,T})::FOp{D,0,false,0,false,T,T} where
        {D,T <: Number}
    n = dom.n

    str = strides(n)
    len = prod(n)
    idx(i::CartesianIndex{D}) = 1 + sum((i[d] - 1) * str[d] for d in 1:D)

    I = Int[]
    J = Int[]
    V = T[]
    maxsize = 2 * sum(len ÷ dom.n[d] for d in 1:D)
    sizehint!(I, maxsize)
    sizehint!(J, maxsize)
    sizehint!(V, maxsize)
    function ins!(i, j, v)
        push!(I, idx(i))
        push!(J, idx(j))
        push!(V, v)
    end
    # Choose initial conditions for timelike and boundary conditions
    # for spacelike directions
    b1 = Vec{D,Int}(ntuple(d -> 1, D))
    b2 = Vec{D,Int}(ntuple(d -> dom.metric[d] < 0 ? 2 : n[d], D))
    for i in CartesianIndices(dom.n.elts)
        if any(i.I .== b1.elts) || any(i.I .== b2.elts)
            ins!(i, i, T(1))
        end
    end
    mat = sparse(I, J, V, len, len)
    comp00 = Op{D,T,T}(dom, dom, mat)

    comps = Dict(Vec{0,Int}(()) => Dict(Vec{0,Int}(()) => comp00))
    FOp{D,0,false,0,false,T,T}(comps)
end



export wedge

@generated function wedge(
        f::Form{D,RI,false,T,U},
        g::Form{D,RJ,false,T,U})::Form{D,RI + RJ,false,T,U} where {D,RI,RJ,T,U}
    R = RI + RJ
    mksym(sym, inds) = Symbol(sym, [Symbol("_", i) for i in inds]...)
    quote
        dom = f.dom
        @assert g.dom == dom
        $([begin
            fstag = Vec{D,Bool}(ntuple(d->Bool(fstagc[d]), D))
            if count(fstag) == RI
                fidx = Vec{RI,Int}(Tuple(staggered2idx(fstag)))
                quote
                    $(mksym(:fcomps, fidx)) = f[$fidx]
                end # quote
            end # if
        end for fstagc in CartesianIndices(ntuple(d->0:1, D))]...)
        $([begin
            gstag = Vec{D,Bool}(ntuple(d->Bool(gstagc[d]), D))
            if count(gstag) == RJ
                gidx = Vec{RJ,Int}(Tuple(staggered2idx(gstag)))
                quote
                    $(mksym(:gcomps, gidx)) = g[$gidx]
                end # quote
            end # if
        end for gstagc in CartesianIndices(ntuple(d->0:1, D))]...)
        rcomps = Dict{Vec{$R,Int}, Fun{D,T,U}}()
        # Loop over all components of the result
        $([begin
            rstag = Vec{D,Bool}(ntuple(d->Bool(rstagc[d]), D))
            if count(rstag) == R
                ridx = Vec{R,Int}(Tuple(staggered2idx(rstag)))
                quote
                    rdom = makestaggered(dom, $rstag)
                    rc = Array{U}(undef, rdom.n.elts)
                    # Loop over all R-cells
                    for i in CartesianIndices(size(rc))
                        rci = U(0)
                        # Loop over all contributing components of f and g
                        $([begin
                            fstag = Vec{D,Bool}(ntuple(d->Bool(fstagc[d]), D))
                            if all(~fstag | rstag) && count(fstag) == RI
                                gstag = rstag & ~fstag
                                @assert count(gstag) == RJ
                                @assert all(~ (fstag & gstag))
                                @assert all(fstag | gstag == rstag)
                                fidx = Vec{RI,Int}(Tuple(staggered2idx(fstag)))
                                gidx = Vec{RJ,Int}(Tuple(staggered2idx(gstag)))
                                qidx = Int[fidx..., gidx...]
                                s = levicivita(sortperm(qidx))
                                @assert abs(s) == 1
                                # Loop over all vertices of the R-cell
                                dirange = ntuple(d -> 0 : Int(rstag[d]), D)
                                quote
                                    $([begin
                                        fdi = CartesianIndex(ntuple(d -> fstag[d] ? 0 : di[d], D))
                                        gdi = CartesianIndex(ntuple(d -> gstag[d] ? 0 : di[d], D))
                                        @assert all(fdi + gdi == di)
                                        quote
                                            rci += $s * ($(mksym(:fcomps, fidx))[i + $fdi] * $(mksym(:gcomps, gidx))[i + $gdi])
                                        end # quote
                                    end for di in CartesianIndices(dirange)]...)
                                end # quote
                            end # if
                        end for fstagc in CartesianIndices(ntuple(d->0:1, D))]...)
                        rc[i] = rci / 2^$R
                    end # for i
                    rcomps[$ridx] = Fun{D,T,U}(rdom, rc)
                end # quote
            end # if
        end for rstagc in CartesianIndices(ntuple(d->0:1, D))]...)
        Form(rcomps)
    end # quote
end

@generated function wedge(
        f::Form{D,RI,false,T,U},
    g::Form{D,RJ,true,T,U})::Form{D,RI+RJ,false,T,U} where {D,RI,RJ,T,U}
    R = RI + RJ
    mksym(sym, inds) = Symbol(sym, [Symbol("_", i) for i in inds]...)
    quote
        dom = f.dom
        @assert makeunstaggered(makeprimal(g.dom)) == dom
        $([begin
            fstag = Vec{D,Bool}(ntuple(d->Bool(fstagc[d]), D))
            if count(fstag) == RI
                fidx = Vec{RI,Int}(Tuple(staggered2idx(fstag)))
                quote
                    $(mksym(:fcomps, fidx)) = f[$fidx]
                end # quote
            end # if
        end for fstagc in CartesianIndices(ntuple(d->0:1, D))]...)
        $([begin
            gstag = Vec{D,Bool}(ntuple(d->Bool(gstagc[d]), D))
            if count(gstag) == RJ
                gidx = Vec{RJ,Int}(Tuple(staggered2idx(gstag)))
                quote
                    $(mksym(:gcomps, gidx)) = g[$gidx]
                end # quote
            end # if
        end for gstagc in CartesianIndices(ntuple(d->0:1, D))]...)
        rcomps = Dict{Vec{$R,Int}, Fun{D,T,U}}()
        # Loop over all components of the result
        $([begin
            rstag = Vec{D,Bool}(ntuple(d->Bool(rstagc[d]), D))
            if count(rstag) == R
                ridx = Vec{R,Int}(Tuple(staggered2idx(rstag)))
                quote
                    rdom = makestaggered(dom, $rstag)
                    rc = Array{U}(undef, rdom.n.elts)
                    # Loop over all R-cells
                    for i in CartesianIndices(size(rc))
                        rci = U(0)
                        # Loop over all contributing components of f and g
                        $([begin
                            fstag = Vec{D,Bool}(ntuple(d->Bool(fstagc[d]), D))
                            if all(~fstag | rstag) && count(fstag) == RI
                                gstag = rstag & ~fstag
                                @assert count(gstag) == RJ
                                @assert all(~ (fstag & gstag))
                                @assert all(fstag | gstag == rstag)
                                fidx = Vec{RI,Int}(Tuple(staggered2idx(fstag)))
                                gidx = Vec{RJ,Int}(Tuple(staggered2idx(gstag)))
                                qidx = Int[fidx..., gidx...]
                                s = levicivita(sortperm(qidx))
                                @assert abs(s) == 1
                                # Loop over all vertices of the R-cell
                                dirange = ntuple(d -> 0 : Int(rstag[d]), D)
                                quote
                                    $([begin
                                        # Loop over all contributing cells
                                        fdi = CartesianIndex(ntuple(d -> fstag[d] ? 0 : di[d], D))
                                        gdirange = ntuple(D) do d
                                            if !rstag[d] & gstag[d]
                                                di[d]:di[d]
                                            elseif !rstag[d] & !gstag[d]
                                                @assert di[d] == 0
                                                -1:0
                                            elseif rstag[d] & gstag[d]
                                                di[d]:di[d]
                                            elseif rstag[d] & !gstag[d]
                                                0:0
                                            else
                                                @assert false
                                            end
                                        end
                                        quote
                                            rcg = U(0)
                                            c = 0
                                            $([quote
                                                if (&)($([!rstag[d] & !gstag[d] ? :(1 <= i[$d] + $gdi[$d] <= size($(mksym(:gcomps, gidx)).coeffs,$d)) : :(true) for d in 1:D]...))
                                                    rcg += $(mksym(:fcomps, fidx))[i + $fdi] * $(mksym(:gcomps, gidx))[i + $gdi]
                                                    c += 1
                                                end # if
                                            end for gdi in CartesianIndices(gdirange)]...)
                                            rci += $s * rcg / c
                                        end # quote
                                   end for di in CartesianIndices(dirange)]...)
                               end # quote
                            end # if
                        end for fstagc in CartesianIndices(ntuple(d->0:1, D))]...)
                        rc[i] = rci / 2^$R
                    end # for i
                    rcomps[$ridx] = Fun{D,T,U}(rdom, rc)
                end # quote
            end # if
       end for rstagc in CartesianIndices(ntuple(d->0:1, D))]...)
       Form(rcomps)
    end # quote
end



export star

# star[k] ∘ star[n-k] = (-1)^(k (n-k))   [only for Euclidean manifolds?]

function star(form::Form{D,R,Dual,T,U})::Form{D,D - R,!Dual,T,U} where
        {D,R,Dual,T <: Number,U}
    @assert D >= 0
    @assert 0 <= R <= D

    dx = spacing(form.dom)
    rcomps = Dict{Vec{D - R,Int},Fun{D,T,U}}()
    for (idx, comp) in form.comps
        dom = comp.dom
        idx::Vec{R,Int}
        @assert collect(idx) == staggered2idx(dom.staggered)
        rdom = makedual(dom, !Dual)
        ridx = Vec{D - R,Int}(Tuple(staggered2idx(rdom.staggered)))
        s = levicivita([idx..., ridx...]) * prod(dom.metric[idx])
        scale = prod(!rdom.staggered[d] ? inv(dx[d]) : dx[d] for d in 1:D)
        rcomp = U(s * scale) * Fun(rdom, comp.coeffs)
        @assert !haskey(rcomps, ridx)
        rcomps[ridx] = rcomp
    end
    Form(rcomps)
end

function star(::Val{R}, ::Val{Dual},
              dom::Domain{D,T})::FOp{D,D - R,!Dual,R,Dual,T,T} where
        {R,Dual,D,T <: Number}
    @assert D >= 0
    @assert 0 <= R <= D

    dom = makedual(dom, Dual)
    rcomps = Dict{Vec{D - R,Int},Dict{Vec{R,Int},Op{D,T,T}}}()
    for staggeredc in CartesianIndices(ntuple(d->0:1, D))
        staggered = Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
        if count(staggered) == R

            odom = makestaggered(dom, staggered)
            rdom = makedual(odom, !odom.dual)

            ni = rdom.n
            nj = odom.n
            @assert all(ni == nj)
            stri = strides(ni)
            idxi(i::CartesianIndex{D}) =
                1 + sum((i[d] - 1) * stri[d] for d in 1:D)
            strj = strides(nj)
            idxj(i::CartesianIndex{D}) =
                1 + sum((i[d] - 1) * strj[d] for d in 1:D)
            I = Int[]
            J = Int[]
            V = T[]
            sizehint!(I, prod(ni))
            sizehint!(J, prod(ni))
            sizehint!(V, prod(ni))
            function ins!(i, j, v)
                @assert all(0 .< i.I .<= ni)
                @assert all(0 .< j.I .<= nj)
                push!(I, idxi(i))
                push!(J, idxj(j))
                push!(V, v)
            end

            dx = spacing(rdom)
            oidx = Vec{R,Int}(Tuple(staggered2idx(odom.staggered)))
            ridx = Vec{D - R,Int}(Tuple(staggered2idx(rdom.staggered)))
            s = levicivita([oidx..., ridx...]) * prod(dom.metric[oidx])
            scale = prod(!rdom.staggered[d] ? inv(dx[d]) : dx[d] for d in 1:D)

            for i in CartesianIndices(ni.elts)
                ins!(i, i, s * scale)
            end
            mat = sparse(I, J, V, prod(ni), prod(nj))

            rcomps[ridx] = Dict(oidx => Op{D,T,T}(rdom, odom, mat))
        end
    end

    FOp{D,D - R,!Dual,R,Dual,T,T}(rcomps)
end



export deriv

function deriv(form::Form{D,R,Dual,T,U})::Form{D,R + 1,Dual,T,U} where
        {D,R,Dual,T <: Number,U}
    @assert D >= 0
    @assert 0 <= R <= D
    @assert R + 1 <= D

    idict = Dict{Vec{R + 1,Int},Fun{D,T,U}}()
    icount = 0
    for (j, fj) in form.comps
        for dir in 1:D
            if !fj.dom.staggered[dir]
                c = count(dir .> j)
                s = bitsign(c)
                i = Vec{R + 1,Int}(Tuple(sort([dir, j...])))
                dfj = s * diff(fj, dir)
                prev = get(idict, i, missing)
                idict[i] = prev === missing ? dfj : prev + dfj
                icount += 1
            end
        end
    end
    @assert icount == (D - R) * length(form.comps)
    Form(idict)
end

function deriv(::Val{R}, ::Val{Dual},
               dom::Domain{D,T})::FOp{D,R + 1,Dual,R,Dual,T,T} where
        {R,Dual,D,T <: Number}
    @assert D >= 0
    @assert 0 <= R <= D
    @assert R + 1 <= D

    dom = makedual(dom, Dual)
    idict = Dict{Vec{R + 1,Int},Dict{Vec{R,Int},Op{D,T,T}}}()
    for staggeredc in CartesianIndices(ntuple(d->0:1, D))
        staggered = Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
        if count(staggered) == R
            j = Vec{R,Int}(Tuple(staggered2idx(staggered)))
            domj = makestaggered(dom, staggered)
            for dir in 1:D
                if !domj.staggered[dir]
                    c = count(dir .> j)
                    s = bitsign(c)
                    i = Vec{R + 1,Int}(Tuple(sort([dir, j...])))
                    dfj = s * diff(domj, dir)
                    jdict = get!(idict, i) do
                        Dict{Vec{R,Int},Op{D,T,T}}()
                    end
                    @assert !haskey(jdict, j)
                    jdict[j] = dfj
                end
            end
        end
    end
    FOp{D,R + 1,Dual,R,Dual,T,T}(idict)
end



export coderiv
function coderiv(form::Form{D,R,Dual,T,U})::Form{D,R - 1,Dual,T,U} where
        {D,R,Dual,T <: Number,U}
    @assert D >= 0
    @assert 0 <= R <= D
    @assert R - 1 >= 0
    # TODO: Improve performance
    # s = bitsign(D * (R+1) + 1) * prod(form.dom.metric)
    s = bitsign(iseven(D) && isodd(R)) * prod(form.dom.metric)
    T(s) * star(deriv(star(form)))
end
function coderiv(::Val{R}, ::Val{Dual},
                 dom::Domain{D,T})::FOp{D,R - 1,Dual,R,Dual,T,T} where
        {R,Dual,D,T <: Number}
    @assert D >= 0
    @assert 0 <= R <= D
    @assert R - 1 >= 0
    (star(Val(D - R + 1), Val(!Dual), dom) *
     deriv(Val(D - R), Val(!Dual), dom) *
     star(Val(R), Val(Dual), dom))
end



export laplace
function laplace(form::Form{D,R,Dual,T,U})::Form{D,R,Dual,T,U} where
        {D,R,Dual,T <: Number,U}
    @assert R == 0              # TODO: Add deriv ∘ coderiv
    # TODO: Improve performance
    r = zeros(Form{D,R,Dual,T,U}, form.dom)
    if R > 0
        r += deriv(coderiv(form))
    end
    if R < D
        r += coderiv(deriv(form))
    end
    r
end
function laplace(::Val{R}, ::Val{Dual},
                 dom::Domain{D,T})::FOp{D,R,Dual,R,Dual,T,T} where
        {R,Dual,D,T <: Number}
    op = zero(FOp{D,R,Dual,R,Dual,T,T})
    if R > 0
        op += deriv(Val(R-1), Val(Dual), dom) * coderiv(Val(R), Val(Dual), dom)
    end
    if R < D
        op += coderiv(Val(R+1), Val(Dual), dom) * deriv(Val(R), Val(Dual), dom)
    end
    op
end



export dirichlet
const dirichlet = boundary

function Ops.mix_op_bc(bnd::FOp{D,R,Dual,R,Dual,T,U},
                       iop::FOp{D,R,Dual,R,Dual,T,U},
                       bop::FOp{D,R,Dual,R,Dual,T,U},
                       dom::Domain{D,T})::FOp{D,R,Dual,R,Dual,T,U} where
        {D,R,Dual,T,U}
    di = Vec{D,Int}(ntuple(d -> dom.metric[d] < 0 ? 1 : 0, D))
    sh = shift(typeof(bnd), dom, di)

    id = one(typeof(bnd), dom)
    int = id - bnd

    int * sh * iop + bnd * bop
end

function Ops.mix_op_bc(bnd::FOp{D,R,Dual,R,Dual,T,U},
                       rhs::Form{D,R,Dual,T,U},
                       bvals::Form{D,R,Dual,T,U})::Form{D,R,Dual,T,U} where
        {D,R,Dual,T,U}
    dom = rhs.dom
    @assert bvals.dom == dom

    di = Vec{D,Int}(ntuple(d -> dom.metric[d] < 0 ? 1 : 0, D))
    sh = shift(typeof(bnd), dom, di)

    id = one(typeof(bnd), dom)
    int = id - bnd

    int * sh * rhs + bnd * bvals
end



export solve_dAlembert_Dirichlet
function solve_dAlembert_Dirichlet(pot::Form{D,0,false,T,U},
        bvals::Form{D,0,false,T,U})::Form{D,0,false,T,U} where {D,T <: Number,U}
    dom = pot.dom
    @assert bvals.dom == dom

    n = dom.n
    dx = spacing(dom)
    dx2 = dx .* dx

    di = ntuple(dir->CartesianIndex(ntuple(d->d == dir, D - 1)), D - 1)

    pc = pot[()].coeffs
    bc = bvals[()].coeffs
    sol = Array{U}(undef, dom.n.elts)
    if D == 2
        # Initial conditions
        sol[:,1] = bc[:,1]
        sol[:,2] = bc[:,2]
        for j = 3:n[2]
            # Boundary conditions
            sol[1,j] = bc[1,j]
            sol[end,j] = bc[end,j]
            # Wave equation
            for i in CartesianIndices(ntuple(d->2:n[d] - 1, D - 1))
                lsol = (+ (+ sol[i - di[1],j - 1]
                           - 2 * sol[i,j - 1]
                           + sol[i + di[1],j - 1]) / dx2[1])
                sol[i,j] = (- sol[i,j - 2] + 2 * sol[i,j - 1]
                            + dx2[2] * (lsol - pc[i,j - 1]))
            end
        end
    elseif D == 3
        # Initial conditions
        sol[:,:,1] = bc[:,:,1]
        sol[:,:,2] = bc[:,:,2]
        for j = 3:n[3]
            # Boundary conditions
            sol[1,:,j] = bc[1,:,j]
            sol[end,:,j] = bc[end,:,j]
            sol[:,1,j] = bc[:,1,j]
            sol[:,end,j] = bc[:,end,j]
            # Wave equation
            for i in CartesianIndices(ntuple(d->2:n[d] - 1, D - 1))
                lsol = (+ (+ sol[i - di[1],j - 1]
                           - 2 * sol[i,j - 1]
                           + sol[i + di[1],j - 1]) / dx2[1]
                        + (+ sol[i - di[2],j - 1]
                           - 2 * sol[i,j - 1]
                           + sol[i + di[2],j - 1]) / dx2[2])
                sol[i,j] = (- sol[i,j - 2] + 2 * sol[i,j - 1]
                            + dx2[3] * (lsol - pc[i,j - 1]))
            end
        end
    elseif D == 4
        # Initial conditions
        sol[:,:,:,1] = bc[:,:,:,1]
        sol[:,:,:,2] = bc[:,:,:,2]
        for j = 3:n[4]
            # Boundary conditions
            sol[1,:,:,j] = bc[1,:,:,j]
            sol[end,:,:,j] = bc[end,:,:,j]
            sol[:,1,:,j] = bc[:,1,:,j]
            sol[:,end,:,j] = bc[:,end,:,j]
            sol[:,:,1,j] = bc[:,:,1,j]
            sol[:,:,end,j] = bc[:,:,end,j]
            # Wave equation
            for i in CartesianIndices(ntuple(d->2:n[d] - 1, D - 1))
                lsol = (+ (+ sol[i - di[1],j - 1]
                           - 2 * sol[i,j - 1]
                           + sol[i + di[1],j - 1]) / dx2[1]
                        + (+ sol[i - di[2],j - 1]
                           - 2 * sol[i,j - 1]
                           + sol[i + di[2],j - 1]) / dx2[2]
                        + (+ sol[i - di[3],j - 1]
                           - 2 * sol[i,j - 1]
                           + sol[i + di[3],j - 1]) / dx2[3])
                sol[i,j] = (- sol[i,j - 2] + 2 * sol[i,j - 1]
                            + dx2[4] * (lsol - pc[i,j - 1]))
            end
        end
    else
        @assert false
    end

    Form(Dict(() => Fun(dom, sol)))
end



# ################################################################################
# 
# # TODO: REVISE AND REMOVE EVERYTHING BELOW
# 
# 
# 
# # TODO: REMOVE THIS
# function star1(form::Form{D,R,T,U})::Form{D,D-R,T,U} where {D, R, T<:Number, U}
#     di = ntuple(dir -> CartesianIndex(ntuple(d -> d==dir, D)), D)
#     dom = makeunstaggered(first(form).second)
#     dx = spacing(dom)
# 
#     if R == 0
#         if D == 2
#             u0 = form[()]
#             dom0 = u0.dom
#             dom2xy = makestaggered(dom0, Vec((true, true)))
#             cs0 = u0.coeffs
#             scs2xy = Array{U}(undef, dom2xy.n.elts)
#             for i in CartesianIndices(size(scs2xy))
#                 s = (+ cs0[i] + cs0[i + di[1]]
#                      + cs0[i + di[2]] + cs0[i + di[1] + di[2]])
#                 scs2xy[i] = 1/(dx[1]*dx[2]) * s / 4
#             end
#             return Form(Dict((1,2) => Fun{D,T,U}(dom2xy, scs2xy)))
#         else
#             @assert false
#         end
#     elseif R == 1
#         if D == 2
#             u1x = form[(1,)]
#             u1y = form[(2,)]
#             dom1x = u1x.dom
#             dom1y = u1y.dom
#             s1x = dom1x.staggered
#             s1y = dom1y.staggered
#             n1x = dom1x.n
#             n1y = dom1y.n
#             n = dom.n
#             cs1x = u1x.coeffs
#             cs1y = u1y.coeffs
#             scs1x = Array{U}(undef, n1x.elts)
#             for i in CartesianIndices(size(scs1x))
#                 s = U(0)
#                 c = 0
#                 if i[2] > 1
#                     s += cs1y[i-di[2]] + cs1y[i-di[2]+di[1]]
#                     c += 2
#                 end
#                 if i[2] < n[2]
#                     s += cs1y[i] + cs1y[i+di[1]]
#                     c += 2
#                 end
#                 scs1x[i] = - dx[2]/dx[1] * s / c
#             end
#             scs1y = Array{U}(undef, n1y.elts) 
#             for i in CartesianIndices(size(scs1y))
#                 s = U(0)
#                 c = 0
#                 if i[1] > 1
#                     s += cs1x[i-di[1]] + cs1x[i-di[1]+di[2]]
#                     c += 2
#                 end
#                 if i[1] < n[1]
#                     s += cs1x[i] + cs1x[i+di[2]]
#                     c += 2
#                 end
#                 scs1y[i] = + dx[1]/dx[2] * s / c
#             end
#             return Form(Dict((1,) => Fun{D,T,U}(dom1x, scs1x),
#                              (2,) => (Fun{D,T,U}(dom1y, scs1y))))
#         else
#             @assert false
#         end
#     elseif R == 2
#         if D == 2
#             u2xy = form[(1,2)]
#             dom2xy = u2xy.dom
#             dom0 = makeunstaggered(dom2xy)
#             n0 = dom0.n
#             cs2xy = u2xy.coeffs
#             scs0 = Array{U}(undef, n0.elts)
#             for i in CartesianIndices(size(scs0))
#                 j = CartesianIndex(min.(dom2xy.n.elts, i.I))
#                 scs0[i] = dx[1]*dx[2] * cs2xy[j]
#             end
#             return Form(Dict(() => Fun{D,T,U}(dom0, scs0)))
#         else
#             @assert false
#         end
#     else
#         @assert false
#     end
# end
# 
# 
# 
# function deriv2(form::Form{D,R,T,U})::Form{D,R+1,T,U} where {D, R, T<:Number, U}
#     if R == 0
#         if D == 2
#             u0 = form[()]
#             r1x = diff(u0, 1)
#             r1y = diff(u0, 2)
#             return Form(Dict((1,) => r1x, (2,) => r1y))
#         else
#             @assert false
#         end
#     elseif R == 1
#         if D == 2
#             u1x = form[(1,)]
#             u1y = form[(2,)]
#             r2xy = - diff(u1x, 2) + diff(u1y,1)
#             return Form(Dict((1,2) => r2xy))
#         else
#             @assert false
#         end
#     else
#         @assert false
#     end
# end
# 
# # TODO: REMOVE THIS
# function deriv1(form::Form{D,R,T,U})::Form{D,R+1,T,U} where {D, R, T<:Number, U}
#     di = ntuple(dir -> CartesianIndex(ntuple(d -> d==dir, D)), D)
#     dx = spacing(form.dom)
# 
#     if R == 0
#         f0 = form[()]
#         cs0 = f0.coeffs
#         if D == 2
#             dom1x = makestaggered(f0.dom, unitvec(Val(D), 1))
#             dom1y = makestaggered(f0.dom, unitvec(Val(D), 2))
#             dcs1x = Array{U}(undef, dom1x.n.elts)
#             for i in CartesianIndices(size(dcs1x))
#                 dcs1x[i] = (cs0[i + di[1]] - cs0[i]) / dx[1]
#             end
#             dcs1y = Array{U}(undef, dom1y.n.elts)
#             for i in CartesianIndices(size(dcs1y))
#                 dcs1y[i] = (cs0[i + di[2]] - cs0[i]) / dx[2]
#             end
#             return Form(Dict((1,) => Fun(dom1x, dcs1x),
#                              (2,) => Fun(dom1y, dcs1y)))
#         elseif D == 3
#             dom1x = makestaggered(f0.dom, unitvec(Val(D), 1))
#             dom1y = makestaggered(f0.dom, unitvec(Val(D), 2))
#             dom1z = makestaggered(f0.dom, unitvec(Val(D), 3))
#             dcs1x = Array{U}(undef, dom1x.n.elts)
#             for i in CartesianIndices(size(dcs1x))
#                 dcs1x[i] = (cs0[i + di[1]] - cs0[i]) / dx[1]
#             end
#             dcs1y = Array{U}(undef, dom1y.n.elts)
#             for i in CartesianIndices(size(dcs1y))
#                 dcs1y[i] = (cs0[i + di[2]] - cs0[i]) / dx[2]
#             end
#             dcs1z = Array{U}(undef, dom1z.n.elts)
#             for i in CartesianIndices(size(dcs1z))
#                 dcs1z[i] = (cs0[i + di[3]] - cs0[i]) / dx[3]
#             end
#             return Form(Dict((1,) => Fun(dom1x, dcs1x),
#                              (2,) => Fun(dom1y, dcs1y),
#                              (3,) => Fun(dom1z, dcs1z)))
#         else
#             @assert false
#         end
#     elseif R == 1
#         if D == 2
#             f1x = form[(1,)]
#             f1y = form[(2,)]
#             cs1x = f1x.coeffs
#             cs1y = f1y.coeffs
#             dom2xy = makestaggered(f1x.dom, Vec((true, true)))
#             dcs2xy = Array{U}(undef, dom2xy.n.elts)
#             for i in CartesianIndices(size(dcs2xy))
#                 dcs2xy[i] = (+ (cs1y[i + di[1]] - cs1y[i]) / dx[1]
#                              - (cs1x[i + di[2]] - cs1x[i]) / dx[2])
#             end
#             return Form(Dict((1,2) => Fun(dom2xy, dcs2xy)))
#         else
#             @assert false
#         end
#     else
#         @assert false
#     end
# end
# 
# 
# 
# function coderiv2(form::Form{D,R,T,U})::Form{D,R-1,T,U} where
#         {D, R, T<:Number, U}
#     # (star ∘ deriv ∘ star)(form)
#     if R == 1
#         if D == 2
#             u1x = form[(1,)]
#             u1y = form[(2,)]
#             r0 = codiff(u1x, 1) + codiff(u1y, 2)
#             return Form(Dict(() => r0))
#         else
#             @assert false
#         end
#     else
#         @assert false
#     end
# end
# 
# # TODO: REMOVE THIS
# function coderiv1(form::Form{D,R,T,U})::Form{D,R-1,T,U} where
#         {D, R, T<:Number, U}
#     di = ntuple(dir -> CartesianIndex(ntuple(d -> d==dir, D)), D)
#     dx = spacing(form.dom)
# 
#     if R == 1
#         if D == 2
#             f1x = form[(1,)]
#             f1y = form[(2,)]
#             cs1x = f1x.coeffs
#             cs1y = f1y.coeffs
#             dom0 = makeunstaggered(f1x.dom)
#             n0 = dom0.n
#             jmin = 0 * n0 .+ 2
#             jmax = n0 .- 1
#             dcs0 = Array{U}(undef, n0.elts)
#             for i in CartesianIndices(size(dcs0))
#                 j = CartesianIndex(max.(jmin, min.(jmax, Vec(i.I))).elts)
#                 dcs0[i] = (+ (cs1x[j] - cs1x[j - di[1]]) / dx[1]
#                            + (cs1y[j] - cs1y[j - di[2]]) / dx[2])
#             end
#             return Form(Dict(() => Fun(dom0, dcs0)))
#         elseif D == 3
#             f1x = form[(1,)]
#             f1y = form[(2,)]
#             f1z = form[(3,)]
#             cs1x = f1x.coeffs
#             cs1y = f1y.coeffs
#             cs1z = f1z.coeffs
#             dom0 = makeunstaggered(f1x.dom)
#             n0 = dom0.n
#             jmin = 0 * n0 .+ 2
#             jmax = n0 .- 1
#             dcs0 = Array{U}(undef, n0.elts)
#             for i in CartesianIndices(size(dcs0))
#                 j = CartesianIndex(max.(jmin, min.(jmax, Vec(i.I))).elts)
#                 dcs0[i] = (+ (cs1x[j] - cs1x[j - di[1]]) / dx[1]
#                            + (cs1y[j] - cs1y[j - di[2]]) / dx[2]
#                            + (cs1z[j] - cs1z[j - di[3]]) / dx[3])
#             end
#             return Form(Dict(() => Fun(dom0, dcs0)))
#         else
#             @assert false
#         end
#     else
#         @assert false
#     end
# end



export laplace1
function laplace1(form::Form{D,R,T,U})::Form{D,R,T,U} where
        {D,R,T <: Number,U}
    di = ntuple(dir->CartesianIndex(ntuple(d->d == dir, D)), D)
    dx = spacing(form.dom)

    if R == 0
        f0 = form[()]
        cs0 = f0.coeffs
        dom0 = f0.dom
        dcs0 = Array{U}(undef, dom0.n.elts)
        n = dom0.n
        for i in CartesianIndices(size(dcs0))
            if all(1 .< i.I .< n)
                dcs0[i] = sum((cs0[i - di[d]] - 2 * cs0[i] + cs0[i + di[d]]) / dx[d]^2
                    for d in 1:D)
            else
                dcs0[i] = 0
            end
        end
        return Form(Dict(() => Fun(dom0, dcs0)))
    else
        @assert false
    end
end

function laplace1(::Val{R}, ::Val{Dual},
                  dom::Domain{D,T})::FOp{D,R,Dual,R,Dual,T,T} where
        {D,R,Dual,T <: Number}
    @assert R == 0
    @assert !Dual
    n = dom.n
    di = ntuple(dir->CartesianIndex(ntuple(d->d == dir, D)), D)
    dx = spacing(dom)
    dx2 = dx .* dx

    str = strides(n)
    len = prod(n)
    idx(i::CartesianIndex{D}) = 1 + sum((i[d] - 1) * str[d] for d in 1:D)

    I = Int[]
    J = Int[]
    V = T[]
    maxsize = 3 * D * len
    sizehint!(I, maxsize)
    sizehint!(J, maxsize)
    sizehint!(V, maxsize)
    function ins!(i, j, v)
        @assert all(1 .<= i.I .<= n)
        @assert all(1 .<= j.I .<= n)
        push!(I, idx(i))
        push!(J, idx(j))
        push!(V, v)
    end
    for i in CartesianIndices(dom.n.elts)
        for d in 1:D
            if 1 < i[d] < n[d]
                ins!(i, i - di[d], 1 / dx2[d])
                ins!(i, i, -2 / dx2[d])
                ins!(i, i + di[d], 1 / dx2[d])
            end
        end
    end
    mat = sparse(I, J, V, len, len)
    FOp{D,R,Dual,R,Dual,T,T}(Dict(Vec{0,Int}(()) =>
                                        Dict(Vec{0,Int}(()) =>
                                        Op{D,T,T}(dom, dom, mat))))
end

end
