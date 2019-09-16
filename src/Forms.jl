"""
Differential forms
"""
module Forms

using Combinatorics
using SparseArrays

using ..Defs
using ..Domains
using ..Funs
using ..Ops
using ..Vecs



export Form
struct Form{D, R, T, U}
    dom::Domain{D, T}
    comps::Dict{Vec{R,Int}, Fun{D,T,U}}
    # infval::U                   # value at infinity

    function Form(dict::Dict{Vec{R,Int}, Fun{D,T,U}}) where
            {D, R, T, U}
        @assert D isa Int && 0 <= D
        @assert R isa Int && 0 <= R <= D

        i0 = Vec{R,Int}(ntuple(d -> d, R))
        f0 = dict[i0]
        dom = makeunstaggered(f0.dom)

        count = 0
        for i in CartesianIndices(ntuple(d -> D, R))
            !all(i[d+1] > i[d] for d in 1:R-1) && continue
            count += 1
            f = dict[Vec{R,Int}(i.I)]
            @assert all(f.dom.staggered[d] == (d in i.I) for d in 1:D)
            if !f.dom.dual
                @assert all(f.dom.n + f.dom.staggered .== dom.n)
            else
                @assert all(f.dom.n - f.dom.staggered .== dom.n)
            end
            @assert f.dom.metric == dom.metric
            @assert f.dom.xmin == dom.xmin
            @assert f.dom.xmax == dom.xmax
        end
        @assert length(dict) == count

        new{D, R, T, U}(dom, copy(dict))
    end
end

function Form(dict::Dict{NTuple{R,Int}, Fun{D,T,U}}) where {D, R, T, U}
    Form(Dict(Vec{R,Int}(k) => v for (k,v) in dict))
end

function Base.getindex(form::Form{D,R,T,U}, i::Vec{R,Int})::Fun{D,T,U} where
        {D,R,T,U}
    form.comps[i]
end
function Base.getindex(form::Form{D,R,T,U}, i::NTuple{R,Int})::Fun{D,T,U} where
        {D,R,T,U}
    form.comps[Vec{R,Int}(i)]
end



# struct FOp{D, R, T}
# end



function diff(f::Fun{D,T,U}, dir::Int)::Fun{D,T,U} where {D, T<:Number, U}
    @assert 1 <= dir <= D

    dom = f.dom
    @assert !dom.staggered[dir]
    rdom = makestaggered(dom, dom.staggered | unitvec(Val(D), dir))
    @assert rdom.staggered[dir]

    di = CartesianIndex(ntuple(d -> d==dir, D))
    dx = spacing(dom)[dir]

    cs = f.coeffs
    rcs = Array{U}(undef, rdom.n.elts)
    if !dom.dual
        for i in CartesianIndices(size(rcs))
            rcs[i] = cs[i+di] - cs[i]
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
            elseif i[dir] == size(rcs,dir)
                # rcs[i] = - cs[i-di]
                # rcs[i] = cs[i-di] - cs[i-2di]
                rcs[i] = 0
            else
                rcs[i] = cs[i] - cs[i-di]
            end
        end
    end

    Fun{D,T,U}(rdom, rcs)
end

# function codiff(f::Fun{D,T,U}, dir::Int)::Fun{D,T,U} where {D, T<:Number, U}
#     @assert 1 <= dir <= D
# 
#     dom = f.dom
#     @assert dom.staggered[dir]
#     rdom = makestaggered(dom, dom.staggered & ~unitvec(Val(D), dir))
#     @assert !rdom.staggered[dir]
# 
#     di = CartesianIndex(ntuple(d -> d==dir, D))
#     dx = spacing(dom)[dir]
# 
#     cs = f.coeffs
#     rcs = Array{U}(undef, rdom.n.elts)
#     for i in CartesianIndices(size(rcs))
#         j = CartesianIndex(ntuple(d -> (d==dir
#                                         ? max(2, min(dom.n[d], i[d]))
#                                         : i[d]), D))
#         rcs[i] = (cs[j] - cs[j-di]) / dx
#     end
# 
#     Fun{D,T,U}(rdom, rcs)
# end
# 
# function avg(f::Fun{D,T,U}, dirs::Vec{D,Bool})::Fun{D,T,U} where
#         {D, T<:Number, U}
#     dom = f.dom
#     @assert !any(dom.staggered & dirs)
#     rdom = makestaggered(dom, dom.staggered | dirs)
#     @assert all(rdom.staggered & dirs == dirs)
# 
#     # di = ntuple(dir -> CartesianIndex(ntuple(d -> d==dir, D)), D)
#     # dx = spacing(dom)[dir]
# 
#     cs = f.coeffs
#     rcs = Array{U}(undef, rdom.n.elts)
#     # jmin = CartesianIndex(ntuple(d -> 2, D))
#     # jmax = CartesianIndex(ntuple(d -> dom.n[d] - dirs[d], D))
#     for i in CartesianIndices(size(rcs))
#         s = U(0)
#         # c = U(0)
#         for di in CartesianIndices(ntuple(d -> 0:dirs[d], D))
#             s += cs[i + di]
#             # j = i + di
#             # if j >= jmin && j <= jmax
#             #     s += cs[j]
#             #     c += U(1)
#             # end
#         end
#         # rcs[i] = s / c
#         rcs[i] = s / (1 << sum(dirs.elts))
#     end
# 
#     Fun{D,T,U}(rdom, rcs)
# end
# 
# # avg ∘ coavg = id
# function coavg(f::Fun{D,T,U}, dir::Int)::Fun{D,T,U} where {D, T<:Number, U}
#     @assert 1 <= dir <= D
# 
#     dom = f.dom
#     @assert dom.staggered[dir]
#     rdom = makestaggered(dom, dom.staggered & ~unitvec(Val(D), dir))
#     @assert !rdom.staggered[dir]
# 
#     di = CartesianIndex(ntuple(d -> d==dir, D))
#     # dx = spacing(dom)[dir]
# 
#     perm = [dir, (1:dir-1)..., (dir+1:D)...]
#     pcs = permutedims(f.coeffs, perm)
#     prcs = Array{U}(undef, rdom.n.elts[perm])
#     for j in CartesianIndices(size(pcs)[2:end])
#         prcs[1,j] = 0
#         for i in 2:size(prcs,1)
#             prcs[i,j] = 2 * pcs[i-1,j] - prcs[i-1,j]
#         end
#         # This is a choice: Set highest mode (alternating sum) to zero
#         altavg =
#             sum(bitsign(i) * prcs[i,j] for i in 1:size(prcs,1)) / size(prcs,1)
#         for i in 1:size(prcs,1)
#             prcs[i,j] -= bitsign(i) * altavg
#         end
#     end
#     invperm = Array{Int}(undef, D)
#     invperm[perm] = [(1:D)...]
#     @assert perm[invperm] == [(1:D)...]
#     rcs = permutedims(prcs, invperm)
# 
#     Fun{D,T,U}(rdom, rcs)
# end



export star
# star[k] ∘ star[n-k] = (-1)^(k (n-k))   [only for Euclidean manifolds?]
function star(form::Form{D,R,T,U})::Form{D,D-R,T,U} where {D, R, T<:Number, U}
    dx = spacing(form.dom)

    function staggered2idx(staggered::Vec{D,Bool})::Vector{Int}
        idx = map(ib->ib[1], filter(ib->ib[2],
                                    collect(enumerate(staggered.elts))))
        @assert idx2staggered(idx) == staggered
        idx
    end
    function idx2staggered(idx::AbstractVector{Int})::Vec{D,Bool}
        staggered = falses(D)
        for i in idx
            @assert !staggered[i]
            staggered[i] = true
        end
        Vec(Tuple(staggered))
    end

    rcomps = Dict{NTuple{D-R,Int}, Fun{D,T,U}}()
    for (idx, comp) in form.comps
        dom = comp.dom
        @assert idx isa Vec{R, Int}
        @assert collect(idx) == staggered2idx(dom.staggered)
        rdom = makedual(dom, !dom.dual)
        ridx = Tuple(staggered2idx(rdom.staggered))
        s = levicivita([idx..., ridx...])
        scale = prod(!rdom.staggered[d] ? inv(dx[d]) : dx[d] for d in 1:D)
        rcomp = U(s * scale) * Fun(rdom, comp.coeffs)
        @assert ridx isa NTuple{D-R, Int}
        @assert !haskey(rcomps, ridx)
        rcomps[ridx] = rcomp
    end
    Form(rcomps)
end

export deriv
function deriv(form::Form{D,R,T,U})::Form{D,R+1,T,U} where {D, R, T<:Number, U}
    if R == 0
        u0 = form[()]
        return Form(Dict([(d,) => diff(u0, d) for d in 1:D]))
    elseif R == 1
        if D == 2
            u1x = form[(1,)]
            u1y = form[(2,)]
            r2xy = T(1)/2 * (diff(u1y, 1) - diff(u1x, 2))
            return Form(Dict((1,2) => r2xy))
        elseif D == 3
            u1x = form[(1,)]
            u1y = form[(2,)]
            u1z = form[(3,)]
            r2xy = T(1)/2 * (+ diff(u1y, 1) - diff(u1x, 2))
            r2xz = T(1)/2 * (- diff(u1x, 3) + diff(u1z, 1))
            r2yz = T(1)/2 * (+ diff(u1z, 2) - diff(u1y, 3))
            return Form(Dict((1,2) => r2xy, (1,3) => r2xz, (2,3) => r2yz))
        else
            @assert false
        end
    elseif R == 2
        if D == 3
            u2xy = form[(1,2)]
            u2xz = form[(1,3)]
            u2yz = form[(2,3)]
            r3xyz = + diff(u2yz, 1) - diff(u2xz, 2) - diff(u2xy, 3)
            return Form(Dict((1,2,3) => r3xyz))
        else
            @assert false
        end
    else
        @assert false
    end
end

export coderiv
function coderiv(form::Form{D,R,T,U})::Form{D,R-1,T,U} where
        {D, R, T<:Number, U}
    # TODO: Improve performance
    star(deriv(star(form)))
end

export laplace
function laplace(form::Form{D,R,T,U})::Form{D,R,T,U} where {D, R, T<:Number, U}
    # TODO: Improve performance
    coderiv(deriv(form))
end



################################################################################

# TODO: REVISE AND REMOVE EVERYTHING BELOW



# TODO: REMOVE THIS
function star1(form::Form{D,R,T,U})::Form{D,D-R,T,U} where {D, R, T<:Number, U}
    di = ntuple(dir -> CartesianIndex(ntuple(d -> d==dir, D)), D)
    dom = makeunstaggered(first(form).second)
    dx = spacing(dom)

    if R == 0
        if D == 2
            u0 = form[()]
            dom0 = u0.dom
            dom2xy = makestaggered(dom0, Vec((true, true)))
            cs0 = u0.coeffs
            scs2xy = Array{U}(undef, dom2xy.n.elts)
            for i in CartesianIndices(size(scs2xy))
                s = (+ cs0[i] + cs0[i + di[1]]
                     + cs0[i + di[2]] + cs0[i + di[1] + di[2]])
                scs2xy[i] = 1/(dx[1]*dx[2]) * s / 4
            end
            return Form(Dict((1,2) => Fun{D,T,U}(dom2xy, scs2xy)))
        else
            @assert false
        end
    elseif R == 1
        if D == 2
            u1x = form[(1,)]
            u1y = form[(2,)]
            dom1x = u1x.dom
            dom1y = u1y.dom
            s1x = dom1x.staggered
            s1y = dom1y.staggered
            n1x = dom1x.n
            n1y = dom1y.n
            n = dom.n
            cs1x = u1x.coeffs
            cs1y = u1y.coeffs
            scs1x = Array{U}(undef, n1x.elts)
            for i in CartesianIndices(size(scs1x))
                s = U(0)
                c = 0
                if i[2] > 1
                    s += cs1y[i-di[2]] + cs1y[i-di[2]+di[1]]
                    c += 2
                end
                if i[2] < n[2]
                    s += cs1y[i] + cs1y[i+di[1]]
                    c += 2
                end
                scs1x[i] = - dx[2]/dx[1] * s / c
            end
            scs1y = Array{U}(undef, n1y.elts) 
            for i in CartesianIndices(size(scs1y))
                s = U(0)
                c = 0
                if i[1] > 1
                    s += cs1x[i-di[1]] + cs1x[i-di[1]+di[2]]
                    c += 2
                end
                if i[1] < n[1]
                    s += cs1x[i] + cs1x[i+di[2]]
                    c += 2
                end
                scs1y[i] = + dx[1]/dx[2] * s / c
            end
            return Form(Dict((1,) => Fun{D,T,U}(dom1x, scs1x),
                             (2,) => (Fun{D,T,U}(dom1y, scs1y))))
        else
            @assert false
        end
    elseif R == 2
        if D == 2
            u2xy = form[(1,2)]
            dom2xy = u2xy.dom
            dom0 = makeunstaggered(dom2xy)
            n0 = dom0.n
            cs2xy = u2xy.coeffs
            scs0 = Array{U}(undef, n0.elts)
            for i in CartesianIndices(size(scs0))
                j = CartesianIndex(min.(dom2xy.n.elts, i.I))
                scs0[i] = dx[1]*dx[2] * cs2xy[j]
            end
            return Form(Dict(() => Fun{D,T,U}(dom0, scs0)))
        else
            @assert false
        end
    else
        @assert false
    end
end



function deriv2(form::Form{D,R,T,U})::Form{D,R+1,T,U} where {D, R, T<:Number, U}
    if R == 0
        if D == 2
            u0 = form[()]
            r1x = diff(u0, 1)
            r1y = diff(u0, 2)
            return Form(Dict((1,) => r1x, (2,) => r1y))
        else
            @assert false
        end
    elseif R == 1
        if D == 2
            u1x = form[(1,)]
            u1y = form[(2,)]
            r2xy = - diff(u1x, 2) + diff(u1y,1)
            return Form(Dict((1,2) => r2xy))
        else
            @assert false
        end
    else
        @assert false
    end
end

# TODO: REMOVE THIS
function deriv1(form::Form{D,R,T,U})::Form{D,R+1,T,U} where {D, R, T<:Number, U}
    di = ntuple(dir -> CartesianIndex(ntuple(d -> d==dir, D)), D)
    dx = spacing(form.dom)

    if R == 0
        f0 = form[()]
        cs0 = f0.coeffs
        if D == 2
            dom1x = makestaggered(f0.dom, unitvec(Val(D), 1))
            dom1y = makestaggered(f0.dom, unitvec(Val(D), 2))
            dcs1x = Array{U}(undef, dom1x.n.elts)
            for i in CartesianIndices(size(dcs1x))
                dcs1x[i] = (cs0[i + di[1]] - cs0[i]) / dx[1]
            end
            dcs1y = Array{U}(undef, dom1y.n.elts)
            for i in CartesianIndices(size(dcs1y))
                dcs1y[i] = (cs0[i + di[2]] - cs0[i]) / dx[2]
            end
            return Form(Dict((1,) => Fun(dom1x, dcs1x),
                             (2,) => Fun(dom1y, dcs1y)))
        elseif D == 3
            dom1x = makestaggered(f0.dom, unitvec(Val(D), 1))
            dom1y = makestaggered(f0.dom, unitvec(Val(D), 2))
            dom1z = makestaggered(f0.dom, unitvec(Val(D), 3))
            dcs1x = Array{U}(undef, dom1x.n.elts)
            for i in CartesianIndices(size(dcs1x))
                dcs1x[i] = (cs0[i + di[1]] - cs0[i]) / dx[1]
            end
            dcs1y = Array{U}(undef, dom1y.n.elts)
            for i in CartesianIndices(size(dcs1y))
                dcs1y[i] = (cs0[i + di[2]] - cs0[i]) / dx[2]
            end
            dcs1z = Array{U}(undef, dom1z.n.elts)
            for i in CartesianIndices(size(dcs1z))
                dcs1z[i] = (cs0[i + di[3]] - cs0[i]) / dx[3]
            end
            return Form(Dict((1,) => Fun(dom1x, dcs1x),
                             (2,) => Fun(dom1y, dcs1y),
                             (3,) => Fun(dom1z, dcs1z)))
        else
            @assert false
        end
    elseif R == 1
        if D == 2
            f1x = form[(1,)]
            f1y = form[(2,)]
            cs1x = f1x.coeffs
            cs1y = f1y.coeffs
            dom2xy = makestaggered(f1x.dom, Vec((true, true)))
            dcs2xy = Array{U}(undef, dom2xy.n.elts)
            for i in CartesianIndices(size(dcs2xy))
                dcs2xy[i] = (+ (cs1y[i + di[1]] - cs1y[i]) / dx[1]
                             - (cs1x[i + di[2]] - cs1x[i]) / dx[2])
            end
            return Form(Dict((1,2) => Fun(dom2xy, dcs2xy)))
        else
            @assert false
        end
    else
        @assert false
    end
end



function coderiv2(form::Form{D,R,T,U})::Form{D,R-1,T,U} where
        {D, R, T<:Number, U}
    # (star ∘ deriv ∘ star)(form)
    if R == 1
        if D == 2
            u1x = form[(1,)]
            u1y = form[(2,)]
            r0 = codiff(u1x, 1) + codiff(u1y, 2)
            return Form(Dict(() => r0))
        else
            @assert false
        end
    else
        @assert false
    end
end

# TODO: REMOVE THIS
function coderiv1(form::Form{D,R,T,U})::Form{D,R-1,T,U} where
        {D, R, T<:Number, U}
    di = ntuple(dir -> CartesianIndex(ntuple(d -> d==dir, D)), D)
    dx = spacing(form.dom)

    if R == 1
        if D == 2
            f1x = form[(1,)]
            f1y = form[(2,)]
            cs1x = f1x.coeffs
            cs1y = f1y.coeffs
            dom0 = makeunstaggered(f1x.dom)
            n0 = dom0.n
            jmin = 0 * n0 .+ 2
            jmax = n0 .- 1
            dcs0 = Array{U}(undef, n0.elts)
            for i in CartesianIndices(size(dcs0))
                j = CartesianIndex(max.(jmin, min.(jmax, Vec(i.I))).elts)
                dcs0[i] = (+ (cs1x[j] - cs1x[j - di[1]]) / dx[1]
                           + (cs1y[j] - cs1y[j - di[2]]) / dx[2])
            end
            return Form(Dict(() => Fun(dom0, dcs0)))
        elseif D == 3
            f1x = form[(1,)]
            f1y = form[(2,)]
            f1z = form[(3,)]
            cs1x = f1x.coeffs
            cs1y = f1y.coeffs
            cs1z = f1z.coeffs
            dom0 = makeunstaggered(f1x.dom)
            n0 = dom0.n
            jmin = 0 * n0 .+ 2
            jmax = n0 .- 1
            dcs0 = Array{U}(undef, n0.elts)
            for i in CartesianIndices(size(dcs0))
                j = CartesianIndex(max.(jmin, min.(jmax, Vec(i.I))).elts)
                dcs0[i] = (+ (cs1x[j] - cs1x[j - di[1]]) / dx[1]
                           + (cs1y[j] - cs1y[j - di[2]]) / dx[2]
                           + (cs1z[j] - cs1z[j - di[3]]) / dx[3])
            end
            return Form(Dict(() => Fun(dom0, dcs0)))
        else
            @assert false
        end
    else
        @assert false
    end
end



function laplace2(form::Form{D,R,T,U})::Form{D,R,T,U} where
        {D, R, T<:Number, U}
    di = ntuple(dir -> CartesianIndex(ntuple(d -> d==dir, D)), D)
    dx = spacing(form.dom)

    if R == 0
        f0 = form[()]
        cs0 = f0.coeffs
        dom0 = f0.dom
        dcs0 = Array{U}(undef, dom0.n.elts)
        n = dom0.n
        jmin = 0 * n .+ 2
        jmax = n .- 1
        if D == 2
            for i in CartesianIndices(size(dcs0))
                j = CartesianIndex(max.(jmin, min.(jmax, Vec(i.I))).elts)
                dcs0[i] =
                    (+ (cs0[j - di[1]] - 2 * cs0[j] + cs0[j + di[1]]) / dx[1]^2
                     + (cs0[j - di[2]] - 2 * cs0[j] + cs0[j + di[2]]) / dx[2]^2)
            end
        elseif D == 3
            for i in CartesianIndices(size(dcs0))
                j = CartesianIndex(max.(jmin, min.(jmax, Vec(i.I))).elts)
                dcs0[i] =
                    (+ (cs0[j - di[1]] - 2 * cs0[j] + cs0[j + di[1]]) / dx[1]^2
                     + (cs0[j - di[2]] - 2 * cs0[j] + cs0[j + di[2]]) / dx[2]^2
                     + (cs0[j - di[3]] - 2 * cs0[j] + cs0[j + di[3]]) / dx[3]^2)
            end
        else
            @assert false
        end
        return Form(Dict(() => Fun(dom0, dcs0)))
    else
        @assert false
    end
end

function laplace(dom::Domain{D,T})::Op{D,T,T} where {D, T<:Number}
    @assert !any(dom.staggered)
    n = dom.n
    dx = spacing(dom)
    dx2 = dx .* dx

    str = Vec{D,Int}(ntuple(dir -> dir==1 ? 1 : prod(n[d] for d in 1:dir-1), D))
    len = prod(n)
    idx(i::Vec{D,Int}) = 1 + sum(i[d] * str[d] for d in 1:D)

    I = Int[]
    J = Int[]
    V = T[]
    maxsize = 3*D * len
    sizehint!(I, maxsize)
    sizehint!(J, maxsize)
    sizehint!(V, maxsize)
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
            ins!(i, j - di, 1 / dx2[dir])
            ins!(i, j, -2 / dx2[dir])
            ins!(i, j + di, 1 / dx2[dir])
        end
    end
    mat = sparse(I, J, V, len, len)
    Op{D,T,T}(dom, mat)
end



export boundary
function boundary(dom::Domain{D,T})::Op{D,T,T} where {D, T}
    n = dom.n

    str = Vec{D,Int}(ntuple(dir -> dir==1 ? 1 : prod(n[d] for d in 1:dir-1), D))
    len = prod(n)
    idx(i::Vec{D,Int}) = 1 + sum(i[d] * str[d] for d in 1:D)

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
    for ic in CartesianIndices(dom.n.elts)
        i = Vec(ic.I) .- 1
        if any(i .== 0) || any(i .== n .- 1)
            ins!(i, i, T(1))
        end
    end
    mat = sparse(I, J, V, len, len)
    Op{D,T,T}(dom, mat)
end

export dirichlet
const dirichlet = boundary



# TODO: Remove these

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

end
