"""
Differential forms
"""
module Forms

using SparseArrays

using ..Domains
using ..Funs
using ..Ops
using ..Vecs



export Form
struct Form{D, R, T, U}
    dom::Domain{D, T}
    comps::Dict{Vec{R,Int}, Fun{D,T,U}}

    function Form(dict::Dict{Vec{R,Int}, Fun{D,T,U}}) where {D, R, T, U}
        @assert D isa Int && 0 <= D
        @assert R isa Int && 0 <= R <= D

        i0 = Vec{R,Int}(ntuple(d -> d, R))
        f0 = dict[i0]
        dom = makeunstaggered(f0.dom)

        count = 0
        for i in CartesianIndices(ntuple(d -> D, R))
            !all(i[d] > i[d+1] for d in 1:R-1) && continue
            count += 1
            f = dict[Vec{R,Int}(i.I)]
            @assert all(f.dom.staggered[d] == (d in i.I) for d in 1:D)
            @assert all(f.dom.n + f.dom.staggered .== dom.n)
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



export deriv
function deriv(form::Form{D,R,T,U})::Form{D,R+1,T,U} where {D, R, T<:Number, U}
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
    else
        @assert false
    end
end



export coderiv
function coderiv(form::Form{D,R,T,U})::Form{D,R-1,T,U} where
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



export laplace
function laplace(form::Form{D,R,T,U})::Form{D,R,T,U} where
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
