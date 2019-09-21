module ScalarSelfForce4d

using LinearAlgebra
using Reexport
using SparseArrays



# The order of include statements matters
include("Defs.jl")
include("Quadrature.jl")
include("Vecs.jl")
include("Domains.jl")
include("Bases.jl")
include("Funs.jl")
include("Ops.jl")
include("Forms.jl")

@reexport using .Bases
@reexport using .Defs
@reexport using .Domains
@reexport using .Forms
@reexport using .Funs
@reexport using .Ops
@reexport using .Quadrature
@reexport using .Vecs



################################################################################



export boundaryIV
function boundaryIV(::Type{U}, dom::Domain{D,T})::Op{D,T,U} where
        {D,T <: Number,U <: Number}
    @assert !any(dom.staggered) # TODO
    n = dom.n

    str = Vec{D,Int}(ntuple(dir->dir == 1 ? 1 : prod(n[d] for d in 1:dir - 1), D))
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
        {D,T <: Number,U <: Number}
    @assert !any(dom.staggered) # TODO
    n = dom.n
    dx2 = Vec(ntuple(d->((dom.xmax[d] - dom.xmin[d]) / (n[d] - 1))^2, D))

    str = Vec{D,Int}(ntuple(dir->dir == 1 ? 1 : prod(n[d] for d in 1:dir - 1), D))
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
            di = Vec(ntuple(d->d == dir ? 1 : 0, D))
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



################################################################################

# Discrete differential forms

# Derivative of a 0-form
function deriv0(u0::Fun{D,T,U})::NTuple{D,Fun{D,T,U}} where {D,T,U}
    if D == 2
        dom0 = u0.dom
        s0 = dom0.staggered
        n0 = dom0.n
        di = ntuple(dir->CartesianIndex(ntuple(d->d == dir, D)), D)
        dx = ntuple(d->(dom0.xmax[d] - dom0.xmin[d]) / (n0[d] - 1), D)
        @assert s0 == Vec((false, false))
        s1x = Vec((true, false))
        s1t = Vec((false, true))
        n1x = Vec((n0[1] - s1x[1], n0[2] - s1x[2]))
        n1t = Vec((n0[1] - s1t[1], n0[2] - s1t[2]))
        dom1x = Domain{D,T}(s1x, dom0.metric, n1x, dom0.xmin, dom0.xmax)
        dom1t = Domain{D,T}(s1t, dom0.metric, n1t, dom0.xmin, dom0.xmax)
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

# Wedge of two 1-forms
function wedge11(u1::NTuple{D,Fun{D,T,U}},
                 v1::NTuple{D,Fun{D,T,U}})::Fun{D,T,U} where {D,T,U}
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
        di = ntuple(dir->CartesianIndex(ntuple(d->d == dir, D)), D)
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
                          + ucs1t[i + di[1]] * vcs1x[i]
                          + ucs1t[i + di[1]] * vcs1x[i + di[2]]
                          + ucs1t[i] * vcs1x[i + di[2]])
                       - (+ ucs1x[i] * vcs1t[i]
                          + ucs1x[i + di[2]] * vcs1t[i]
                          + ucs1x[i + di[2]] * vcs1t[i + di[1]]
                          + ucs1x[i] * vcs1t[i + di[1]])) / 8
        end
        return Fun{D,T,U}(dom2, wcs2)
    else
        @assert false
    end
end



################################################################################

# Scalar wave equation

export scalarwave_energy
function scalarwave_energy(phi::Fun{D,T,T})::Fun{D,T,T} where {D,T <: Number}
    @assert all(!phi.dom.staggered)

    dphi = deriv0(phi)
    sdphi = star1(dphi)
    eps = wedge11(dphi, sdphi)

    eps
end

function scalarwave_energy1(phi::Fun{D,T,T})::Fun{D,T,T} where {D,T <: Number}
    @assert all(!phi.dom.staggered)
    sdom = makestaggered(phi.dom)

    n = sdom.n
    dx = Vec(ntuple(d->(sdom.xmax[d] - sdom.xmin[d]) / n[d], D))
    di = ntuple(dir->Vec(ntuple(d->Int(d == dir), D)), D)

    eps = Array{T}(undef, n.elts)
    if D == 4
        for ic in CartesianIndices(size(eps))
            i = Vec(ic.I)
            s = T(0)
            # x
            for c in 0:1, b in 0:1, a in 0:1
                im = i +         a * di[2] + b * di[3] + c * di[4];
                ip = i + di[1] + a * di[2] + b * di[3] + c * di[4];
                s += ((+ phi.coeffs[CartesianIndex(ip.elts)]
                       - phi.coeffs[CartesianIndex(im.elts)]) / dx[1])^2 / 8
            end
            # y
            for c in 0:1, b in 0:1, a in 0:1
                im = i +         a * di[1] + b * di[3] + c * di[4];
                ip = i + di[2] + a * di[1] + b * di[3] + c * di[4];
                s += ((+ phi.coeffs[CartesianIndex(ip.elts)]
                       - phi.coeffs[CartesianIndex(im.elts)]) / dx[2])^2 / 8
            end
            # z
            for c in 0:1, b in 0:1, a in 0:1
                im = i +         a * di[1] + b * di[2] + c * di[4];
                ip = i + di[3] + a * di[1] + b * di[2] + c * di[4];
                s += ((+ phi.coeffs[CartesianIndex(ip.elts)]
                       - phi.coeffs[CartesianIndex(im.elts)]) / dx[3])^2 / 8
            end
            # t
            for c in 0:1, b in 0:1, a in 0:1
                im = i +         a * di[1] + b * di[2] + c * di[3];
                ip = i + di[4] + a * di[1] + b * di[2] + c * di[3];
                s += ((+ phi.coeffs[CartesianIndex(ip.elts)]
                       - phi.coeffs[CartesianIndex(im.elts)]) / dx[4])^2 / 8
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

    grad_pot = ntuple(d->deriv(pot, d), D)

    acc = ntuple(d->sum(rho .* grad_pot[d]), D)
    Vec{D,T}(acc)
end

end
