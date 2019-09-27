using Gadfly
using LinearAlgebra
using ScalarSelfForce4d

dom = Domain{1,Float64}(9)
lap = laplace(Val(0), Val(false), dom)
dir = dirichlet(Val(0), Val(false), dom)
bnd = boundary(Val(0), Val(false), dom)
op = mix_op_bc(bnd, lap, -dir, dom)
opc = op.comps[Vec{0,Int}(())][Vec{0,Int}(())]
lambda, v = eigen(Matrix(opc))
lambda
spy(v')
n = 6

# ldom = Domain{2,Float64}(9, lorentzian=true)
ldom = Domain{2,Float64}(false, Vec((false, false)), Vec((9, 9)),
                         Vec((1, -1)), Vec((-1,0)), Vec((1,2)))

phic = [v[i,n] * v[j,n] for i in 1:9, j in 1:9]
phi = Form(Dict(() => Fun{2,Float64,Float64}(ldom, phic)))
spy(phi[()].coeffs)

TODO: check solution; need d'Alembertian

make hodge star take signature into account for only dual->primal?



dphi = deriv(phi)
sdphi = star(dphi)
# Note: The action has a sign error because the metric signature is
# not yet handled properly
act = wedge(dphi, sdphi) / BigRat(2)
# plot((x,t) -> act[(1,2)](Vec((x,t))), -1, 1, 0, 1)
spy(act[(1,2)].coeffs)

ex = Form(Dict(() => approximate(xt -> xt[1], dom)))
et = Form(Dict(() => approximate(xt -> xt[end], dom)))
nx = deriv(ex)
nt = deriv(et)

phit = wedge(dphi, star(nt))
phix = wedge(dphi, star(nx))

phix2 = wedge(phix, star(phix))
phit2 = wedge(phit, star(phit))
ene = (phix2 + phit2) / 2.0
# hstack(
#     plot((x,t) -> phix2[(1,2)](Vec((x,t))), -1, 1, 0, 1),
#     plot((x,t) -> phit2[(1,2)](Vec((x,t))), -1, 1, 0, 1),
# )
# plot((x,t) -> ene[(1,2)](Vec((x,t))), -1, 1, 0, 1)
spy(ene[(1,2)].coeffs)
# plot(x -> ene[(1,2)](Vec((x,0.5))), -1, 1)

# mom = phit
# ham = wedge(mom, star(mom)) - act
# plot((x,t) -> ham[(1,2)](Vec((x,t))), -1, 1, 0, 1)
