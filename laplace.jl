dom = Domain{2,Float64}(3)

lap = laplace(Val(0), Val(false), dom)
lap1 = laplace1(Val(0), Val(false), dom)
adel = 2pi * approximate_delta(dom, zeros(Vec{2,Float64}))
del = Form(Dict(() => adel))
dir = dirichlet(Val(0), Val(false), dom)
bvals = zeros(typeof(del), dom)
bnd = boundary(Val(0), Val(false), dom)
op = mix_op_bc(bnd, lap, dir, dom)
op1 = mix_op_bc(bnd, lap1, dir, dom)
rhs = mix_op_bc(bnd, del, bvals)
pot = op \ rhs
pot1 = op1 \ rhs
res = op * pot - rhs
maxres = norm(res[()], Inf)
