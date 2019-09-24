using Arpack
using DataStructures
using Gadfly
using LinearAlgebra
using ScalarSelfForce4d



dom = Domain{1,Float64}(9)
lap = laplace(Val(0), Val(false), dom)
dir = dirichlet(Val(0), Val(false), dom)
bnd = boundary(Val(0), Val(false), dom)
op = mix_op_bc(bnd, lap, -dir, dom)

lapc = op.comps[Vec{0,Int}(())][Vec{0,Int}(())]
lambda, v = eigen(Matrix(lapc))

# Count boundary eigenvalues
@assert count(x -> abs(x+1)<1.0e-12, lambda) == 9^1 - (9-2)^1
# Normalize eigenvalues
lambda2 = Array{Float64}(undef, length(lambda))
for i in 1:length(lambda2)
    if i>1 && abs(lambda[i] - lambda2[i-1]) < 1.0e-12
        lambda2[i] = lambda2[i-1]
    else
        lambda2[i] = lambda[i]
    end
end
# Determine second non-boundary eigenvalue with multiplicity 1
ms = counter(lambda2)
(l,m) = collect(sort(filter((l,m)->!(abs(l+1)<1.0e-12 || m>1), ms), rev=true))[2]
n = findfirst(==(l), lambda)
@assert n !== nothing

lambda
spy(v')



dom = Domain{2,Float64}(9)
lap = laplace(Val(0), Val(false), dom)
dir = dirichlet(Val(0), Val(false), dom)
bnd = boundary(Val(0), Val(false), dom)
op = mix_op_bc(bnd, lap, -dir, dom)

lapc = op.comps[Vec{0,Int}(())][Vec{0,Int}(())]
lambda, v = eigs(lapc; nev=40, which=:LR)
# lambda, v = eigen(Matrix(lapc))

lambda

# Count boundary eigenvalues
@assert count(x -> abs(x+1)<1.0e-12, lambda) == 9^2 - (9-2)^2
# Normalize eigenvalues
lambda2 = Array{Float64}(undef, length(lambda))
for i in 1:length(lambda2)
    if i>1 && abs(lambda[i] - lambda2[i-1]) < 1.0e-12
        lambda2[i] = lambda2[i-1]
    else
        lambda2[i] = lambda[i]
    end
end
# Determine second non-boundary eigenvalue with multiplicity 1
ms = counter(lambda2)
(l,m) = collect(sort(filter((l,m)->!(abs(l+1)<1.0e-12 || m>1), ms), rev=true))[2]
n = findfirst(==(l), lambda)
@assert n !== nothing

spy(reshape(real.(v[:,n]), (9,9)))



dom = Domain{3,Float64}(9)
lap = laplace(Val(0), Val(false), dom)
dir = dirichlet(Val(0), Val(false), dom)
bnd = boundary(Val(0), Val(false), dom)
op = mix_op_bc(bnd, lap, -dir, dom)

lapc = op.comps[Vec{0,Int}(())][Vec{0,Int}(())]
lambda, v = eigs(lapc; nev=420, which=:LR)
# lambda, v = eigen(Matrix(lapc))

# collect(filter(nx->nx[2]<0, collect(enumerate(lambda))))

# Count boundary eigenvalues
@assert count(x -> abs(x+1)<1.0e-12, lambda) == 9^3 - (9-2)^3
# Normalize eigenvalues
lambda2 = Array{Float64}(undef, length(lambda))
for i in 1:length(lambda2)
    if i>1 && abs(lambda[i] - lambda2[i-1]) < 1.0e-12
        lambda2[i] = lambda2[i-1]
    else
        lambda2[i] = lambda[i]
    end
end
# Determine second non-boundary eigenvalue with multiplicity 1
ms = counter(lambda2)
(l,m) = collect(sort(filter((l,m)->!(abs(l+1)<1.0e-12 || m>1), ms), rev=true))[2]
n = findfirst(==(l), lambda)
@assert n !== nothing

# second eigenvalue with multiplicity 1
spy(reshape(real.(v[:,n]), (9,9,9))[:,:,3])
