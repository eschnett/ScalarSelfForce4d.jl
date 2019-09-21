using LinearAlgebra
using ScalarSelfForce4d

@generated function waveD(x::Vec{D,T})::T where {D,T}
    quote
        $(Expr(:meta, :inline))
        k = tuple($([d<D ? :(T(1)) : :(sqrt(T(D-1))) for d in 1:D]...))
        *($([:(sinpi(k[$d] * x[$d])) for d in 1:D]...))
    end
end

D = 4
n = 5
dom3 = Domain{D-1,Float64}(n)
ldom4 = Domain{D,Float64}(n, lorentzian=true)

# Source
lap3 = laplace(Val(0), Val(false), dom3)
adel3 = 2pi * approximate_delta(dom3, zeros(Vec{D-1,Float64}))
del3 = Form(Dict(() => adel3))
dir3 = dirichlet(Val(0), Val(false), dom3)
bvals3 = zeros(typeof(del3), dom3)

bnd3 = boundary(Val(0), Val(false), dom3)
op3 = mix_op_bc(bnd3, lap3, dir3, dom3)
rhs3 = mix_op_bc(bnd3, del3, bvals3)
pot3 = op3 \ rhs3

# Potential (source)
asrc = zeros(Fun{D,Float64,Float64}, ldom4)
if D==2
    for i4 in 1:ldom4.n[D]
        # asrc.coeffs[:,i4] = del3[()].coeffs[:]
        asrc.coeffs[:,i4] = rhs3[()].coeffs[:]
    end
elseif D==3
    for i4 in 1:ldom4.n[D]
        # asrc.coeffs[:,:,i4] = del3[()].coeffs[:,:]
        asrc.coeffs[:,:,i4] = rhs3[()].coeffs[:,:]
    end
elseif D==4
    for i4 in 1:ldom4.n[D]
        # asrc.coeffs[:,:,:,i4] = del3[()].coeffs[:,:,:]
        asrc.coeffs[:,:,:,i4] = rhs3[()].coeffs[:,:,:]
    end
end
src = Form(Dict(() => asrc))
# Initial and boundary conditions
abvals = zeros(Fun{D,Float64,Float64}, ldom4)
if D==2
    for i4 in 1:ldom4.n[D]
        abvals.coeffs[:,i4] = pot3[()].coeffs[:]
    end
elseif D==3
    for i4 in 1:ldom4.n[D]
        abvals.coeffs[:,:,i4] = pot3[()].coeffs[:,:]
    end
elseif D==4
    for i4 in 1:ldom4.n[D]
        abvals.coeffs[:,:,:,i4] = pot3[()].coeffs[:,:,:]
    end
end
bvals = Form(Dict(() => abvals))

sol = solve_dAlembert_Dirichlet(src, bvals)

err = sol - bvals
maxerr = norm(err[()], Inf)
@test maxerr < 1.0e-12
