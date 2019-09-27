using ScalarSelfForce4d
using Gadfly
using LinearAlgebra

@generated function waveD(x::Vec{D,T})::T where {D,T}
    quote
        k = tuple($([d < D ? :(T(1)) : :(sqrt(T(D - 1))) for d in 1:D]...))
        *($([:(sinpi(k[$d] * x[$d])) for d in 1:D]...))
    end
end



ldom4 = Domain{2,Float64}(5, lorentzian=true)

pot = zeros(Form{2,0,false,Float64,Float64}, ldom4)
abvals = sample(waveD, ldom4)
bvals = Form(Dict(() => abvals))
sol = solve_dAlembert_Dirichlet(pot, bvals)

err = sol - bvals
maxerr = norm(err[()], Inf)

phi = sol
dphi = deriv(phi)
sdphi = star(dphi)
act = - wedge(dphi, sdphi) / 2.0



dom3 = Domain{1,Float64}(ldom4.n[1])

j = 1
# j = 5
phi3 = Form(Dict(() => Fun(dom3, phi[()].coeffs[:,j])))
phit3 = Form(Dict(() => Fun(dom3, dphi[(2,)].coeffs[:,j])))
dphi3 = deriv(phi3)

wedge(phi, star(dphi))
