using ScalarSelfForce4d
using Gadfly
using LinearAlgebra

@generated function waveD(x::Vec{D,T})::T where {D,T}
    quote
        k = tuple($([d < D ? :(T(1)) : :(sqrt(T(D - 1))) for d in 1:D]...))
        *($([:(sinpi(k[$d] * x[$d])) for d in 1:D]...))
    end
end

@generated function epsD(x::Vec{D,T})::T where {D,T}
    quote
        k = tuple($([d < D ? :(T(1)) : :(sqrt(T(D - 1))) for d in 1:D]...))
        dphi = tuple($([
            :(*($([d == dir ?
                   :(pi * k[$d] * cospi(k[$d] * x[$d])) :
                   :(sinpi(k[$d] * x[$d])) for d in 1:D]...)))
            for dir in 1:D]...))
        eps = +($([:(dphi[$d]^2) for d in 1:D]...)) / 2
        eps
    end
end
function etotD(dom::Domain{D,T}, t::T)::T where {D,T}
    T(pi)^2 * D/2
end



ldom4 = Domain{2,Float64}(9, lorentzian=true)

pot = zeros(Form{2,0,false,Float64,Float64}, ldom4)
# abvals = approximate(waveD, ldom4)
abvals = Fun{2,Float64,Float64}(ldom4, [sinpi(x)*sinpi(t) for x in LinRange(-1,1,9), t in LinRange(0,1,9)])
bvals = Form(Dict(() => abvals))
sol = solve_dAlembert_Dirichlet(pot, bvals)

err = sol - bvals
maxerr = norm(err[()], Inf)

phi = sol
dphi = deriv(phi)
act = -wedge(dphi, star(dphi)) / 2.0

et = Form(Dict(() => sample(xt -> xt[end], ldom4)))
nt = deriv(et)
mom = wedge(dphi, star(nt))
ham = - wedge(mom, star(mom)) - act
sldom4 = makestaggered(ldom4, trues(Vec{2,Bool}))
eham = Form(Dict((1,2) => approximate(epsD, sldom4)))
maxerr = norm((ham - eham)[(1,2)], Inf)
# @assert isapprox(maxerr, 0.0027673958110688065; atol=1.0e-6)

# sdom3 = makestaggered(dom3, Vec(ntuple(d -> true, 3)))
# etot = Array{Float64}(undef, size(ham[(1,2,3,4)].coeffs, 4))
# dx4 = spacing(ldom4)[4]
# for i in 1:length(etot)
#     cs = ham[(1,2,3,4)].coeffs[:,:,:,i] ./ dx4
#     ene = Form(Dict((1,2,3) => Fun{3,Float64,Float64}(sdom3, cs)))
#     etot[i] = sum(ene[(1,2,3)])
# end
# etot

calculate energy in 3d, instead of in 4d and then projecting
