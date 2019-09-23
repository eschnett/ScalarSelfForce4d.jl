using Gadfly
using LinearAlgebra
using ScalarSelfForce4d

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

dom = Domain{2,Float64}(65, lorentzian=true)

phi = Form(Dict(() => approximate(waveD, dom)))
plot((x,t) -> phi[()](Vec((x,t))), -1, 1, 0, 1)

dphi = deriv(phi)
sdphi = star(dphi)
# Note: The action has a sign error because the metric signature is
# not yet handled properly
act = wedge(dphi, sdphi) / 2.0
plot((x,t) -> act[(1,2)](Vec((x,t))), -1, 1, 0, 1)

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
plot((x,t) -> ene[(1,2)](Vec((x,t))), -1, 1, 0, 1)
# plot(x -> ene[(1,2)](Vec((x,0.5))), -1, 1)

# mom = phit
# ham = wedge(mom, star(mom)) - act
# plot((x,t) -> ham[(1,2)](Vec((x,t))), -1, 1, 0, 1)
