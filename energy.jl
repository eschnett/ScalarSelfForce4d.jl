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

dom = Domain{2,Float64}(129, lorentzian=true)

phi = Form(Dict(() => approximate(waveD, dom)))
plot((x,t) -> phi[()](Vec((x,t))), -1, 1, 0, 1)

dphi = deriv(phi)
sdphi = star(dphi)
ene = wedge(dphi, sdphi)
plot((x,t) -> ene[(1,2)](Vec((x,t))), -1, 1, 0, 1)

# how to check energy conservation?
# can we look for a conserved current?
# should we project to t=const hypersurfaces and integrate?
# what is the momentum?
