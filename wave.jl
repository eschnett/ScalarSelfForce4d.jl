@generated function waveD(x::Vec{D,T})::T where {D,T}
    quote
        k = tuple($([d<D ? :(T(1)) : :(sqrt(T(D-1))) for d in 1:D]...))
        *($([:(sinpi(k[$d] * x[$d])) for d in 1:D]...))
    end
end

ldom = Domain{2,Float64}(5, lorentzian=true)

pot = zeros(Form{2,0,false,Float64,Float64}, ldom)
abvals = approximate(waveD, ldom)
bvals = Form(Dict(() => abvals))
sol = solve_dAlembert_Dirichlet(pot, bvals)

err = sol - bvals
maxerr = norm(err[()], Inf)
@test 0.037 <= maxerr < 0.038
