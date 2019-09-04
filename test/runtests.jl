using ScalarSelf4d
using Test

const par = Par{3,Float64}(9)



const fsinpi = project_basis(sinpi, Float64, par, 1)

function find_maxerr1()
    maxerr = 0.0
    for x in LinRange(-1.0, 1.0, 11)
        f0 = sinpi(x)
        f = value(par, 1, fsinpi, x)
        df = f - f0
        maxerr = max(maxerr, abs(df))
    end
    maxerr
end
@test find_maxerr1() < 0.12



function sinpi2(x::Vec{D,T})::T where {D,T}
    prod(sinpi(x[d]) for d in 1:D)
end

const fsinpi2 = project_basis(sinpi2, Float64, par)

function find_maxerr()
    maxerr = 0.0
    for z in LinRange(-1.0, 1.0, 11),
        y in LinRange(-1.0, 1.0, 11),
        x in LinRange(-1.0, 1.0, 11)
        f0 = sinpi(x) * sinpi(y) * sinpi(z)
        f = value(par, fsinpi2, Vec((x, y, z)))
        df = f - f0
        maxerr = max(maxerr, abs(df))
    end
    maxerr
end
@test find_maxerr() < 0.12
