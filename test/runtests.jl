using ScalarSelf4d
using Test

const par = Par{3,Float64}(9)



function sinpiD(x::Vec{D,T})::T where {D,T}
    prod(sinpi(x[d]) for d in 1:D)
end

const fsinpi = approximate(sinpiD, Float64, par)

function find_maxerr()
    maxerr = 0.0
    for z in LinRange(-1.0, 1.0, 11),
        y in LinRange(-1.0, 1.0, 11),
        x in LinRange(-1.0, 1.0, 11)
        f0 = sinpiD(Vec((x, y, z)))
        f = fsinpi(Vec((x, y, z)))
        df = f - f0
        maxerr = max(maxerr, abs(df))
    end
    maxerr
end
@test find_maxerr() < 0.12
