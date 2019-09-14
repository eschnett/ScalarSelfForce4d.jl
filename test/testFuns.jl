using ScalarSelfForce4d.Funs



function sinpiD(x::Vec{D,T})::T where {D,T}
    prod(sinpi(x[d]) for d in 1:D)
end

function find_maxerr(f1, f2)
    maxerr = 0.0
    for z in LinRange(-1.0, 1.0, 11),
        y in LinRange(-1.0, 1.0, 11),
        x in LinRange(-1.0, 1.0, 11)
        v1 = f1(Vec((x, y, z)))
        v2 = f2(Vec((x, y, z)))
        dv = v1 - v2
        maxerr = max(maxerr, abs(dv))
    end
    maxerr
end



function testFuns()
    @testset "Funs.Approximation" begin
        atol = 100 * eps(1.0)
    
        ac = approximate(x->1.0, dom3)
        maxerr = find_maxerr(x->1.0, ac)
        @test isapprox(maxerr, 0; atol=atol)
    
        ax = approximate(x->x[1], dom3)
        maxerr = find_maxerr(x->x[1], ax)
        @test isapprox(maxerr, 0; atol=atol)
    
        axy = approximate(x->x[1]*x[2], dom3)
        maxerr = find_maxerr(x->x[1]*x[2], axy)
        @test isapprox(maxerr, 0; atol=atol)
    
        axyz = approximate(x->x[1]*x[2]*x[3], dom3)
        maxerr = find_maxerr(x->x[1]*x[2]*x[3], axyz)
        @test isapprox(maxerr, 0; atol=atol)
    
        acs = approximate(x->1.0, sdom3)
        maxerr = find_maxerr(x->1.0, acs)
        @test isapprox(maxerr, 0; atol=atol)
    
        global asinpi = approximate(sinpiD, dom3)
        maxerr = find_maxerr(sinpiD, asinpi)
        @test 0.059 <= maxerr < 0.060
    end
end

testFuns()
