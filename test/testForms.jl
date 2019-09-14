using ScalarSelfForce4d.Forms

using LinearAlgebra



function dsinpiD(x::Vec{D,T}, dir::Int)::T where {D,T}
    prod(d == dir ? pi * cospi(x[d]) : sinpi(x[d]) for d in 1:D)
end

function d2sinpiD(x::Vec{D,T}, dir::Int)::T where {D,T}
    prod(d == dir ? - pi^2 * sinpi(x[d]) : sinpi(x[d]) for d in 1:D)
end



function testForms()

    @testset "Forms.Star" begin
        atol = 100 * eps(1.0)

        dom2 = Domain{2,Float64}(9)

        asinpi2 = approximate(sinpiD, dom2)
        fsinpi2 = Form(Dict(() => asinpi2))

        sfsinpi2 = star(fsinpi2)
        ssfsinpi2 = star(sfsinpi2)
        sssfsinpi2 = star(ssfsinpi2)
        ssssfsinpi2 = star(sssfsinpi2)
        maxerr = norm(sfsinpi2[(1,2)] - sssfsinpi2[(1,2)], Inf)
        @test isapprox(maxerr, 0; atol=atol)
        maxerr = norm(ssfsinpi2[()] - ssssfsinpi2[()], Inf)
        @test isapprox(maxerr, 0; atol=atol)

        dfsinpi2 = deriv(fsinpi2)
        sdfsinpi2 = star(dfsinpi2)
        dsdfsinpi2 = deriv(sdfsinpi2)
        sdsdfsinpi2 = star(dsdfsinpi2)
        cdfsinpi2 = coderiv(dfsinpi2)
        maxerr = norm(cdfsinpi2[()] - sdsdfsinpi2[()], Inf)
        @test isapprox(maxerr, 0; atol=atol)

        lfsinpi2 = laplace(fsinpi2)
        maxerr = norm(lfsinpi2[()] - cdfsinpi2[()], Inf)
        @test isapprox(maxerr, 0; atol=atol)
    end

    @testset "Forms.Derivatives" begin
        global fsinpi = Form(Dict(() => asinpi))
        global dfsinpi = deriv(fsinpi)
        for dir in 1:3
            dom3x = makestaggered(dom3, unitvec(Val(3), dir))
            adsinpix = approximate(x -> dsinpiD(x, dir), dom3x)
            maxdiffx = norm(dfsinpi.comps[Vec((dir,))] - adsinpix, Inf)
            @test 0.16 <= maxdiffx < 0.17
        end
    end

    @testset "Forms.Laplacian" begin
        atol = 100 * eps(1.0)

        global lfsinpi = laplace(fsinpi)

        cdfsinpi = coderiv(dfsinpi)
        maxerr = norm(lfsinpi[()] - cdfsinpi[()], Inf)
        @test isapprox(maxerr, 0; atol=atol)

        # sdfsinpi = star(dfsinpi)
        # dsdfsinpi = deriv(sdfsinpi)
        # sdsdfsinpi = star(dsdfsinpi)
        # maxerr = norm(lfsinpi[()] - sdsdfsinpi[()], Inf)
        # @test isapprox(maxerr, 0; atol=atol)

        ad2sinpi = approximate(x -> sum(d2sinpiD(x, d) for d in 1:3), dom3)
        maxerr = norm(lfsinpi[()] - ad2sinpi, Inf)
        @test 22.0 <= maxerr <= 23.0
        # Test without outer boundary
        maxerr = norm((lfsinpi[()] - ad2sinpi).coeffs[2:end-1,2:end-1,2:end-1],
                      Inf)
        @test 1.5 <= maxerr < 1.6
    end

end

testForms()
