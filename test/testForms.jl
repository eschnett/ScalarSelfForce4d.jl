using ScalarSelfForce4d.Forms

using LinearAlgebra



function dsinpiD(x::Vec{D,T}, dir::Int)::T where {D,T}
    prod(d == dir ? pi * cospi(x[d]) : sinpi(x[d]) for d in 1:D)
end

function d2sinpiD(x::Vec{D,T}, dir::Int)::T where {D,T}
    prod(d == dir ? - pi^2 * sinpi(x[d]) : sinpi(x[d]) for d in 1:D)
end



const fsinpi = Dict{Int, Form}()
const dfsinpi = Dict{Int, Form}()



function testForms()

    @testset "Forms.Derivatives D=$D" for D in 1:3
        fsinpi[D] = Form(Dict(() => asinpi[D]))
        dfsinpi[D] = deriv(fsinpi[D])
        for dir in 1:D
            domDx = makestaggered(dom[D], unitvec(Val(D), dir))
            adsinpix = approximate(x -> dsinpiD(x, dir), domDx)
            maxdiffx = norm(dfsinpi[D].comps[Vec((dir,))] - adsinpix, Inf)
            if D == 1
                @test 0.39 <= maxdiffx < 0.40
            elseif D == 2
                @test 0.41 <= maxdiffx < 0.42
            elseif D == 3
                @test 0.43 <= maxdiffx < 0.44
            else
                @assert false
            end
        end
    end

    @testset "Forms.Star D=$D" for D in 1:3
        atol = 100 * eps(1.0)

        sfsinpiD = star(fsinpi[D])
        ssfsinpiD = star(sfsinpiD)
        sssfsinpiD = star(ssfsinpiD)
        scale = bitsign(0 * (D - 0))
        maxerr = norm(fsinpi[D][()] - scale * ssfsinpiD[()], Inf)
        @test isapprox(maxerr, 0; atol=atol)
        sidx = ntuple(d -> d, D)
        maxerr = norm(sfsinpiD[sidx] - scale * sssfsinpiD[sidx], Inf)
        @test isapprox(maxerr, 0; atol=atol)

        sdfsinpiD = star(dfsinpi[D])
        ssdfsinpiD = star(sdfsinpiD)
        sssdfsinpiD = star(ssdfsinpiD)
        scale = bitsign(1 * (D - 1))
        for dir in 1:D
            maxerr = norm(dfsinpi[D][(dir,)] - scale * ssdfsinpiD[(dir,)], Inf)
            @test isapprox(maxerr, 0; atol=atol)
            sidx = ntuple(d -> d < dir ? d : d + 1, D-1)
            maxerr = norm(sdfsinpiD[sidx] - scale * sssdfsinpiD[sidx], Inf)
            @test isapprox(maxerr, 0; atol=atol)
        end
    end

    @testset "Forms.Laplacian D=$D" for D in 1:3
        atol = 100 * eps(1.0)

        sdfsinpiD = star(dfsinpi[D])
        dsdfsinpiD = deriv(sdfsinpiD)
        sdsdfsinpiD = star(dsdfsinpiD)
        ad2sinpiD = approximate(x -> sum(d2sinpiD(x, d) for d in 1:D), dom[D])
        maxerr = norm(sdsdfsinpiD[()] - ad2sinpiD, Inf)
        if D == 1
            @test 0.60 <= maxerr < 0.61
        elseif D == 2
            @test 11.0 <= maxerr < 12.0
        elseif D == 3
            @test 23.0 <= maxerr < 24.0
        else
            @assert false
        end

        cdfsinpiD = coderiv(dfsinpi[D])
        maxerr = norm(cdfsinpiD[()] - sdsdfsinpiD[()], Inf)
        @test isapprox(maxerr, 0; atol=atol)

        lfsinpiD = laplace(fsinpi[D])
        maxerr = norm(lfsinpiD[()] - sdsdfsinpiD[()], Inf)
        @test isapprox(maxerr, 0; atol=atol)
    end

end

testForms()
