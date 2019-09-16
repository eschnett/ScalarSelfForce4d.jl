using ScalarSelfForce4d.Domains



const dom = Dict{Int, Domain}()
const sdom = Dict{Int, Domain}()



function testDomains()

    @testset "Domains D=$D" for D in 0:3
        dom[D] = Domain{D,Float64}(9)
        @test !dom[D].dual
        @test all(~dom[D].staggered)
        @test all(dom[D].n .== 9)
        @test all(dom[D].metric .== 1)

        sdom[D] = makestaggered(dom[D], trues(Vec{D,Bool}))
        @test !sdom[D].dual
        @test all(sdom[D].staggered)
        @test all(sdom[D].n .== 8)
        @test spacing(sdom[D]) == spacing(dom[D])

        ddomD = makedual(dom[D], true)
        @test ddomD.dual
        @test all(ddomD.staggered)
        @test all(ddomD.n .== 9)
        @test spacing(ddomD) == spacing(dom[D])

        sddomD = makestaggered(ddomD, falses(Vec{D,Bool}))
        @test sddomD.dual
        @test all(~sddomD.staggered)
        @test all(sddomD.n .== 8)
        @test spacing(sddomD) == spacing(dom[D])

        dsdomD = makedual(sdom[D], true)
        @test dsdomD.dual
        @test all(~dsdomD.staggered)
        @test all(dsdomD.n .== 8)
        @test spacing(dsdomD) == spacing(dom[D])

        @test sddomD == dsdomD
        @test makeunstaggered(sdom[D]) == dom[D]
        @test makeprimal(ddomD) == dom[D]
        @test (makeprimal(makestaggered(sddomD, trues(Vec{D,Bool}))) ==
               makeunstaggered(makeprimal(sddomD)))
    end

end

testDomains()
