using ScalarSelfForce4d.Domains

using Memoize



@memoize Dict function dom(D::Int)::Domain
    Domain{D, Float64}(9)
end
@memoize Dict function sdom(D::Int)::Domain
    makestaggered(dom(D), trues(Vec{D,Bool}))
end



function testDomains()

    @testset "Domains D=$D" for D in 0:3
        @test !dom(D).dual
        @test all(~dom(D).staggered)
        @test all(dom(D).n .== 9)
        @test all(dom(D).metric .== 1)

        @test !sdom(D).dual
        @test all(sdom(D).staggered)
        @test all(sdom(D).n .== 8)
        @test spacing(sdom(D)) == spacing(dom(D))

        ddomD = makedual(dom(D), true)
        @test ddomD.dual
        @test all(ddomD.staggered)
        @test all(ddomD.n .== 9)
        @test spacing(ddomD) == spacing(dom(D))

        sddomD = makestaggered(ddomD, falses(Vec{D,Bool}))
        @test sddomD.dual
        @test all(~sddomD.staggered)
        @test all(sddomD.n .== 8)
        @test spacing(sddomD) == spacing(dom(D))

        dsdomD = makedual(sdom(D), true)
        @test dsdomD.dual
        @test all(~dsdomD.staggered)
        @test all(dsdomD.n .== 8)
        @test spacing(dsdomD) == spacing(dom(D))

        @test sddomD == dsdomD
        @test makeunstaggered(sdom(D)) == dom(D)
        @test makeprimal(ddomD) == dom(D)
        @test (makeprimal(makestaggered(sddomD, trues(Vec{D,Bool}))) ==
               makeunstaggered(makeprimal(sddomD)))
    end

end

@warn "testDomains()"
