using ScalarSelfForce4d.Domains



function testDomains()

    @testset "Domains" begin
        global dom3 = Domain{3,Float64}(9)
        global sdom3 = makestaggered(dom3, Vec((true, true, true)))
        @test all(!dom3.staggered)
        @test all(dom3.n .== 9)
        @test all(sdom3.staggered)
        @test all(sdom3.n .== 8)
    end

end

testDomains()
