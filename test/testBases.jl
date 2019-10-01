using ScalarSelfForce4d.Bases



function testBases()

    @testset "Bases.weights D=$D" for D in 1:3
        dom = Domain{D,Rat}(5 + D)

        Ws = weights(dom)
        for d in 1:D
            @test isequal(sum(Ws[d]), Rat(2))
        end
    end
end

if runtests
    testBases()
end
