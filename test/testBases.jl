using ScalarSelfForce4d.Bases



function testBases()

    @testset "Bases.weights D=$D" for D in 1:3
        BigRat = Rational{BigInt}
        dom = Domain{D,BigRat}(5 + D)

        Ws = weights(dom)
        for d in 1:D
            @test isequal(sum(Ws[d]), BigRat(2))
        end
    end
end

@warn "testBases()"
