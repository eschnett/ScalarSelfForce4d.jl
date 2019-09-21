using ScalarSelfForce4d.Vecs



function testVecs()

    BigRat = Rational{BigInt}
    for D in 1:2
        as = [BigRat(rand(-100:100)) for i in 1:100]
        xs = [Vec{D,BigRat}(ntuple(d->rand(-100:100), D)) for i in 1:100]
        testVectorspace(zeros(Vec{D,BigRat}), as, xs, isequal)
    end

    @testset "Vecs" begin
    end

end

testVecs()
