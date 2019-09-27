using ScalarSelfForce4d.Vecs



function testVecs()

    BigRat = Rational{BigInt}
    bigrange = -10 : big(1//10) : 10
    for D in 1:2
        as = [rand(bigrange) for i in 1:100]
        xs = [Vec{D,BigRat}(ntuple(d->rand(bigrange), D)) for i in 1:100]
        testVectorspace(zeros(Vec{D,BigRat}), as, xs, isequal)
    end

    @testset "Vecs" begin
    end

end

if runtests
    testVecs()
end
