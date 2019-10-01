using ScalarSelfForce4d.Vecs



function testVecs()

    for D in 1:2
        as = [rand(ratrange) for i in 1:100]
        xs = [Vec{D,Rat}(ntuple(d->rand(ratrange), D)) for i in 1:100]
        testVectorspace(zeros(Vec{D,Rat}), as, xs, isequal)
    end

    @testset "Vecs" begin
    end

end

if runtests
    testVecs()
end
