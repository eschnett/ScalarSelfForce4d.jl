using ScalarSelfForce4d.Vecs



function testVecs()

    @testset "Vecs" begin
    end

end

testVecs()

xs = [Vec{1,Int}((rand(-100:100),)) for i in 1:100]
testVectorspace(zeros(Vec{1,Int}), xs, isequal)
