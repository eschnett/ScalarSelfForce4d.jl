using ScalarSelfForce4d.Vecs



function testVecs()

    xs = [Vec{1,Int}((rand(-100:100),)) for i in 1:100]
    testVectorspace(zeros(Vec{1,Int}), xs, isequal)

    xs = [Vec{2,Int}((rand(-100:100), rand(-100:100))) for i in 1:100]
    testVectorspace(zeros(Vec{2,Int}), xs, isequal)

    @testset "Vecs" begin
    end

end

testVecs()
