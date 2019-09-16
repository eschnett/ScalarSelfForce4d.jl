function testVectorspace(z::T, xs::AbstractVector{T}, iseq) where {T}
    @testset "Vectorspace.setup $T" begin
        @test !isempty(xs)
        x = first(xs)
        @test iseq(x, x) === true
    end

    ys = rand(xs, length(xs))
    zs = rand(xs, length(xs))

    @testset "Vectorspace.add.types $T" begin
        @test z isa T
        for x in xs
            @test x isa T
            @test (x + z) isa T
        end
        for (x,y) in zip(xs, ys)
            @test x + y isa T
        end
    end

    @testset "Vectorspace.add.assoc $T" begin
        for (x,y,z) in zip(xs, ys, zs)
            @test iseq(x + (y + z), (x + y) + z)
        end
    end

    @testset "Vectorspace.add.comm $T" begin
        for (x,y) in zip(xs, ys)
            @test iseq(x + y, y + x)
        end
    end

    @testset "Vectorspace.add.zero $T" begin
        @test iseq(z, z)
        @test iseq(+z, z)
        @test iseq(z + z, z)
        for x in xs
            @test iseq(x + z, x)
            @test iseq(z + x, x)
        end
    end

    @testset "Vectorspace.add.inv $T" begin
        @test iseq(-z, z)
        for x in xs
            @test iseq(-(-x), x)
            @test iseq(x + (-x), z)
            @test iseq((-x) + x, z)
            @test iseq(x - x, z)
        end
        for (x,y) in zip(xs, ys)
            @test iseq(x - y, x + (-y))
        end
    end
end
