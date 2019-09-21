function testVectorspace(z::U, as::AbstractVector{T},
                         xs::AbstractVector{U}, iseq) where {T <: Number,U}
    @test T === eltype(U)

    @testset "Vectorspace.setup $U" begin
        @test !isempty(xs)
        x = first(xs)
        @test iseq(x, x) === true
    end

    bs = rand(as, length(as))
    ys = rand(xs, length(xs))
    zs = rand(xs, length(xs))

    @testset "Vectorspace.add.types $U" begin
        @test z isa U
        for x in xs
            @test x isa U
            @test (x + z) isa U
        end
        for (x, y) in zip(xs, ys)
            @test (x + y) isa U
        end
        for a in as
            @test a isa T
        end
        for (a, x) in zip(as, xs)
            @test (a * x) isa U
        end
    end

    @testset "Vectorspace.add.assoc $U" begin
        for (x, y, z) in zip(xs, ys, zs)
            @test iseq(x + (y + z), (x + y) + z)
        end
    end

    @testset "Vectorspace.add.comm $U" begin
        for (x, y) in zip(xs, ys)
            @test iseq(x + y, y + x)
        end
    end

    @testset "Vectorspace.add.zero $U" begin
        @test iseq(z, z)
        @test iseq(+z, z)
        @test iseq(z + z, z)
        @test iszero(z)
        for x in xs
            @test iszero(x) === iseq(x, z)
            @test iseq(x + z, x)
            @test iseq(z + x, x)
        end
    end

    @testset "Vectorspace.add.inv $U" begin
        @test iseq(-z, z)
        for x in xs
            @test iseq(-(-x), x)
            @test iseq(x + (-x), z)
            @test iseq((-x) + x, z)
            @test iseq(x - x, z)
        end
        for (x, y) in zip(xs, ys)
            @test iseq(x - y, x + (-y))
        end
    end

    @testset "Vectorspace.scale $U" begin
        for x in xs
            @test iseq(zero(T) * x, z)
            @test iseq(x * zero(T), z)
            @test iseq(one(T) * x, x)
            @test iseq(x * one(T), x)
        end
        for a in as
            @test iseq(a * z, z)
            @test iseq(z * a, z)
            if a != zero(T)
                @test iseq(a \ z, z)
                @test iseq(z / a, z)
            end
        end
        for (a, x) in zip(as, xs)
            @test iseq(a * x, x * a)
            @test iseq((-a) * x, a * (-x))
            if a != zero(T)
                @test iseq(inv(a) * x, a \ x)
                @test iseq(x * inv(a), x / a)
                @test iseq(inv(a) * (a * x), x)
                @test iseq((x * a) * inv(a), x)
            end
        end
        for (a, b, x) in zip(as, bs, xs)
            @test iseq(a * (b * x), (a * b) * x)
        end
        for (a, x, y) in zip(as, xs, ys)
            @test iseq(a * (x + y), a * x + a * y)
            @test iseq((x + y) * a, x * a + y * a)
        end
    end
end
