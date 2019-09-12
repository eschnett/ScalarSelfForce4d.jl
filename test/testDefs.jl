using ScalarSelfForce4d.Defs



function testDefs()

    @testset "Defs.bitsign" begin
        @test signbit(bitsign(false)) === false
        @test signbit(bitsign(true)) === true
        @test bitsign(signbit(1)) === 1
        @test bitsign(signbit(-1)) === -1
    end



    @testset "Defs.linear $T" for T in [Float32, Float64, BigFloat,
                                        Rational{BigInt}]
        if T <: Rational
            atol = T(0)
        else
            atol = sqrt(eps(T))
        end
        for i in 1:100
            xmin = rand(Float64)
            xmax = rand(Float64)
            ymin = T(rand(Float64))
            ymax = T(rand(Float64))
            x = rand(Float64)
            if abs(xmax - xmin) >= 0.01
                @test isequal(linear(xmin, ymin, xmax, ymax, xmin), ymin)
                @test isequal(linear(xmin, ymin, xmax, ymax, xmax), ymax)
                y = linear(xmin, ymin, xmax, ymax, x)
                @test typeof(y) === T
                slope = (ymax - ymin) / T(xmax - xmin)
                if x != xmin
                    @test isapprox((y - ymin) / T(x - xmin), slope; atol=atol)
                end
                if x != xmax
                    @test isapprox((y - ymax) / T(x - xmax), slope; atol=atol)
                end
            end
        end
    end



    @testset "Defs.characteristic $T" for T in [Float32, Float64, BigFloat,
                                                Rational{BigInt}]
        for i in 1:100
            xmin = rand(Float64)
            xmax = rand(Float64)
            xmin,xmax = minmax(xmin, xmax)
            x = rand(Float64)
            if xmin != xmax
                @test isequal(characteristic(xmin, T(1)/2, xmax, T(1)/2, xmin),
                              T(1)/2)
                @test isequal(characteristic(xmin, T(1)/2, xmax, T(1)/2, xmax),
                              T(1)/2)
                y = characteristic(xmin, T(1)/2, xmax, T(1)/2, x)
                @test typeof(y) === T
                if xmin < x < xmax
                    @test y == T(1)
                end
                if x == xmin || x == xmax
                    @test y == T(1)/2
                end
                if x < xmin || x > xmax
                    @test y == T(0)
                end
            end
        end
    end

end

testDefs()
