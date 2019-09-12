using ScalarSelfForce4d.Quadrature



function testQuadrature()

    @testset "Quadrature $T" for T in [Float32, Float64, BigFloat,
                                       Rational{BigInt}]
        # Test a constant function
        @test isequal(quad(x->T(1), T, (T(0),), (T(1),), (1,)), T(1))
        if T <: AbstractFloat
            # Test sin
            @test isapprox(quad(x->sin(x[1]), T,
                                (T(0),), (T(pi),), (40,)), T(2))
            # Test log
            @test isapprox(quad(x->log(x[1]), T,
                                (T(1),), (T(ℯ),), (60,)), T(1))
        end
    
        gmean(x, y) = sqrt(x * y)
        if T <: Rational
            atol = T(0)
        else
            atol = gmean(eps(T), sqrt(eps(T)))
        end
    
        # TODO: test with Vec codomain
        # TODO: test multi-dim
        for n in 1:20
            # Define a polynomial
            coeffs = T[T(rand(-10:10)) / 10 for i in 1:n]
            poly(cs, x) = sum(c * x^(i-1) for (i,c) in enumerate(cs))
            poly(cs, x::Tuple) = poly(x[1])
            intcoeffs = T[0; [coeffs[i] / i for i in 1:n]]
            poly(x) = poly(coeffs, x)
            intpoly(x) = poly(intcoeffs, x)
    
            xmin = T(rand(-10:10)) / 10
            xmax = T(rand(-10:10)) / 10
            xmin,xmax = minmax(xmin, xmax)
            if xmin != xmax
                int = quad(poly, T, (xmin,), (xmax,), (n,))
                @test typeof(int) === T
                @test isapprox(int, intpoly(xmax) - intpoly(xmin); atol=atol)
            end
        end
    end

end

testQuadrature()
