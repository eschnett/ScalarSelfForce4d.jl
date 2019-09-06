using ScalarSelfForce4d
using Test



const par3 = Par{3,Float64}(9)

@testset "Parameters" begin
    @test all(par3.n .== 9)
end



function sinpiD(x::Vec{D,T})::T where {D,T}
    prod(sinpi(x[d]) for d in 1:D)
end

const fsinpi = approximate(sinpiD, Float64, par3)

@testset "Approximation" begin
    function find_maxerr()
        maxerr = 0.0
        for z in LinRange(-1.0, 1.0, 11),
            y in LinRange(-1.0, 1.0, 11),
            x in LinRange(-1.0, 1.0, 11)
            f0 = sinpiD(Vec((x, y, z)))
            f = fsinpi(Vec((x, y, z)))
            df = f - f0
            maxerr = max(maxerr, abs(df))
        end
        maxerr
    end
    maxerr = find_maxerr()
    @test 0.05 <= maxerr < 0.06
end



@testset "Derivatives" begin
    function sinpiDx(x::Vec{D,T})::T where {D,T}
        pi * cospi(x[1]) * prod(sinpi(x[d]) for d in 2:D)
    end
    
    fsinpix1 = deriv(fsinpi, 1)
    fsinpix2 = approximate(sinpiDx, Float64, par3)
    
    maxdiffx = maximum(abs.(fsinpix1.coeffs .- fsinpix2.coeffs))
    @test 0.42 <= maxdiffx < 0.43
end



@testset "Second derivatives" begin
    function sinpiDxx(x::Vec{D,T})::T where {D,T}
        - pi^2 * sinpi(x[1]) * prod(sinpi(x[d]) for d in 2:D)
    end
    
    fsinpixx1 = deriv2(fsinpi, 1, 1)
    fsinpixx2 = approximate(sinpiDxx, Float64, par3)
    
    maxdiffxx = maximum(abs.(fsinpixx1.coeffs .- fsinpixx2.coeffs))
    @test 7.3 <= maxdiffxx < 7.4
    
    function sinpiDxy(x::Vec{D,T})::T where {D,T}
        pi^2 * cospi(x[1]) * cospi(x[2]) * prod(sinpi(x[d]) for d in 3:D)
    end
    
    fsinpixy1 = deriv2(fsinpi, 1, 2)
    fsinpixy2 = approximate(sinpiDxy, Float64, par3)
    
    maxdiffxy = maximum(abs.(fsinpixy1.coeffs .- fsinpixy2.coeffs))
    @test 2.4 <= maxdiffxy < 2.5
end



@testset "Poisson equation" begin
    lap = laplace(Float64, par3)
    del = 2pi * approximate_delta(Float64, par3, Vec((0.0, 0.0, 0.0)))
    dir = dirichlet(Float64, par3)
    bvals = zero(typeof(del), par3)

    bnd = boundary(Float64, par3)
    op = mix_op_bc(bnd, lap, dir)
    rhs = mix_op_bc(bnd, del, bvals)
    pot = op \ rhs

    res = op * pot - rhs
    maxres = maximum(abs.(res.coeffs))
    @test maxres < 1.0e-12
end



const par4 = Par{4,Float64}(9)

function waveD(x::Vec{D,T})::T where {D,T}
    w = sqrt(T(3))
    prod(sinpi(x[d]) for d in 1:D-1) * sinpi(w * x[D])
end

@testset "Scalar wave equation" begin
    bnd = boundaryIV(Float64, par4)

    dal = dAlembert(Float64, par4)
    pot = zero(Fun{4,Float64,Float64}, par4)
    dir = dirichletIV(Float64, par4)
    bvals = approximate(waveD, Float64, par4, mask=bnd, rtol=1.0e-4)

    op = mix_op_bc(bnd, dal, dir)
    rhs = mix_op_bc(bnd, pot, bvals)
    sol = op \ rhs

    res = op * sol - rhs
    maxres = maximum(abs.(res.coeffs))
    @test maxres < 1.0e-12
end
