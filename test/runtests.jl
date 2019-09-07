using ScalarSelfForce4d

using LinearAlgebra
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
    @test 0.064 <= maxerr < 0.065
end



@testset "Derivatives" begin
    function sinpiDx(x::Vec{D,T})::T where {D,T}
        pi * cospi(x[1]) * prod(sinpi(x[d]) for d in 2:D)
    end
    
    fsinpix1 = deriv(fsinpi, 1)
    fsinpix2 = approximate(sinpiDx, Float64, par3)
    
    maxdiffx = norm(fsinpix1 - fsinpix2, Inf)
    @test 0.44 <= maxdiffx < 0.45
end



@testset "Second derivatives" begin
    function sinpiDxx(x::Vec{D,T})::T where {D,T}
        - pi^2 * sinpi(x[1]) * prod(sinpi(x[d]) for d in 2:D)
    end
    
    fsinpixx1 = deriv2(fsinpi, 1, 1)
    fsinpixx2 = approximate(sinpiDxx, Float64, par3)
    
    maxdiffxx = norm(fsinpixx1 - fsinpixx2, Inf)
    @test 7.1 <= maxdiffxx < 7.2
    
    function sinpiDxy(x::Vec{D,T})::T where {D,T}
        pi^2 * cospi(x[1]) * cospi(x[2]) * prod(sinpi(x[d]) for d in 3:D)
    end
    
    fsinpixy1 = deriv2(fsinpi, 1, 2)
    fsinpixy2 = approximate(sinpiDxy, Float64, par3)
    
    maxdiffxy = norm(fsinpixy1 - fsinpixy2, Inf)
    @test 2.6 <= maxdiffxy < 2.7
end



@testset "Poisson equation" begin
    lap = laplace(Float64, par3)
    del = 2pi * approximate_delta(Float64, par3, Vec((0.0, 0.0, 0.0)))
    dir = dirichlet(Float64, par3)
    bvals = zeros(typeof(del), par3)

    bnd = boundary(Float64, par3)
    op = mix_op_bc(bnd, lap, dir)
    rhs = mix_op_bc(bnd, del, bvals)
    pot = op \ rhs

    res = op * pot - rhs
    maxres = norm(res, Inf)
    @test maxres < 1.0e-12
end



const par4 = Par{4,Float64}(9)

@generated function waveD(x::Vec{D,T})::T where {D,T}
    fs = [:(sinpi(x[$d])) for d in 1:D-1]
    quote
        $(Expr(:meta, :inline))
        w = sqrt(T(3))
        *($(fs...), sinpi(w * x[D]))
    end
end
waveD(x) = waveD(Vec(x))

@testset "Scalar wave equation" begin
    pot = zeros(Fun{4,Float64,Float64}, par4)
    bvals = approximate(waveD, Float64, par4)
    sol = solve_dAlembert_Dirichlet(pot, bvals)

    err = sol - bvals
    maxerr = norm(err, Inf)
    @test 0.046 <= maxerr < 0.047
end

@testset "Scalar wave equation with singularity" begin
    @assert all(par4.n[d] == par3.n[d] for d in 1:3)

    lap3 = laplace(Float64, par3)
    del3 = 2pi * approximate_delta(Float64, par3, Vec((0.0, 0.0, 0.0)))
    dir3 = dirichlet(Float64, par3)
    bvals3 = zeros(typeof(del3), par3)

    bnd3 = boundary(Float64, par3)
    op3 = mix_op_bc(bnd3, lap3, dir3)
    rhs3 = mix_op_bc(bnd3, del3, bvals3)
    pot3 = op3 \ rhs3

    pot = zeros(Fun{4,Float64,Float64}, par4)
    for i4 in 1:par4.n[4]
        pot.coeffs[:,:,:,i4] = del3.coeffs[:,:,:]
    end
    bvals = zeros(Fun{4,Float64,Float64}, par4)
    for i4 in 1:par4.n[4]
        bvals.coeffs[:,:,:,i4] = pot3.coeffs[:,:,:]
    end

    sol = solve_dAlembert_Dirichlet(pot, bvals)

    err = sol - bvals
    maxerr = norm(err, Inf)
    @test maxerr < 1.0e-12
end
