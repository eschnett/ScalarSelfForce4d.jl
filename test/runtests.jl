using ScalarSelfForce4d

using LinearAlgebra
using Test



const par3 = Par{3,Float64}(9)
const spar3 = makestaggered(par3)

@testset "Parameters" begin
    @test all(!staggered(par3))
    @test all(par3.n .== 9)
    @test all(staggered(spar3))
    @test all(spar3.n .== 8)
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



@generated function waveD(x::Vec{D,T})::T where {D,T}
    fs = [:(sinpi(x[$d])) for d in 1:D-1]
    quote
        $(Expr(:meta, :inline))
        w = sqrt(T(3))
        *($(fs...), sinpi(w * x[D]))
    end
end

function epsD(spar::Par{D,T,Val{Staggered}}, x::Vec{D,T})::T where
        {D,T,Staggered}
    w = sqrt(T(D-1))
    if D == 2
        phix = pi * cospi(x[1]) * sinpi(w * x[2])
        phit = pi * w * sinpi(x[1]) * cospi(w * x[2])
        eps = (phix^2 + phit^2) / 2
    elseif D == 4
        phi = sinpi(x[1]) * sinpi(x[2]) * sinpi(x[3]) * sinpi(w * x[4])
        phix = pi * cospi(x[1]) * sinpi(x[2]) * sinpi(x[3]) * sinpi(w * x[4])
        phiy = pi * sinpi(x[1]) * cospi(x[2]) * sinpi(x[3]) * sinpi(w * x[4])
        phiz = pi * sinpi(x[1]) * sinpi(x[2]) * cospi(x[3]) * sinpi(w * x[4])
        phit = pi * w * sinpi(x[1]) * sinpi(x[2]) * sinpi(x[3]) * cospi(w * x[4])
        eps = (phix^2 + phiy^2 + phiz^2 + phit^2) / 2
    else
        @assert false
    end
    eps
end

const par2 = Par{2,Float64}(9)
const spar2 = makestaggered(par2)

@testset "Energy density of scalar wave" begin
    phi = approximate(waveD, Float64, par2)
    eps = scalarwave_energy(phi)

    epsD1(x) = epsD(spar2, x)
    eps0 = approximate(epsD1, Float64, spar2)
    err = eps - eps0
    maxerr = norm(err, Inf)
    @test 0.21 <= maxerr < 0.22
end



const par4 = Par{4,Float64}(9)
const spar4 = makestaggered(par4)

#TODO @testset "Energy density of scalar wave" begin
#TODO     phi = approximate(waveD, Float64, par4)
#TODO     eps = scalarwave_energy(phi)
#TODO 
#TODO     epsD1(x) = epsD(spar4, x)
#TODO     eps0 = approximate(epsD1, Float64, spar4)
#TODO     err = eps - eps0
#TODO     @show maxerr = norm(err, Inf)
#TODO     @test 0.046 <= maxerr < 0.047
#TODO end



@testset "Scalar wave equation" begin
    pot = zeros(Fun{4,Float64,Float64}, par4)
    bvals = approximate(waveD, Float64, par4)
    sol = solve_dAlembert_Dirichlet(pot, bvals)

    err = sol - bvals
    maxerr = norm(err, Inf)
    @test 0.046 <= maxerr < 0.047
end

@testset "Scalar wave equation with singular source" begin
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
