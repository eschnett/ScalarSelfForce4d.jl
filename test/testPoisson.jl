const dom4 = Domain{4,Float64}(9, lorentzian=true)

# @generated function waveD(x::Vec{D,T})::T where {D,T}
#     fs = [:(sinpi(x[$d])) for d in 1:D-1]
#     quote
#         $(Expr(:meta, :inline))
#         w = sqrt(T(3))
#         *($(fs...), sinpi(w * x[D]))
#     end
# end

function waveD(x::Vec{D,T})::T where {D,T}
    k = ntuple(d -> d<D ? T(1) : sqrt(T(D-1)), D)
    prod(sinpi(k[d] * x[d]) for d in 1:D)
end



function testPoisson()
    @testset "Poisson equation" begin
        lap = laplace(dom3)
        del = 2pi * approximate_delta(dom3, Vec((0.0, 0.0, 0.0)))
        dir = dirichlet(dom3)
        bvals = zeros(typeof(del), dom3)

        bnd = boundary(dom3)
        op = mix_op_bc(bnd, lap, dir)
        rhs = mix_op_bc(bnd, del, bvals)
        pot = op \ rhs

        res = op * pot - rhs
        maxres = norm(res, Inf)
        @test maxres < 1.0e-12
    end

    @testset "Scalar wave equation" begin
        pot = zeros(Fun{4,Float64,Float64}, dom4)
        bvals = approximate(waveD, dom4)
        sol = solve_dAlembert_Dirichlet(pot, bvals)

        err = sol - bvals
        maxerr = norm(err, Inf)
        @test 0.036 <= maxerr < 0.037
    end

    @testset "Scalar wave equation with singular source" begin
        @assert all(dom4.n[d] == dom3.n[d] for d in 1:3)

        # Source
        lap3 = laplace(dom3)
        del3 = 2pi * approximate_delta(dom3, Vec((0.0, 0.0, 0.0)))
        dir3 = dirichlet(dom3)
        bvals3 = zeros(typeof(del3), dom3)

        bnd3 = boundary(dom3)
        op3 = mix_op_bc(bnd3, lap3, dir3)
        rhs3 = mix_op_bc(bnd3, del3, bvals3)
        pot3 = op3 \ rhs3

        # Potential
        pot = zeros(Fun{4,Float64,Float64}, dom4)
        for i4 in 1:dom4.n[4]
            pot.coeffs[:,:,:,i4] = del3.coeffs[:,:,:]
        end
        # Initial and boundary conditions
        bvals = zeros(Fun{4,Float64,Float64}, dom4)
        for i4 in 1:dom4.n[4]
            bvals.coeffs[:,:,:,i4] = pot3.coeffs[:,:,:]
        end

        sol = solve_dAlembert_Dirichlet(pot, bvals)

        err = sol - bvals
        maxerr = norm(err, Inf)
        @test maxerr < 1.0e-12
    end
end

testPoisson()
