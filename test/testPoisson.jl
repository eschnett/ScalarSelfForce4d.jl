using ScalarSelfForce4d.Domains

using Memoize



@memoize Dict function ldom(D::Int)::Domain
    Domain{D, Float64}(9, lorentzian=true)
end



@generated function waveD(x::Vec{D,T})::T where {D,T}
    quote
        k = tuple($([d<D ? :(T(1)) : :(sqrt(T(D-1))) for d in 1:D]...))
        *($([:(sinpi(k[$d] * x[$d])) for d in 1:D]...))
    end
end

@generated function epsD(x::Vec{D,T})::T where {D, T}
    quote
        k = tuple($([d<D ? :(T(1)) : :(sqrt(T(D-1))) for d in 1:D]...))
        dphi = tuple($([
            :(*($([d == dir ?
                   :(pi * k[$d] * cospi(k[$d] * x[$d])) :
                   :(sinpi(k[$d] * x[$d])) for d in 1:D]...)))
            for dir in 1:D]...))
        eps = +($([:(dphi[$d]^2) for d in 1:D]...)) / 2
        eps
    end
end



function testPoisson()
    @testset "Poisson equation" begin
        lap = laplace(Val(0), Val(false), dom(3))
        lap1 = laplace1(Val(0), Val(false), dom(3))
        adel = 2pi * approximate_delta(dom(3), zeros(Vec{3,Float64}))
        del = Form(Dict(() => adel))
        dir = dirichlet(Val(0), Val(false), dom(3))
        bvals = zeros(typeof(del), dom(3))
        bnd = boundary(Val(0), Val(false), dom(3))
        op = mix_op_bc(bnd, lap, dir, dom(3))
        rhs = mix_op_bc(bnd, del, bvals)
        pot = op \ rhs

        res = op * pot - rhs
        maxres = norm(res[()], Inf)
        @test maxres < 1.0e-12
    end

    @testset "Scalar wave equation" begin
        pot = zeros(Form{4,0,false,Float64,Float64}, ldom(4))
        abvals = approximate(waveD, ldom(4))
        bvals = Form(Dict(() => abvals))
        sol = solve_dAlembert_Dirichlet(pot, bvals)

        err = sol - bvals
        maxerr = norm(err[()], Inf)
        @test 0.037 <= maxerr < 0.038
    end

    @testset "Scalar wave equation with singular source" begin
        @assert all(ldom(4).n[d] == dom(3).n[d] for d in 1:3)
    
        # Source
        lap3 = laplace(Val(0), Val(false), dom(3))
        adel3 = 2pi * approximate_delta(dom(3), zeros(Vec{3,Float64}))
        del3 = Form(Dict(() => adel3))
        dir3 = dirichlet(Val(0), Val(false), dom(3))
        bvals3 = zeros(typeof(del3), dom(3))
    
        bnd3 = boundary(Val(0), Val(false), dom(3))
        op3 = mix_op_bc(bnd3, lap3, dir3, dom(3))
        rhs3 = mix_op_bc(bnd3, del3, bvals3)
        pot3 = op3 \ rhs3
    
        # Potential (source)
        asrc = zeros(Fun{4,Float64,Float64}, ldom(4))
        for i4 in 1:ldom(4).n[4]
            # asrc.coeffs[:,:,:,i4] = del3[()].coeffs[:,:,:]
            asrc.coeffs[:,:,:,i4] = rhs3[()].coeffs[:,:,:]
        end
        src = Form(Dict(() => asrc))
        # Initial and boundary conditions
        abvals = zeros(Fun{4,Float64,Float64}, ldom(4))
        for i4 in 1:ldom(4).n[4]
            abvals.coeffs[:,:,:,i4] = pot3[()].coeffs[:,:,:]
        end
        bvals = Form(Dict(() => abvals))
    
        sol = solve_dAlembert_Dirichlet(src, bvals)
    
        err = sol - bvals
        maxerr = norm(err[()], Inf)
        @test maxerr < 1.0e-12
    end
end

testPoisson()
