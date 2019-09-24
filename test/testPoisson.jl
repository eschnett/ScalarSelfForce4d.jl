using ScalarSelfForce4d.Domains

using Arpack
using DataStructures
using Memoize



@memoize Dict function ldom(D::Int)::Domain
    Domain{D,Float64}(9, lorentzian = true)
end



@generated function waveD(x::Vec{D,T})::T where {D,T}
    quote
        k = tuple($([d < D ? :(T(1)) : :(sqrt(T(D - 1))) for d in 1:D]...))
        *($([:(sinpi(k[$d] * x[$d])) for d in 1:D]...))
    end
end

@generated function epsD(x::Vec{D,T})::T where {D,T}
    quote
        k = tuple($([d < D ? :(T(1)) : :(sqrt(T(D - 1))) for d in 1:D]...))
        dphi = tuple($([
            :(*($([d == dir ?
                   :(pi * k[$d] * cospi(k[$d] * x[$d])) :
                   :(sinpi(k[$d] * x[$d])) for d in 1:D]...)))
            for dir in 1:D]...))
        eps = +($([:(dphi[$d]^2) for d in 1:D]...)) / 2
        eps
    end
end
function etotD(dom::Domain{D,T}, t::T)::T where {D,T}
    T(pi)^2 * D/2
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

        phi = sol
        dphi = deriv(phi)
        sdphi = star(dphi)
        act = wedge(dphi, sdphi) / 2.0

        # Note: The action has a sign error because the metric
        # signature is not yet handled properly
        # et = Form(Dict(() => approximate(xt -> xt[end], dom)))
        # nt = deriv(et)
        # mom = phit
        # ham = wedge(mom, star(mom)) - act
        ham = act
        sldom = makestaggered(ldom(4), Vec(ntuple(d -> true, 4)))
        eham = Form(Dict((1,2,3,4) => approximate(epsD, sldom)))
        maxerr = norm((ham - eham)[(1,2,3,4)], Inf)
        @test isapprox(maxerr, 0.0027673958110688065; atol=1.0e-6)

        sdom3 = makestaggered(dom(3), Vec(ntuple(d -> true, 3)))
        etot = Array{Float64}(undef, size(ham[(1,2,3,4)].coeffs, 4))
        dx4 = spacing(ldom(4))[4]
        for i in 1:length(etot)
            cs = ham[(1,2,3,4)].coeffs[:,:,:,i] ./ dx4
            ene = Form(Dict((1,2,3) => Fun{3,Float64,Float64}(sdom3, cs)))
            etot[i] = sum(ene[(1,2,3)])
        end
        # etot = 14.804406601634037
        @test isapprox(minimum(etot), 18.071224176938696; atol=1.0e-6)
        @test isapprox(maximum(etot), 22.494721301641462; atol=1.0e-6)
    end

    @testset "Scalar wave equation with eigenmodes" begin
        lap3 = laplace(Val(0), Val(false), dom(3))
        dir3 = dirichlet(Val(0), Val(false), dom(3))
        bnd3 = boundary(Val(0), Val(false), dom(3))
        op3 = mix_op_bc(bnd3, lap3, -dir3, dom(3))

        nbndev = 9^3 - (9-2)^3
        opc3 = op3.comps[Vec{0,Int}(())][Vec{0,Int}(())]
        lambda, v = eigs(opc3; nev=nbndev+50, which=:LR)
        # lambda, v = eigen(Matrix(opc))

        # Count boundary eigenvalues
        @assert count(x -> abs(x+1)<1.0e-12, lambda) == nbndev
        # Normalize eigenvalues (avoid round-off)
        lambda2 = Array{Float64}(undef, length(lambda))
        for i in 1:length(lambda2)
            if i>1 && abs(lambda[i] - lambda2[i-1]) < 1.0e-12
                lambda2[i] = lambda2[i-1]
            else
                lambda2[i] = lambda[i]
            end
        end
        # Determine second non-boundary eigenvalue with multiplicity 1
        # (this is the mode we want)
        ms = counter(lambda2)
        l = sort(collect(keys(filter(lm->lm[2]==1, ms))), rev=true)[2]
        n = findfirst(==(l), lambda)
        @assert n !== nothing

        pot = zeros(Form{4,0,false,Float64,Float64}, ldom(4))
        abvals = zeros(Fun{4,Float64,Float64}, ldom(4))
        for i in 1:size(abvals.coeffs,4)
            abvals.coeffs[:,:,:,i] = reshape(real.(v[:,n]), (9,9,9))
        end
        bvals = Form(Dict(() => abvals))
        sol = solve_dAlembert_Dirichlet(pot, bvals)

        err = sol - bvals
        maxerr = norm(err[()], Inf)
        @test isapprox(maxerr, 0.2568071901840646; atol=1.0e-6)
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

if runtests
    testPoisson()
end
