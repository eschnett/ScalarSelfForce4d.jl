using ScalarSelfForce4d.Forms

using Combinatorics
using LinearAlgebra
using Memoize



@generated function dsinpiD(x::Vec{D,T}, dir::Int)::T where {D,T}
    quote
        $([quote
            if dir == $dir1
                return *($([d == dir1 ?
                               :(pi * cospi(x[$d])) :
                               :(sinpi(x[$d])) for d in 1:D]...))
            end
        end
           for dir1 in 1:D]...)
        T(0)
    end
end

function d2sinpiD(x::Vec{D,T}, dir::Int)::T where {D,T}
    -pi^2 * sinpiD(x)
end

function lsinpiD(x::Vec{D,T})::T where {D,T}
    -D * pi^2 * sinpiD(x)
end



@memoize Dict function fsinpi(D::Int)::Form
    Form(Dict(() => asinpi(D)))
end
@memoize Dict function dfsinpi(D::Int)::Form
    deriv(fsinpi(D))
end



function testForms()

    BigRat = Rational{BigInt}
    bigrange = -10 : big(1//10) : 10

    # Forms form a vector space
    for D in 1:4, lorentzian in [false]
        dom = Domain{D,BigRat}(ntuple(d->d+2, D), lorentzian=lorentzian)
        for R in 0:D, Dual in false:true
            z = zeros(Form{D,R,Dual,BigRat,BigRat}, dom)
            as = [rand(bigrange) for i in 1:10]
            xs = [rand(bigrange, Form{D,R,Dual,BigRat,BigRat}, dom)
                  for i in 1:10]
            testVectorspace(z, as, xs, isequal)
        end
    end
    
    # Operators form a vector space
    for D in 1:4, lorentzian in [false]
        dom = Domain{D,BigRat}(ntuple(d->d+2, D), lorentzian=lorentzian)
        for RI in 0:D, DualI in false:true
            for RJ in 0:D, DualJ in false:true
                z = zeros(FOp{D,RI,DualI,RJ,DualJ,BigRat,BigRat})
                xs = FOp{D,RI,DualI,RJ,DualJ,BigRat,BigRat}[]
                if RI == RJ && DualI == DualJ
                    push!(xs, one(FOp{D,RI,DualI,RJ,DualJ,BigRat,BigRat}, dom))
                end
                if RI == D - RJ && DualI == !DualJ
                    push!(xs, star(Val(RJ), Val(DualJ), dom))
                end
                if RI == RJ + 1 && DualI == DualJ
                    push!(xs, deriv(Val(RJ), Val(DualJ), dom))
                end
                if RI == D - (RJ + 1) && DualI == !DualJ
                    d = deriv(Val(RJ), Val(DualJ), dom)
                    s = star(Val(RJ + 1), Val(DualJ), dom)
                    push!(xs, s * d)
                end
                if RI == (D - RJ) + 1 && DualI == !DualJ
                    s = star(Val(RJ), Val(DualJ), dom)
                    d = deriv(Val(D - RJ), Val(!DualJ), dom)
                    push!(xs, d * s)
                end
                if RI == RJ - 1 && DualI == DualJ
                    push!(xs, coderiv(Val(RJ), Val(DualJ), dom))
                end
                if RI == D - (RJ - 1) && DualI == !DualJ
                    c = coderiv(Val(RJ), Val(DualJ), dom)
                    s = star(Val(RJ - 1), Val(DualJ), dom)
                    push!(xs, s * c)
                end
                if RI == (D - RJ) - 1 && DualI == !DualJ
                    s = star(Val(RJ), Val(DualJ), dom)
                    c = coderiv(Val(D - RJ), Val(!DualJ), dom)
                    push!(xs, c * s)
                end
                if !isempty(xs)
                    as = [rand(bigrange) for i in 1:10]
                    testVectorspace(z, as, xs, isequal)
                end
            end
        end
    end
    
    @testset "Forms.Wedge D=$D lorentzian=$lorentzian" for D in 1:4, lorentzian in [false]
        dom = Domain{D,BigRat}(ntuple(d->d+2, D), lorentzian=lorentzian)
        for RI in 0:D, RJ in 0:D - RI, DualI in false:true, DualJ in false:true
            f0 = zeros(Form{D,RI,DualI,BigRat,BigRat}, dom)
            g0 = zeros(Form{D,RJ,DualJ,BigRat,BigRat}, dom)
    
            # we test all primal-primal as well as certain primal-dual
            # operations
            (!DualI && !DualJ) || (!DualI && DualJ && RI + RJ == D) || continue
    
            # Unit vectors
            for (i,fi) in f0.comps, (j,gj) in g0.comps
                istag = idx2staggered(Val(D), i)
                jstag = idx2staggered(Val(D), j)
                if any(istag & jstag)
                    s = 0
                else
                    ij = Int[i..., j...]
                    s = 1
                    while !isempty(ij)
                        y,x = findmin(ij)
                        if iseven(x)
                            s = -s
                        end
                        deleteat!(ij, x)
                    end
                end
                k = Vec{RI+RJ,Int}(Tuple(sort(Int[i..., j...])))
                f = zeros(Form{D,RI,DualI,BigRat,BigRat}, dom)
                g = zeros(Form{D,RJ,DualJ,BigRat,BigRat}, dom)
                h = zeros(Form{D,RI+RJ,false,BigRat,BigRat}, dom)
                f.comps[i] = fconst(f.comps[i].dom, BigRat(1))
                g.comps[j] = fconst(g.comps[j].dom, BigRat(1))
                if s != 0
                    h.comps[k] = fconst(h.comps[k].dom, BigRat(s))
                end
                r = wedge(f, g)
                @test r == h
            end
    
            for n in 1:10
                a = rand(bigrange)
                f = rand(bigrange, Form{D,RI,DualI,BigRat,BigRat}, dom)
                f2 = rand(bigrange, Form{D,RI,DualI,BigRat,BigRat}, dom)
                g = rand(bigrange, Form{D,RJ,DualJ,BigRat,BigRat}, dom)
                g2 = rand(bigrange, Form{D,RJ,DualJ,BigRat,BigRat}, dom)
    
                # Linearity
                @test isequal(wedge(a * f, g), a * wedge(f, g))
                @test isequal(wedge(f, a * g), a * wedge(f, g))
                @test isequal(wedge(f + f2, g), wedge(f, g) + wedge(f2, g))
                @test isequal(wedge(f, g + g2), wedge(f, g) + wedge(f, g2))
    
                # Note: the discrete primal-primal wedge is not associative
    
                # TODO: associative for closed forms
                # TODO: natural under pullbacks
    
                # Antisymmetry
                if DualI == DualJ
                    if RJ == RI && isodd(RI)
                        @test iszero(wedge(f, f))
                    end
                    s = BigRat(bitsign(RI * RJ))
                    @test isequal(wedge(g, f), s * wedge(f, g))
                end
            end
        end
    end
    
    @testset "Forms.Star D=$D lorentzian=$lorentzian" for D in 1:4, lorentzian in false:true
        dom = Domain{D,BigRat}(ntuple(d->d+2, D), lorentzian=lorentzian)
    
        for R in 0:D, Dual in false:true
            f = rand(bigrange, Form{D,R,Dual,BigRat,BigRat}, dom)
            sf = star(f)
            ssf = star(sf)
            s = bitsign(R * (D - R)) * bitsign(lorentzian)
            @test f == BigRat(s) * ssf
        end
    
        if !lorentzian
        for R in 0:D, Dual in [false]
            f = rand(bigrange, Form{D,R,Dual,BigRat,BigRat}, dom)
            sf = star(f)
            fsf = wedge(f, sf)
            for (i,fsfi) in fsf.comps
                @test all(fsfi.coeffs .>= 0)
            end
                @test integral(fsf) > 0
                z = zeros(Form{D,R,Dual,BigRat,BigRat}, dom)
                @test integral(wedge(z, star(z))) == 0
            end
        end
    end

    @testset "Forms.Derivatives D=$D" for D in 1:4, lorentzian in [false]
        dom = Domain{D,BigRat}(ntuple(d->d+3, D), lorentzian=lorentzian)
        for R in 0:D-1, Dual in [false] # false:true

            # Zero
            z = zeros(Form{D,R,Dual,BigRat,BigRat}, dom)
            dz = deriv(z)
            @test iszero(dz)

            # Constant
            c = zeros(Form{D,R,Dual,BigRat,BigRat}, dom)
            for (i,ci) in c.comps
                c.comps[i] = fconst(c.comps[i].dom, BigRat(1))
            end
            dc = deriv(c)
            @test iszero(dc)

            # Linear functions
            f0 = zeros(Form{D,R,Dual,BigRat,BigRat}, dom)
            for (i,fi) in f0.comps, dir in 1:D
                f = zeros(Form{D,R,Dual,BigRat,BigRat}, dom)
                f.comps[i] = sample(x->x[dir], f.comps[i].dom)
                df = deriv(f)
                for (j,dfj) in df.comps
                    if !(dir in i) && issubset(i, j) && dir in j
                        s = bitsign(count(dir .> i))
                    else
                        s = 0
                    end
                    rc = sample(x->BigRat(s), dfj.dom)
                    @test dfj == rc
                end
            end

            # Stokes's theorem
            for (i,fi) in f0.comps, dir in 1:D
                f = zeros(Form{D,R,Dual,BigRat,BigRat}, dom)
                f.comps[i] = sample(x->x[dir], f.comps[i].dom)
                df = deriv(f)
                bf = boundary(f)
                @test integral(df) == integral(bf)
            end

            # Leibniz rule
            for (i,fi) in f0.comps, idir in 1:D
                f = zeros(Form{D,R,Dual,BigRat,BigRat}, dom)
                f.comps[i] = sample(x->x[idir], f.comps[i].dom)
                RJ = D - R - 1
                DualJ = false # false:true
                g0 = zeros(Form{D,RJ,DualJ,BigRat,BigRat}, dom)
                for (j,gj) in g0.comps, jdir in 1:D
                    g = zeros(Form{D,RJ,DualJ,BigRat,BigRat}, dom)
                    g.comps[j] = sample(x->x[jdir], g.comps[j].dom)
                    # Why does s=1 always work? Is the result always
                    # zero?
                    s = 1
                    @test (integral(wedge(deriv(f), g)) ==
                           s * integral(wedge(f, deriv(g))) +
                           integral(boundary(wedge(f, g))))
                end
            end

            @warn "1:10"
            for n in 1:1
                a = rand(bigrange)
                f = rand(bigrange, Form{D,R,Dual,BigRat,BigRat}, dom)
                f2 = rand(bigrange, Form{D,R,Dual,BigRat,BigRat}, dom)

                if R <= D - 2
                    @test iszero(deriv(deriv(f)))
                end

                # Linearity
                @test deriv(a * f) == a * deriv(f)
                @test deriv(f + f2) == deriv(f) + deriv(f2)

                # Stokes
                @test integral(deriv(f)) == integral(boundary(f))

                # Leibniz rule
                RJ = D - R - 1
                DualJ = false # false:true
                g = rand(bigrange, Form{D,RJ,DualJ,BigRat,BigRat}, dom)
                s = iseven(R) ? -1 : 1
                # if (D,R) == (2,1) || (D,R) == (3,1)
                #     s = 1
                # elseif (D,R) == (1,0) || (D,R) == (2,0) || (D,R) == (3,0) || (D,R) == (3,2) || (D,R) == (4,0)
                #     s = -1
                # else
                #     @assert false
                # end
                @show (D, lorentzian) (R, Dual) (RJ, DualJ) s
                @test (integral(wedge(deriv(f), g)) ==
                       s * integral(wedge(f, deriv(g))) +
                       integral(boundary(wedge(f, g))))
                @assert (integral(wedge(deriv(f), g)) ==
                         s * integral(wedge(f, deriv(g))) +
                         integral(boundary(wedge(f, g))))
            end

        end
    end

    @testset "Forms.Derivatives D=$D" for D in 1:3
        atol = (D <= 3 ? 100 : 1000) * eps(1.0)

        # Constants have zero derivative
        ac = approximate(x->1.0, dom(D))
        fac = Form(Dict(() => ac))
        dfac = deriv(fac)
        for dir in 1:D
            err = maxabsdiff(dom(D), x->0.0, dfac[(dir,)])
            @test isapprox(err, 0; atol = atol)
        end

        # Linear functions have unit derivative
        for dir in 1:D
            ax = approximate(x->x[dir], dom(D))
            fax = Form(Dict(() => ax))
            dfax = deriv(fax)
            for d in 1:D
                err = maxabsdiff(dom(D), x->d == dir ? 1.0 : 0.0, dfax[(d,)])
                @test isapprox(err, 0; atol = atol)
            end
        end

        # TODO: linearity, product rule, (chain rule)

        fsinpi(D) = Form(Dict(() => asinpi(D)))
        dfsinpi(D) = deriv(fsinpi(D))
        for dir in 1:D
            domDx = makestaggered(dom(D), unitvec(Val(D), dir))
            adsinpix = approximate(x->dsinpiD(x, dir), domDx)
            err = norm(dfsinpi(D).comps[Vec((dir,))] - adsinpix, Inf)
            if D == 1
                @test isapprox(err, 0.03683663404089432; atol = 1.0e-6)
            elseif D == 2
                @test isapprox(err, 0.03879219734864847; atol = 1.0e-6)
            elseif D == 3
                @test isapprox(err, 0.04085157654377347; atol = 1.0e-6)
            elseif D == 4
                @test isapprox(err, 0.04302028294795979; atol = 1.0e-6)
            else
                @assert false
            end
        end
        if D > 1
            ddfsinpi = deriv(dfsinpi(D))
            for dir1 in 1:D, dir2 in dir1 + 1:D
                idx12 = Vec{2,Int}((dir1, dir2))
                maxerr = norm(ddfsinpi[idx12], Inf)
                @test isapprox(maxerr, 0; atol = atol)
            end
        end
    end

    @testset "Forms.Laplacian D=$D" for D in 1:4
        atol = 100 * eps(1.0)

        sdfsinpiD = star(dfsinpi(D))
        dsdfsinpiD = deriv(sdfsinpiD)
        sdsdfsinpiD = star(dsdfsinpiD)
        ad2sinpiD = approximate(lsinpiD, dom(D))
        err = norm(sdsdfsinpiD[()] - ad2sinpiD, Inf)
        if D == 1
            @test isapprox(err, 0.6047119237688179; atol = 1.0e-6)
        elseif D == 2
            @test isapprox(err, 0.9781685656156345; atol = 1.0e-6)
        elseif D == 3
            @test isapprox(err, 1.5317929733380282; atol = 1.0e-6)
        elseif D == 4
            @test isapprox(err, 2.150815960478276; atol = 1.0e-6)
        else
            @assert false
        end

        cdfsinpiD = coderiv(dfsinpi(D))
        s = Float64(bitsign(iseven(D) && isodd(D-1)))
        err = norm(cdfsinpiD[()] - s * sdsdfsinpiD[()], Inf)
        @test isapprox(err, 0; atol = atol)

        lfsinpiD = laplace(fsinpi(D))
        err = norm(lfsinpiD[()] - s * sdsdfsinpiD[()], Inf)
        @test isapprox(err, 0; atol = atol)
    end

    @testset "Forms.Operators D=$D lorentzian=$lorentzian" for D in 1:4,
            lorentzian in [false]   # false:true
        BigRat = Rational{BigInt}
        dom = Domain{D,BigRat}(ntuple(d->d+2, D), lorentzian = lorentzian)
        f = Form(Dict(() => Fun(dom, BigRat.(rand(-100:100, dom.n.elts)))))

        # Star

        idx0 = Vec{0,Int}(())
        idxD = Vec{D,Int}(ntuple(d->d, D))

        s0 = star(Val(0), Val(false), dom)
        sf = star(f)
        @test length(s0.comps) == 1
        @test haskey(s0.comps, idxD)
        @test length(s0.comps[idxD]) == 1
        @test haskey(s0.comps[idxD], idx0)
        s0f = s0 * f
        @test s0f[idxD] == sf[idxD]

        sD = star(Val(D), Val(true), makedual(dom, !dom.dual))
        ssf = star(sf)
        @test length(sD.comps) == 1
        @test haskey(sD.comps, idx0)
        @test length(sD.comps[idx0]) == 1
        @test haskey(sD.comps[idx0], idxD)
        @test sD.comps[idx0][idxD] * sf[idxD] == ssf[idx0]

        sD0 = sD * s0
        @test sD0.comps[idx0][idx0] == I

        df = deriv(f)
        sdf = star(df)
        s1 = star(Val(1), Val(false), dom)
        @test length(s1.comps) == D
        for dir in 1:D
            idx1 = Vec{1,Int}((dir,))
            idxD1 = Vec{D - 1,Int}(ntuple(d->d < dir ? d : d + 1, D - 1))
            @test haskey(s1.comps, idxD1)
            @test length(s1.comps[idxD1]) == 1
            @test haskey(s1.comps[idxD1], idx1)
            @test s1.comps[idxD1][idx1] * df[idx1] == sdf[idxD1]
        end

        sD1 = star(Val(D - 1), Val(true), makedual(dom, !dom.dual))
        ssdf = star(sdf)
        for dir in 1:D
            idx1 = Vec{1,Int}((dir,))
            idxD1 = Vec{D - 1,Int}(ntuple(d->d < dir ? d : d + 1, D - 1))
            @test length(sD1.comps) == D
            @test haskey(sD1.comps, idx1)
            @test length(sD1.comps[idx1]) == 1
            @test haskey(sD1.comps[idx1], idxD1)
            @test sD1.comps[idx1][idxD1] * sdf[idxD1] == ssdf[idx1]
        end

        s = bitsign(1 * (D - 1))
        for dir in 1:D
            idx1 = Vec{1,Int}((dir,))
            idxD1 = Vec{D - 1,Int}(ntuple(d->d < dir ? d : d + 1, D - 1))
            @test sD1.comps[idx1][idxD1] * s1.comps[idxD1][idx1] == s * I
        end

        # Derivatives

        d0 = deriv(Val(0), Val(false), dom)
        @test length(d0.comps) == D
        for dir in 1:D
            idx1 = Vec{1,Int}((dir,))
            @test haskey(d0.comps, idx1)
            @test length(d0.comps[idx1]) == 1
            @test haskey(d0.comps[idx1], idx0)
            @test d0.comps[idx1][idx0] * f[idx0] == df[idx1]
        end

        if D > 1
            d1 = deriv(Val(1), Val(false), dom)
            d10 = d1 * d0
            for (i, d10i) in d10.comps, (j, d10ij) in d10i
                @test all(d10ij .== 0)
            end
        end

        # Laplace
        l0 = laplace(Val(0), Val(false), dom)
        @test l0 == laplace1(Val(0), Val(false), dom)
    end
end

if runtests
    testForms()
end
