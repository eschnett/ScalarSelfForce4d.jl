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

    for D in 1:3
        for R in 0:D
            for Dual in false:true
                dom = Domain{D,BigRat}(3)
                z = zeros(Form{D,R,Dual,BigRat,BigRat}, dom)
                as = [BigRat(rand(-100:100)) for i in 1:10]
                xs = Form{D,R,Dual,BigRat,BigRat}[]
                for i in 1:10
                    comps = Dict{Vec{R,Int},Fun{D,BigRat,BigRat}}()
                    for staggeredc in CartesianIndices(ntuple(d->0:1, D))
                        staggered =
                            Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
                        if count(staggered) == R
                            idx = Vec{R,Int}(Tuple(staggered2idx(staggered)))
                            fdom = makestaggered(makedual(dom, Dual), staggered)
                            fvals = BigRat.(rand(-100:100, fdom.n.elts))
                            fun = Fun{D,BigRat,BigRat}(fdom, fvals)
                            comps[idx] = fun
                        end
                    end
                    f = Form(comps)
                    push!(xs, f)
                end
                testVectorspace(z, as, xs, isequal)
            end
        end
    end
    
    for D in 1:3, lorentzian in false:true
        for RI in 0:D, DualI in false:true
            for RJ in 0:D, DualJ in false:true
                dom = Domain{D,BigRat}(3, lorentzian = lorentzian)
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
                    as = [BigRat(rand(-100:100)) for i in 1:10]
                    testVectorspace(z, as, xs, isequal)
                end
            end
        end
    end

    @testset "Forms.Wedge D=$D" for D in 1:2:4
        dom = Domain{D,BigRat}(3)
    
        if D >= 2
            for dir1 in 1:D, dir2 in dir1 + 1:D, Dual in [false] # false:true
                f = zeros(Form{D,1,Dual,BigRat,BigRat}, dom)
                g = zeros(Form{D,1,Dual,BigRat,BigRat}, dom)
                h = zeros(Form{D,2,Dual,BigRat,BigRat}, dom)
                f.comps[Vec{1,Int}((dir1,))] = fconst(f[(dir1,)].dom, BigRat(1))
                g.comps[Vec{1,Int}((dir2,))] = fconst(g[(dir2,)].dom, BigRat(1))
                h.comps[Vec{2,Int}((dir1, dir2))] =
                    fconst(h[(dir1, dir2)].dom, BigRat(1))
                @test wedge(f, g) == h
            end
        end

        for RI in 0:D, RJ in 0:D - RI, Dual in [false] # false:true
    
            icomps = Dict{Vec{RI,Int},Fun{D,BigRat,BigRat}}()
            for staggeredc in CartesianIndices(ntuple(d->0:1, D))
                staggered = Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
                if count(staggered) == RI
                    idx = Vec{RI,Int}(Tuple(staggered2idx(staggered)))
                    fdom = makestaggered(makedual(dom, Dual), staggered)
                    fvals = BigRat.(rand(-100:100, fdom.n.elts))
                    fun = Fun{D,BigRat,BigRat}(fdom, fvals)
                    icomps[idx] = fun
                end
            end
            f = Form(icomps)
    
            jcomps = Dict{Vec{RJ,Int},Fun{D,BigRat,BigRat}}()
            for staggeredc in CartesianIndices(ntuple(d->0:1, D))
                staggered = Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
                if count(staggered) == RJ
                    idx = Vec{RJ,Int}(Tuple(staggered2idx(staggered)))
                    fdom = makestaggered(makedual(dom, Dual), staggered)
                    fvals = BigRat.(rand(-100:100, fdom.n.elts))
                    fun = Fun{D,BigRat,BigRat}(fdom, fvals)
                    jcomps[idx] = fun
                end
            end
            g = Form(jcomps)
    
            a = BigRat(rand(-100:100))
    
            @test isequal(wedge(a * f, g), a * wedge(f, g))
            @test isequal(wedge(f, a * g), a * wedge(f, g))
            if 2 * RI <= D && RI > 0
                @test iszero(wedge(f, f))
            end
            @test isequal(wedge(g, f), BigRat(bitsign(RI * RJ)) * wedge(f, g))
        end
    end

    @testset "Forms.Wedge D=$D" for D in 1:4
        dom = Domain{D,BigRat}(3)
        for R in 1:1
            for dir in 1:D
                domf = makestaggered(dom, unitvec(Val(D), dir))
                domg = makedual(dom, true)
                domh = makestaggered(dom, Vec{D,Bool}(ntuple(d->true, D)))
                f = zeros(Form{D,R,false,BigRat,BigRat}, domf)
                g = zeros(Form{D,D - R,true,BigRat,BigRat}, domg)
                h = zeros(Form{D,D,false,BigRat,BigRat}, domh)
                idxf = Vec{1,Int}((dir,))
                idxg = Vec{D - 1,Int}(Tuple(filter(!=(dir), 1:D)))
                idxh = Vec{D,Int}(Tuple(1:D))
                s = levicivita([idxf..., idxg...])
                f.comps[idxf] = fconst(f[idxf].dom, BigRat(1))
                g.comps[idxg] = fconst(g[idxg].dom, BigRat(1))
                h.comps[idxh] = fconst(h[idxh].dom, BigRat(1))
                @test wedge(f, g) == BigRat(s) * h

                RI = R
                DualI = false
                icomps = Dict{Vec{RI,Int},Fun{D,BigRat,BigRat}}()
                for staggeredc in CartesianIndices(ntuple(d->0:1, D))
                    staggered = Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
                    if count(staggered) == RI
                        idx = Vec{RI,Int}(Tuple(staggered2idx(staggered)))
                        fdom = makestaggered(makedual(dom, DualI), staggered)
                        fvals = BigRat.(rand(-100:100, fdom.n.elts))
                        fun = Fun{D,BigRat,BigRat}(fdom, fvals)
                        icomps[idx] = fun
                    end
                end
                f = Form(icomps)
    
                icomps = Dict{Vec{RI,Int},Fun{D,BigRat,BigRat}}()
                for staggeredc in CartesianIndices(ntuple(d->0:1, D))
                    staggered = Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
                    if count(staggered) == RI
                        idx = Vec{RI,Int}(Tuple(staggered2idx(staggered)))
                        fdom = makestaggered(makedual(dom, DualI), staggered)
                        fvals = BigRat.(rand(-100:100, fdom.n.elts))
                        fun = Fun{D,BigRat,BigRat}(fdom, fvals)
                        icomps[idx] = fun
                    end
                end
                f2 = Form(icomps)
    
                RJ = D - R
                DualJ = true
                jcomps = Dict{Vec{RJ,Int},Fun{D,BigRat,BigRat}}()
                for staggeredc in CartesianIndices(ntuple(d->0:1, D))
                    staggered = Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
                    if count(staggered) == RJ
                        idx = Vec{RJ,Int}(Tuple(staggered2idx(staggered)))
                        fdom = makestaggered(makedual(dom, DualJ), staggered)
                        fvals = BigRat.(rand(-100:100, fdom.n.elts))
                        fun = Fun{D,BigRat,BigRat}(fdom, fvals)
                        jcomps[idx] = fun
                    end
                end
                g = Form(jcomps)
    
                jcomps = Dict{Vec{RJ,Int},Fun{D,BigRat,BigRat}}()
                for staggeredc in CartesianIndices(ntuple(d->0:1, D))
                    staggered = Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
                    if count(staggered) == RJ
                        idx = Vec{RJ,Int}(Tuple(staggered2idx(staggered)))
                        fdom = makestaggered(makedual(dom, DualJ), staggered)
                        fvals = BigRat.(rand(-100:100, fdom.n.elts))
                        fun = Fun{D,BigRat,BigRat}(fdom, fvals)
                        jcomps[idx] = fun
                    end
                end
                g2 = Form(jcomps)
    
                a = BigRat(rand(-100:100))
    
                @test isequal(wedge(a * f, g), a * wedge(f, g))
                @test isequal(wedge(f, a * g), a * wedge(f, g))
                @test isequal(wedge(f + f2, g), wedge(f, g) + wedge(f2, g))
                @test isequal(wedge(f, g + g2), wedge(f, g) + wedge(f, g2))
                # if 2*RI <= D && RI > 0
                #     @test iszero(wedge(f, f))
                # end
                # @test isequal(wedge(g, f),
                #               BigRat(bitsign(RI * RJ)) * wedge(f, g))

            end
        end
    end

    @testset "Forms.Star D=$D" for D in 1:4
        atol = 100 * eps(1.0)
    
        sfsinpiD = star(fsinpi(D))
        ssfsinpiD = star(sfsinpiD)
        sssfsinpiD = star(ssfsinpiD)
        scale = bitsign(0 * (D - 0))
        maxerr = norm(fsinpi(D)[()] - scale * ssfsinpiD[()], Inf)
        @test isapprox(maxerr, 0; atol = atol)
        sidx = ntuple(d->d, D)
        maxerr = norm(sfsinpiD[sidx] - scale * sssfsinpiD[sidx], Inf)
        @test isapprox(maxerr, 0; atol = atol)
    
        sdfsinpiD = star(dfsinpi(D))
        ssdfsinpiD = star(sdfsinpiD)
        sssdfsinpiD = star(ssdfsinpiD)
        scale = bitsign(1 * (D - 1))
        for dir in 1:D
            maxerr = norm(dfsinpi(D)[(dir,)] - scale * ssdfsinpiD[(dir,)], Inf)
            @test isapprox(maxerr, 0; atol = atol)
            sidx = ntuple(d->d < dir ? d : d + 1, D - 1)
            maxerr = norm(sdfsinpiD[sidx] - scale * sssdfsinpiD[sidx], Inf)
            @test isapprox(maxerr, 0; atol = atol)
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
        err = norm(cdfsinpiD[()] - sdsdfsinpiD[()], Inf)
        @test isapprox(err, 0; atol = atol)

        lfsinpiD = laplace(fsinpi(D))
        err = norm(lfsinpiD[()] - sdsdfsinpiD[()], Inf)
        @test isapprox(err, 0; atol = atol)
    end

    @testset "Forms.Operators D=$D lorentzian=$lorentzian" for D in 1:4,
            lorentzian in [false]   # false:true
        BigRat = Rational{BigInt}
        dom = Domain{D,BigRat}(3, lorentzian = lorentzian)
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

testForms()
