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

#TODO     for D in 1:3
#TODO         for R in 0:D
#TODO             for Dual in false:true
#TODO                 dom = Domain{D,BigRat}(3)
#TODO                 z = zeros(Form{D,R,Dual,BigRat,BigRat}, dom)
#TODO                 as = [BigRat(rand(-100:100)) for i in 1:10]
#TODO                 xs = Form{D,R,Dual,BigRat,BigRat}[]
#TODO                 for i in 1:10
#TODO                     comps = Dict{Vec{R,Int},Fun{D,BigRat,BigRat}}()
#TODO                     for staggeredc in CartesianIndices(ntuple(d->0:1, D))
#TODO                         staggered =
#TODO                             Vec{D,Bool}(ntuple(d->Bool(staggeredc[d]), D))
#TODO                         if count(staggered) == R
#TODO                             idx = Vec{R,Int}(Tuple(staggered2idx(staggered)))
#TODO                             fdom = makestaggered(makedual(dom, Dual), staggered)
#TODO                             fvals = BigRat.(rand(-100:100, fdom.n.elts))
#TODO                             fun = Fun{D,BigRat,BigRat}(fdom, fvals)
#TODO                             comps[idx] = fun
#TODO                         end
#TODO                     end
#TODO                     f = Form(comps)
#TODO                     push!(xs, f)
#TODO                 end
#TODO                 testVectorspace(z, as, xs, isequal)
#TODO             end
#TODO         end
#TODO     end
#TODO     
#TODO     for D in 1:3, lorentzian in false:true
#TODO         for RI in 0:D, DualI in false:true
#TODO             for RJ in 0:D, DualJ in false:true
#TODO                 dom = Domain{D,BigRat}(3, lorentzian = lorentzian)
#TODO                 z = zeros(FOp{D,RI,DualI,RJ,DualJ,BigRat,BigRat})
#TODO                 xs = FOp{D,RI,DualI,RJ,DualJ,BigRat,BigRat}[]
#TODO                 if RI == RJ && DualI == DualJ
#TODO                     push!(xs, one(FOp{D,RI,DualI,RJ,DualJ,BigRat,BigRat}, dom))
#TODO                 end
#TODO                 if RI == D - RJ && DualI == !DualJ
#TODO                     push!(xs, star(Val(RJ), Val(DualJ), dom))
#TODO                 end
#TODO                 if RI == RJ + 1 && DualI == DualJ
#TODO                     push!(xs, deriv(Val(RJ), Val(DualJ), dom))
#TODO                 end
#TODO                 if RI == D - (RJ + 1) && DualI == !DualJ
#TODO                     d = deriv(Val(RJ), Val(DualJ), dom)
#TODO                     s = star(Val(RJ + 1), Val(DualJ), dom)
#TODO                     push!(xs, s * d)
#TODO                 end
#TODO                 if RI == (D - RJ) + 1 && DualI == !DualJ
#TODO                     s = star(Val(RJ), Val(DualJ), dom)
#TODO                     d = deriv(Val(D - RJ), Val(!DualJ), dom)
#TODO                     push!(xs, d * s)
#TODO                 end
#TODO                 if RI == RJ - 1 && DualI == DualJ
#TODO                     push!(xs, coderiv(Val(RJ), Val(DualJ), dom))
#TODO                 end
#TODO                 if RI == D - (RJ - 1) && DualI == !DualJ
#TODO                     c = coderiv(Val(RJ), Val(DualJ), dom)
#TODO                     s = star(Val(RJ - 1), Val(DualJ), dom)
#TODO                     push!(xs, s * c)
#TODO                 end
#TODO                 if RI == (D - RJ) - 1 && DualI == !DualJ
#TODO                     s = star(Val(RJ), Val(DualJ), dom)
#TODO                     c = coderiv(Val(D - RJ), Val(!DualJ), dom)
#TODO                     push!(xs, c * s)
#TODO                 end
#TODO                 if !isempty(xs)
#TODO                     as = [BigRat(rand(-100:100)) for i in 1:10]
#TODO                     testVectorspace(z, as, xs, isequal)
#TODO                 end
#TODO             end
#TODO         end
#TODO     end

#TODO     @testset "Forms.Wedge D=$D" for D in 1:2   #TODO :4
#TODO         dom = Domain{D,BigRat}(3)
#TODO     
#TODO         if D >= 2
#TODO             for dir1 in 1:D, dir2 in dir1+1:D, Dual in [false] # false:true
#TODO                 f = zeros(Form{D,1,Dual,BigRat,BigRat}, dom)
#TODO                 g = zeros(Form{D,1,Dual,BigRat,BigRat}, dom)
#TODO                 h = zeros(Form{D,2,Dual,BigRat,BigRat}, dom)
#TODO                 f.comps[Vec{1,Int}((dir1,))] = fconst(f[(dir1,)].dom, BigRat(1))
#TODO                 g.comps[Vec{1,Int}((dir2,))] = fconst(g[(dir2,)].dom, BigRat(1))
#TODO                 h.comps[Vec{2,Int}((dir1,dir2))] =
#TODO                     fconst(h[(dir1,dir2)].dom, BigRat(1))
#TODO                 @test wedge(f, g) == h
#TODO             end
#TODO         end
#TODO 
#TODO         for RI in 0:D, RJ in 0:D-RI, Dual in [false] # false:true
#TODO     
#TODO             icomps = Dict{Vec{RI,Int}, Fun{D,BigRat,BigRat}}()
#TODO             for staggeredc in CartesianIndices(ntuple(d -> 0:1, D))
#TODO                 staggered = Vec{D,Bool}(ntuple(d -> Bool(staggeredc[d]), D))
#TODO                 if count(staggered) == RI
#TODO                     idx = Vec{RI,Int}(Tuple(staggered2idx(staggered)))
#TODO                     fdom = makestaggered(makedual(dom, Dual), staggered)
#TODO                     fvals = BigRat.(rand(-100:100, fdom.n.elts))
#TODO                     fun = Fun{D,BigRat,BigRat}(fdom, fvals)
#TODO                     icomps[idx] = fun
#TODO                 end
#TODO             end
#TODO             f = Form(icomps)
#TODO     
#TODO             jcomps = Dict{Vec{RJ,Int}, Fun{D,BigRat,BigRat}}()
#TODO             for staggeredc in CartesianIndices(ntuple(d -> 0:1, D))
#TODO                 staggered = Vec{D,Bool}(ntuple(d -> Bool(staggeredc[d]), D))
#TODO                 if count(staggered) == RJ
#TODO                     idx = Vec{RJ,Int}(Tuple(staggered2idx(staggered)))
#TODO                     fdom = makestaggered(makedual(dom, Dual), staggered)
#TODO                     fvals = BigRat.(rand(-100:100, fdom.n.elts))
#TODO                     fun = Fun{D,BigRat,BigRat}(fdom, fvals)
#TODO                     jcomps[idx] = fun
#TODO                 end
#TODO             end
#TODO             g = Form(jcomps)
#TODO     
#TODO             a = BigRat(rand(-100:100))
#TODO     
#TODO             @test isequal(wedge(a * f, g), a * wedge(f, g))
#TODO             @test isequal(wedge(f, a * g), a * wedge(f, g))
#TODO             if 2*RI <= D && RI > 0
#TODO                 @test iszero(wedge(f, f))
#TODO             end
#TODO             @test isequal(wedge(g, f), BigRat(bitsign(RI * RJ)) * wedge(f, g))
#TODO         end
#TODO     end

    @testset "Forms.Wedge D=$D" for D in 1:4
        dom = Domain{D,BigRat}(3)
        for R in 1:1
            for dir in 1:D
                domf = makestaggered(dom, unitvec(Val(D), dir))
                domg = makedual(dom, true)
                domh = makestaggered(dom, Vec{D,Bool}(ntuple(d -> true, D)))
                f = zeros(Form{D,R,false,BigRat,BigRat}, domf)
                g = zeros(Form{D,D-R,true,BigRat,BigRat}, domg)
                h = zeros(Form{D,D,false,BigRat,BigRat}, domh)
                idxf = Vec{1,Int}((dir,))
                idxg = Vec{D-1,Int}(Tuple(filter(!=(dir), 1:D)))
                idxh = Vec{D,Int}(Tuple(1:D))
                s = levicivita([idxf..., idxg...])
                f.comps[idxf] = fconst(f[idxf].dom, BigRat(1))
                g.comps[idxg] = fconst(g[idxg].dom, BigRat(1))
                h.comps[idxh] = fconst(h[idxh].dom, BigRat(1))
                @test wedge(f, g) == BigRat(s) * h

                RI = R
                DualI = false
                icomps = Dict{Vec{RI,Int}, Fun{D,BigRat,BigRat}}()
                for staggeredc in CartesianIndices(ntuple(d -> 0:1, D))
                    staggered = Vec{D,Bool}(ntuple(d -> Bool(staggeredc[d]), D))
                    if count(staggered) == RI
                        idx = Vec{RI,Int}(Tuple(staggered2idx(staggered)))
                        fdom = makestaggered(makedual(dom, DualI), staggered)
                        fvals = BigRat.(rand(-100:100, fdom.n.elts))
                        fun = Fun{D,BigRat,BigRat}(fdom, fvals)
                        icomps[idx] = fun
                    end
                end
                f = Form(icomps)
    
                icomps = Dict{Vec{RI,Int}, Fun{D,BigRat,BigRat}}()
                for staggeredc in CartesianIndices(ntuple(d -> 0:1, D))
                    staggered = Vec{D,Bool}(ntuple(d -> Bool(staggeredc[d]), D))
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
                jcomps = Dict{Vec{RJ,Int}, Fun{D,BigRat,BigRat}}()
                for staggeredc in CartesianIndices(ntuple(d -> 0:1, D))
                    staggered = Vec{D,Bool}(ntuple(d -> Bool(staggeredc[d]), D))
                    if count(staggered) == RJ
                        idx = Vec{RJ,Int}(Tuple(staggered2idx(staggered)))
                        fdom = makestaggered(makedual(dom, DualJ), staggered)
                        fvals = BigRat.(rand(-100:100, fdom.n.elts))
                        fun = Fun{D,BigRat,BigRat}(fdom, fvals)
                        jcomps[idx] = fun
                    end
                end
                g = Form(jcomps)
    
                jcomps = Dict{Vec{RJ,Int}, Fun{D,BigRat,BigRat}}()
                for staggeredc in CartesianIndices(ntuple(d -> 0:1, D))
                    staggered = Vec{D,Bool}(ntuple(d -> Bool(staggeredc[d]), D))
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

#TODO     @testset "Forms.Star D=$D" for D in 1:4
#TODO         atol = 100 * eps(1.0)
#TODO     
#TODO         sfsinpiD = star(fsinpi(D))
#TODO         ssfsinpiD = star(sfsinpiD)
#TODO         sssfsinpiD = star(ssfsinpiD)
#TODO         scale = bitsign(0 * (D - 0))
#TODO         maxerr = norm(fsinpi(D)[()] - scale * ssfsinpiD[()], Inf)
#TODO         @test isapprox(maxerr, 0; atol = atol)
#TODO         sidx = ntuple(d->d, D)
#TODO         maxerr = norm(sfsinpiD[sidx] - scale * sssfsinpiD[sidx], Inf)
#TODO         @test isapprox(maxerr, 0; atol = atol)
#TODO     
#TODO         sdfsinpiD = star(dfsinpi(D))
#TODO         ssdfsinpiD = star(sdfsinpiD)
#TODO         sssdfsinpiD = star(ssdfsinpiD)
#TODO         scale = bitsign(1 * (D - 1))
#TODO         for dir in 1:D
#TODO             maxerr = norm(dfsinpi(D)[(dir,)] - scale * ssdfsinpiD[(dir,)], Inf)
#TODO             @test isapprox(maxerr, 0; atol = atol)
#TODO             sidx = ntuple(d->d < dir ? d : d + 1, D - 1)
#TODO             maxerr = norm(sdfsinpiD[sidx] - scale * sssdfsinpiD[sidx], Inf)
#TODO             @test isapprox(maxerr, 0; atol = atol)
#TODO         end
#TODO     end
#TODO 
#TODO     @testset "Forms.Derivatives D=$D" for D in 1:3
#TODO         atol = (D <= 3 ? 100 : 1000) * eps(1.0)
#TODO 
#TODO         # Constants have zero derivative
#TODO         ac = approximate(x->1.0, dom(D))
#TODO         fac = Form(Dict(() => ac))
#TODO         dfac = deriv(fac)
#TODO         for dir in 1:D
#TODO             err = maxabsdiff(dom(D), x->0.0, dfac[(dir,)])
#TODO             @test isapprox(err, 0; atol = atol)
#TODO         end
#TODO 
#TODO         # Linear functions have unit derivative
#TODO         for dir in 1:D
#TODO             ax = approximate(x->x[dir], dom(D))
#TODO             fax = Form(Dict(() => ax))
#TODO             dfax = deriv(fax)
#TODO             for d in 1:D
#TODO                 err = maxabsdiff(dom(D), x->d == dir ? 1.0 : 0.0, dfax[(d,)])
#TODO                 @test isapprox(err, 0; atol = atol)
#TODO             end
#TODO         end
#TODO 
#TODO         # TODO: linearity, product rule, (chain rule)
#TODO 
#TODO         fsinpi(D) = Form(Dict(() => asinpi(D)))
#TODO         dfsinpi(D) = deriv(fsinpi(D))
#TODO         for dir in 1:D
#TODO             domDx = makestaggered(dom(D), unitvec(Val(D), dir))
#TODO             adsinpix = approximate(x->dsinpiD(x, dir), domDx)
#TODO             err = norm(dfsinpi(D).comps[Vec((dir,))] - adsinpix, Inf)
#TODO             if D == 1
#TODO                 @test isapprox(err, 0.03683663404089432; atol = 1.0e-6)
#TODO             elseif D == 2
#TODO                 @test isapprox(err, 0.03879219734864847; atol = 1.0e-6)
#TODO             elseif D == 3
#TODO                 @test isapprox(err, 0.04085157654377347; atol = 1.0e-6)
#TODO             elseif D == 4
#TODO                 @test isapprox(err, 0.04302028294795979; atol = 1.0e-6)
#TODO             else
#TODO                 @assert false
#TODO             end
#TODO         end
#TODO         if D > 1
#TODO             ddfsinpi = deriv(dfsinpi(D))
#TODO             for dir1 in 1:D, dir2 in dir1 + 1:D
#TODO                 idx12 = Vec{2,Int}((dir1, dir2))
#TODO                 maxerr = norm(ddfsinpi[idx12], Inf)
#TODO                 @test isapprox(maxerr, 0; atol = atol)
#TODO             end
#TODO         end
#TODO     end
#TODO 
#TODO     @testset "Forms.Laplacian D=$D" for D in 1:4
#TODO         atol = 100 * eps(1.0)
#TODO 
#TODO         sdfsinpiD = star(dfsinpi(D))
#TODO         dsdfsinpiD = deriv(sdfsinpiD)
#TODO         sdsdfsinpiD = star(dsdfsinpiD)
#TODO         ad2sinpiD = approximate(lsinpiD, dom(D))
#TODO         err = norm(sdsdfsinpiD[()] - ad2sinpiD, Inf)
#TODO         if D == 1
#TODO             @test isapprox(err, 0.6047119237688179; atol = 1.0e-6)
#TODO         elseif D == 2
#TODO             @test isapprox(err, 0.9781685656156345; atol = 1.0e-6)
#TODO         elseif D == 3
#TODO             @test isapprox(err, 1.5317929733380282; atol = 1.0e-6)
#TODO         elseif D == 4
#TODO             @test isapprox(err, 2.150815960478276; atol = 1.0e-6)
#TODO         else
#TODO             @assert false
#TODO         end
#TODO 
#TODO         cdfsinpiD = coderiv(dfsinpi(D))
#TODO         err = norm(cdfsinpiD[()] - sdsdfsinpiD[()], Inf)
#TODO         @test isapprox(err, 0; atol = atol)
#TODO 
#TODO         lfsinpiD = laplace(fsinpi(D))
#TODO         err = norm(lfsinpiD[()] - sdsdfsinpiD[()], Inf)
#TODO         @test isapprox(err, 0; atol = atol)
#TODO     end
#TODO 
#TODO     @testset "Forms.Operators D=$D lorentzian=$lorentzian" for D in 1:4,
#TODO             lorentzian in [false]   # false:true
#TODO         BigRat = Rational{BigInt}
#TODO         dom = Domain{D,BigRat}(3, lorentzian = lorentzian)
#TODO         f = Form(Dict(() => Fun(dom, BigRat.(rand(-100:100, dom.n.elts)))))
#TODO 
#TODO         # Star
#TODO 
#TODO         idx0 = Vec{0,Int}(())
#TODO         idxD = Vec{D,Int}(ntuple(d->d, D))
#TODO 
#TODO         s0 = star(Val(0), Val(false), dom)
#TODO         sf = star(f)
#TODO         @test length(s0.comps) == 1
#TODO         @test haskey(s0.comps, idxD)
#TODO         @test length(s0.comps[idxD]) == 1
#TODO         @test haskey(s0.comps[idxD], idx0)
#TODO         s0f = s0 * f
#TODO         @test s0f[idxD] == sf[idxD]
#TODO 
#TODO         sD = star(Val(D), Val(true), makedual(dom, !dom.dual))
#TODO         ssf = star(sf)
#TODO         @test length(sD.comps) == 1
#TODO         @test haskey(sD.comps, idx0)
#TODO         @test length(sD.comps[idx0]) == 1
#TODO         @test haskey(sD.comps[idx0], idxD)
#TODO         @test sD.comps[idx0][idxD] * sf[idxD] == ssf[idx0]
#TODO 
#TODO         sD0 = sD * s0
#TODO         @test sD0.comps[idx0][idx0] == I
#TODO 
#TODO         df = deriv(f)
#TODO         sdf = star(df)
#TODO         s1 = star(Val(1), Val(false), dom)
#TODO         @test length(s1.comps) == D
#TODO         for dir in 1:D
#TODO             idx1 = Vec{1,Int}((dir,))
#TODO             idxD1 = Vec{D - 1,Int}(ntuple(d->d < dir ? d : d + 1, D - 1))
#TODO             @test haskey(s1.comps, idxD1)
#TODO             @test length(s1.comps[idxD1]) == 1
#TODO             @test haskey(s1.comps[idxD1], idx1)
#TODO             @test s1.comps[idxD1][idx1] * df[idx1] == sdf[idxD1]
#TODO         end
#TODO 
#TODO         sD1 = star(Val(D - 1), Val(true), makedual(dom, !dom.dual))
#TODO         ssdf = star(sdf)
#TODO         for dir in 1:D
#TODO             idx1 = Vec{1,Int}((dir,))
#TODO             idxD1 = Vec{D - 1,Int}(ntuple(d->d < dir ? d : d + 1, D - 1))
#TODO             @test length(sD1.comps) == D
#TODO             @test haskey(sD1.comps, idx1)
#TODO             @test length(sD1.comps[idx1]) == 1
#TODO             @test haskey(sD1.comps[idx1], idxD1)
#TODO             @test sD1.comps[idx1][idxD1] * sdf[idxD1] == ssdf[idx1]
#TODO         end
#TODO 
#TODO         s = bitsign(1 * (D - 1))
#TODO         for dir in 1:D
#TODO             idx1 = Vec{1,Int}((dir,))
#TODO             idxD1 = Vec{D - 1,Int}(ntuple(d->d < dir ? d : d + 1, D - 1))
#TODO             @test sD1.comps[idx1][idxD1] * s1.comps[idxD1][idx1] == s * I
#TODO         end
#TODO 
#TODO         # Derivatives
#TODO 
#TODO         d0 = deriv(Val(0), Val(false), dom)
#TODO         @test length(d0.comps) == D
#TODO         for dir in 1:D
#TODO             idx1 = Vec{1,Int}((dir,))
#TODO             @test haskey(d0.comps, idx1)
#TODO             @test length(d0.comps[idx1]) == 1
#TODO             @test haskey(d0.comps[idx1], idx0)
#TODO             @test d0.comps[idx1][idx0] * f[idx0] == df[idx1]
#TODO         end
#TODO 
#TODO         if D > 1
#TODO             d1 = deriv(Val(1), Val(false), dom)
#TODO             d10 = d1 * d0
#TODO             for (i, d10i) in d10.comps, (j, d10ij) in d10i
#TODO                 @test all(d10ij .== 0)
#TODO             end
#TODO         end
#TODO 
#TODO         # Laplace
#TODO         l0 = laplace(Val(0), Val(false), dom)
#TODO         @test l0 == laplace1(Val(0), Val(false), dom)
#TODO     end
end

testForms()
