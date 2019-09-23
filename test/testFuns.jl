using ScalarSelfForce4d.Funs

using Memoize



function sinpiD(x::Vec{D,T})::T where {D,T}
    prod(sinpi(x[d]) for d in 1:D)
end

function maxabsdiff(dom::Domain{D,T}, f1, f2)::T where {D,T <: Number}
    err = T(0)
    if D == 0
        v1 = f1(Vec(()))
        v2 = f2(Vec(()))
        dv = v1 - v2
        err = max(err, abs(dv))
    elseif D == 1
        for x in LinRange(dom.xmin[1], dom.xmax[1], 5 * dom.n[1] + 2)
            v1 = f1(Vec((x,)))
            v2 = f2(Vec((x,)))
            dv = v1 - v2
            err = max(err, abs(dv))
        end
    elseif D == 2
        for y in LinRange(dom.xmin[2], dom.xmax[2], 5 * dom.n[2] + 2),
            x in LinRange(dom.xmin[1], dom.xmax[1], 5 * dom.n[1] + 2)
            v1 = f1(Vec((x, y)))
            v2 = f2(Vec((x, y)))
            dv = v1 - v2
            err = max(err, abs(dv))
        end
    elseif D == 3
        for z in LinRange(dom.xmin[3], dom.xmax[3], 5 * dom.n[3] + 2),
            y in LinRange(dom.xmin[2], dom.xmax[2], 5 * dom.n[2] + 2),
            x in LinRange(dom.xmin[1], dom.xmax[1], 5 * dom.n[1] + 2)
            v1 = f1(Vec((x, y, z)))
            v2 = f2(Vec((x, y, z)))
            dv = v1 - v2
            err = max(err, abs(dv))
        end
    elseif D == 4
        for t in LinRange(dom.xmin[4], dom.xmax[4], 5 * dom.n[4] + 2),
            z in LinRange(dom.xmin[3], dom.xmax[3], 5 * dom.n[3] + 2),
            y in LinRange(dom.xmin[2], dom.xmax[2], 5 * dom.n[2] + 2),
            x in LinRange(dom.xmin[1], dom.xmax[1], 5 * dom.n[1] + 2)
            v1 = f1(Vec((x, y, z, t)))
            v2 = f2(Vec((x, y, z, t)))
            dv = v1 - v2
            err = max(err, abs(dv))
        end
    else
        @assert false
    end
    err::T
end



@memoize Dict function asinpi(D::Int)::Fun
    approximate(sinpiD, dom(D))
end



function testFuns()

    BigRat = Rational{BigInt}
    for D in 1:2
        dom = Domain{D,BigRat}(5)
        z = zeros(Fun{D,BigRat,BigRat}, dom)
        as = [BigRat(rand(-100:100)) for i in 1:100]
        xs = [Fun{D,BigRat,BigRat}(dom, BigRat.(rand(-100:100, dom.n.elts)))
              for i in 1:100]
        testVectorspace(z, as, xs, isequal)
    end

    @testset "Funs.approximate D=$D" for D in 1:3
        atol = 100 * eps(1.0)

        ac = approximate(x->1.0, dom(D))
        err = maxabsdiff(dom(D), x->1.0, ac)
        @test isapprox(err, 0; atol = atol)
        @test isapprox(sum(ac), 2^D; atol = atol)

        ax = approximate(x->x[1], dom(D))
        err = maxabsdiff(dom(D), x->x[1], ax)
        @test isapprox(err, 0; atol = atol)
        @test isapprox(sum(ax), 0; atol = atol)

        if D >= 2
            axy = approximate(x->x[1] * x[2], dom(D))
            err = maxabsdiff(dom(D), x->x[1] * x[2], axy)
            @test isapprox(err, 0; atol = atol)
            @test isapprox(sum(axy), 0; atol = atol)
        end

        if D >= 3
            axyz = approximate(x->x[1] * x[2] * x[3], dom(D))
            err = maxabsdiff(dom(D), x->x[1] * x[2] * x[3], axyz)
            @test isapprox(err, 0; atol = atol)
            @test isapprox(sum(axyz), 0; atol = atol)
        end

        for dual in [false]     # :true
            for staggerc in CartesianIndices(ntuple(d->0:1, D))
                stagger = Vec(ntuple(d->Bool(staggerc[d]), D))
                sdom = makestaggered(makedual(dom(D), dual), stagger)
                sasinpi = approximate(sinpiD, sdom)
                err = maxabsdiff(sdom, sinpiD, sasinpi)
                sasinpi2 = approximate(x -> sinpiD(x)^2, sdom)
                if D == 1
                    if stagger == Vec((false,))
                        @test isapprox(err, 0.028537112649043128; atol = 1.0e-6)
                    elseif stagger == Vec((true,))
                        @test 0.37 <= err < 0.38
                    else
                        @assert false
                    end
                elseif D == 2
                    if stagger == Vec((false, false))
                        @test isapprox(err, 0.0577555389039901; atol = 1.0e-6)
                    elseif stagger == Vec((true, false))
                        @test 0.38 <= err < 0.39
                    elseif stagger == Vec((false, true))
                        @test 0.38 <= err < 0.39
                    elseif stagger == Vec((true, true))
                        @test isapprox(err, 0.35716917278630295; atol = 1.0e-6)
                    else
                        @assert false
                    end
                elseif D == 3
                    if stagger == Vec((false, false, false))
                        @test isapprox(err, 0.08767328827754184; atol = 1.0e-6)
                    elseif stagger == Vec((true, false, false))
                        @test 0.39 <= err < 0.40
                    elseif stagger == Vec((false, true, false))
                        @test 0.39 <= err < 0.40
                    elseif stagger == Vec((true, true, false))
                        @test isapprox(err, 0.35215143552140193; atol = 1.0e-6)
                    elseif stagger == Vec((false, false, true))
                        @test 0.39 <= err < 0.40
                    elseif stagger == Vec((true, false, true))
                        @test isapprox(err, 0.3521514355214018; atol = 1.0e-6)
                    elseif stagger == Vec((false, true, true))
                        @test isapprox(err, 0.352151435521402; atol = 1.0e-6)
                    elseif stagger == Vec((true, true, true))
                        @test isapprox(err, 0.39012741117989913; atol = 1.0e-6)
                    else
                        @assert false
                    end
                else
                    @assert false
                end
                @test isapprox(sum(sasinpi), 0; atol = atol)
                @test isapprox(sum(sasinpi2), 1; atol = atol)
            end
        end
    end

    @testset "Funs.approximate_delta D=$D" for D in 1:3
        atol = 100 * eps(1.0)
        adelta = approximate_delta(dom(D), zeros(Vec{D,Float64}))
        sum_adelta = sum(adelta)
        @test isapprox(sum_adelta, 1; atol = atol)
    end
end

if runtests
    testFuns()
end
