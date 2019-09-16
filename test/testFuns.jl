using ScalarSelfForce4d.Funs



function sinpiD(x::Vec{D,T})::T where {D,T}
    prod(sinpi(x[d]) for d in 1:D)
end

function maxabsdiff(dom::Domain{D,T}, f1, f2)::T where {D, T<:Number}
    err = T(0)
    if D == 0
        v1 = f1(Vec(()))
        v2 = f2(Vec(()))
        dv = v1 - v2
        err = max(err, abs(dv))
    elseif D == 1
        for x in LinRange(dom.xmin[1], dom.xmax[1], floor(Int, pi*dom.n[1]))
            v1 = f1(Vec((x,)))
            v2 = f2(Vec((x,)))
            dv = v1 - v2
            err = max(err, abs(dv))
        end
    elseif D == 2
        for y in LinRange(dom.xmin[2], dom.xmax[2], floor(Int, pi*dom.n[2])),
            x in LinRange(dom.xmin[1], dom.xmax[1], floor(Int, pi*dom.n[1]))
            v1 = f1(Vec((x, y)))
            v2 = f2(Vec((x, y)))
            dv = v1 - v2
            err = max(err, abs(dv))
        end
    elseif D == 3
        for z in LinRange(dom.xmin[3], dom.xmax[3], floor(Int, pi*dom.n[3])),
            y in LinRange(dom.xmin[2], dom.xmax[2], floor(Int, pi*dom.n[2])),
            x in LinRange(dom.xmin[1], dom.xmax[1], floor(Int, pi*dom.n[1]))
            v1 = f1(Vec((x, y, z)))
            v2 = f2(Vec((x, y, z)))
            dv = v1 - v2
            err = max(err, abs(dv))
        end
    elseif D == 4
        for t in LinRange(dom.xmin[4], dom.xmax[4], floor(Int, pi*dom.n[4])),
            z in LinRange(dom.xmin[3], dom.xmax[3], floor(Int, pi*dom.n[3])),
            y in LinRange(dom.xmin[2], dom.xmax[2], floor(Int, pi*dom.n[2])),
            x in LinRange(dom.xmin[1], dom.xmax[1], floor(Int, pi*dom.n[1]))
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



const asinpi = Dict{Int, Fun}()



function testFuns()

    idom1 = Domain{1,Float64}(5)
    z = zeros(Fun{1,Float64,Int}, idom1)
    xs = [Fun{1,Float64,Int}(idom1, rand(-100:100, 5)) for i in 1:100]
    testVectorspace(z, xs, isequal)

    idom2 = Domain{2,Float64}(5)
    z = zeros(Fun{2,Float64,Int}, idom2)
    xs = [Fun{2,Float64,Int}(idom2, rand(-100:100, 5,5)) for i in 1:100]
    testVectorspace(z, xs, isequal)

    @testset "Funs.Approximation D=$D" for D in 1:3
        atol = 100 * eps(1.0)

        ac = approximate(x->1.0, dom[D])
        err = maxabsdiff(dom[D], x->1.0, ac)
        @test isapprox(err, 0; atol=atol)

        ax = approximate(x->x[1], dom[D])
        err = maxabsdiff(dom[D], x->x[1], ax)
        @test isapprox(err, 0; atol=atol)

        if D >= 2
            axy = approximate(x->x[1]*x[2], dom[D])
            err = maxabsdiff(dom[D], x->x[1]*x[2], axy)
            @test isapprox(err, 0; atol=atol)
        end

        if D >= 3
            axyz = approximate(x->x[1]*x[2]*x[3], dom[D])
            err = maxabsdiff(dom[D], x->x[1]*x[2]*x[3], axyz)
            @test isapprox(err, 0; atol=atol)
        end

        asinpi[D] = approximate(sinpiD, dom[D])
        err = maxabsdiff(dom[D], sinpiD, asinpi[D])
        if D == 1
            @test 0.031 <= err < 0.032
        elseif D == 2
            @test 0.064 <= err < 0.065
        elseif D == 3
            @test 0.098 <= err < 0.099
        else
            @assert false
        end

        for dual in [false]     # :true
            for staggerc in CartesianIndices(ntuple(d -> 0:1, D))
                stagger = Vec(ntuple(d -> Bool(staggerc[d]), D))
                sdom = makestaggered(dual ? makedual(dom[D]) : dom[D], stagger)
                sasinpi = approximate(sinpiD, sdom)
                err = maxabsdiff(sdom, sinpiD, sasinpi)
                if D == 1
                    if stagger == Vec((false,))
                        @test 0.031 <= err < 0.032
                    elseif stagger == Vec((true,))
                        @test 0.37 <= err < 0.38
                    else
                        @assert false
                    end
                elseif D == 2
                    if stagger == Vec((false, false))
                        @test 0.064 <= err < 0.065
                    elseif stagger == Vec((true, false))
                        @test 0.38 <= err < 0.39
                    elseif stagger == Vec((false, true))
                        @test 0.38 <= err < 0.39
                    elseif stagger == Vec((true, true))
                        @test 0.37 <= err < 0.38
                    else
                        @assert false
                    end
                elseif D == 3
                    if stagger == Vec((false, false, false))
                        @test 0.098 <= err < 0.099
                    elseif stagger == Vec((true, false, false))
                        @test 0.39 <= err < 0.40
                    elseif stagger == Vec((false, true, false))
                        @test 0.39 <= err < 0.40
                    elseif stagger == Vec((true, true, false))
                        @test 0.36 <= err < 0.37
                    elseif stagger == Vec((false, false, true))
                        @test 0.39 <= err < 0.40
                    elseif stagger == Vec((true, false, true))
                        @test 0.36 <= err < 0.37
                    elseif stagger == Vec((false, true, true))
                        @test 0.36 <= err < 0.37
                    elseif stagger == Vec((true, true, true))
                        @test 0.40 <= err < 0.41
                    else
                        @assert false
                    end
                else
                    @assert false
                end
            end
        end
    end
end

testFuns()
