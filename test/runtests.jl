using Test

# Helpers
include("testVectorspace.jl")

# Unit tests
include("testDefs.jl")
include("testQuadrature.jl")
include("testVecs.jl")
include("testDomains.jl")
include("testBases.jl")
include("testFuns.jl")
include("testOps.jl")
include("testForms.jl")

# Integration tests
include("testPoisson.jl")


using ScalarSelfForce4d



#TODO const dom2 = Domain{2,Float64}(9)
#TODO const sdom2 = makestaggered(dom2)
#TODO 
#TODO @testset "Energy density of scalar wave" begin
#TODO     phi = approximate(waveD, dom2)
#TODO     eps = scalarwave_energy(phi)
#TODO 
#TODO     epsD1(x) = epsD(sdom2, x)
#TODO     eps0 = approximate(epsD1, sdom2)
#TODO     err = eps - eps0
#TODO     maxerr = norm(err, Inf)
#TODO     @test 0.21 <= maxerr < 0.22
#TODO end



#TODO const dom4 = Domain{4,Float64}(9, lorentzian=true)
#TODO const sdom4 = makestaggered(dom4)

#TODO @testset "Energy density of scalar wave" begin
#TODO     phi = approximate(waveD, dom4)
#TODO     eps = scalarwave_energy(phi)
#TODO 
#TODO     epsD1(x) = epsD(sdom4, x)
#TODO     eps0 = approximate(epsD1, sdom4)
#TODO     err = eps - eps0
#TODO     @show maxerr = norm(err, Inf)
#TODO     @test 0.046 <= maxerr < 0.047
#TODO end
