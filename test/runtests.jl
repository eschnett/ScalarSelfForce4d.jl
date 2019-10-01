using Test


const Rat = Rational{Int64}
# const Rat = Rational{BigInt}
const ratrange = -10 : Rat(1//10) : 10

# Helpers
include("testVectorspace.jl")

# Unit tests
runtests = true
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
