using DocumentFormat

function format!(sourcefile)
    src = String(read(sourcefile))
    res = DocumentFormat.format(src)
    write(sourcefile, res)
end

sourcefiles = [
    "format.jl",
    "src/Bases.jl",
    "src/Defs.jl",
    "src/Domains.jl",
    "src/Forms.jl",
    "src/Funs.jl",
    "src/Ops.jl",
    "src/Quadrature.jl",
    "src/ScalarSelfForce4d.jl",
    "src/Vecs.jl",
    "test/runtests.jl",
    "test/testBases.jl",
    "test/testDefs.jl",
    "test/testDomains.jl",
    "test/testForms.jl",
    "test/testFuns.jl",
    "test/testOps.jl",
    "test/testPoisson.jl",
    "test/testQuadrature.jl",
    "test/testVecs.jl",
    "test/testVectorspace.jl",
]

format!.(sourcefiles)
