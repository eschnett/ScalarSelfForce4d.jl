# [ScalarSelfForce4d.jl](https://github.com/eschnett/ScalarSelfForce4d.jl)

* [GitHub](https://github.com/eschnett/ScalarSelfForce4d.jl): Source code repository
* [Azure
  Pipelines](https://dev.azure.com/schnetter/ScalarSelfForce4d.jl/_build):
  Build Status [![Build
  Status](https://dev.azure.com/schnetter/ScalarSelfForce4d.jl/_apis/build/status/eschnett.ScalarSelfForce4d.jl?branchName=master)](https://dev.azure.com/schnetter/ScalarSelfForce4d.jl/_build/latest?definitionId=1&branchName=master)



## Ideas

* Can we write Seth's Einstein equations with geometric algebra?



## Plotting with Gadfly

```Julia
using Gadfly
plot([x->del(Vec((x,0.0)))/1e3, x->4pi*pot(Vec((x,0.0))), x->max(-10, -1/abs(x))], -1, 1, color=["delta", "potential", "1/r"])
plot((x,y)->pot(Vec((Double64(x),Double64(y)))), -1, 1, -1, 1)
# Coord.cartesian(fixed=true)
```



## Literature

* Marko Seslija, Arjan van der Schaft, Jacquelien M. A. Scherpen:
  Discrete Exterior Geometry Approach to Structure-Preserving
  Discretization of Distributed-Parameter Port-Hamiltonian Systems.
  Journal of Geometry and Physics,
  http://dx.doi.org/10.1016/j.geomphys.2012.02.006,
  https://arxiv.org/abs/1111.6403 [math-ph].
* Keenan Crane: Discrete differentical geometry: An applied
  introduction. Lecture 6: Discrete exterior calculus. CMU
  15-458/858B, Fall 2017.
  http://brickisland.net/DDGFall2017/wp-content/uploads/2017/09/CMU_DDG_Fall2017_06_DiscreteExteriorCalculus.pdf
* Lecture 2: The simplicial complex. http://brickisland.net/DDGFall2017/wp-content/uploads/2017/09/CMU_DDG_Fall2017_02_SimpicialComplex-1.pdf

* Douglas N. Arnold: Finite element exterior calculus.
  http://www-users.math.umn.edu/~arnold/feec-cbms/index.html
* Nathan Bell, Anil N. Hirani: PyDEC: Software and Algorithms for
  Discretization of Exterior Calculus. arXiv:1103.3076 [cs.NA].
* https://fenicsproject.org : large, complex package, probably only 2d
  and 3d.
* https://github.com/hirani/pydec : python, arbitrary dimension, DEC:
  seems good! unmaintained?

* Ari Stern, https://www.newton.ac.uk/seminar/20190930143015301,
  minute 20: hamiltonian, conserved symplected 2-form
* Melvin Leok, https://www.newton.ac.uk/seminar/20191001110012001 :
  variational integrators, spacetime discretizations, multisymplectic
  relations, Lie groups, etc.
* Snorre Christiansen,
  https://www.newton.ac.uk/seminar/20191001143015301 : modify
  langangian to be invariant under group action. Finite elements for
  manifolds with curvature, from first steps, different (?) from cell
  complexes, inspired by elasticity.
* Anil Hirani, https://www.newton.ac.uk/seminar/20191002093010301 :
  similar to de-Rham complex, but D^2 = Riemann instead of 0; basic
  exposition of DEC. definition of curvature on a simplex by moving
  edges/faces to common corner.
