# [ScalarSelfForce4d.jl](https://github.com/eschnett/ScalarSelfForce4d.jl)

* [GitHub](https://github.com/eschnett/ScalarSelfForce4d.jl): Source code repository
* [Azure
  Pipelines](https://dev.azure.com/schnetter/ScalarSelfForce4d.jl/_build):
  Build Status [![Build
  Status](https://dev.azure.com/schnetter/ScalarSelfForce4d.jl/_apis/build/status/eschnett.ScalarSelfForce4d.jl?branchName=master)](https://dev.azure.com/schnetter/ScalarSelfForce4d.jl/_build/latest?definitionId=1&branchName=master)



## Plotting with Gadfly

```Julia
using Gadfly
plot([x->del(Vec((x,0.0)))/1e3, x->4pi*pot(Vec((x,0.0))), x->max(-10, -1/abs(x))], -1, 1, color=["delta", "potential", "1/r"])
plot((x,y)->pot(Vec((Double64(x),Double64(y)))), -1, 1, -1, 1)
# Coord.cartesian(fixed=true)
```
