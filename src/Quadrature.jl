"""
Numerical quadrature
"""
module Quadrature

using ..Defs



export quad

@generated function quad(f, ::Type{U},
                         xmin::NTuple{D,T}, xmax::NTuple{D,T},
                         n::NTuple{D,Int})::U where {D, T, U}
    quote
        s = zero(U)
        for i in CartesianIndices(n)
            # x = ntuple(d -> linear(T(0), xmin[d],
            #                        T(n[d]), xmax[d], i[d] - T(1)/2), D)
            x = tuple($(
                [:(linear(T(0), xmin[$d],
                          T(n[$d]), xmax[$d], i[$d] - T(1)/2))
                 for d in 1:D]...))
            s += f(x)
        end
        # w = U(prod(ntuple(d -> (xmax[d] - xmin[d]) / n[d], D)))
        w = U(*($(
            [:((xmax[$d] - xmin[$d]) / n[$d]) for d in 1:D]...)))
        w * s
    end
end

end
