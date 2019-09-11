"""
Miscallaneous utilities
"""
module Defs



export bitsign
"""
Inverse of signbit
"""
function bitsign(b::Bool)::Int
    b ? -1 : +1
end

export linear
"""
Linear interpolation
"""
function linear(x0::T, y0::U, x1::T, y1::U, x::T)::U where
        {T<:Number, U<:Number}
    U(x - x1) / U(x0 - x1) * y0 + U(x - x0) / U(x1 - x0) * y1
end

export characteristic
"""
Characteristic function
"""
function characteristic(::Type{U}, x0::T, x1::T, x::T)::U where
        {T<:Number, U<:Number}
    (x < x0 || x > x1) && return U(0)
    (x == x0 || x == x1) && return U(1)/2
    return U(1)
end

end
