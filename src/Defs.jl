"""
Miscallaneous utilities
"""
module Defs

using Base.Order



export @swap!
macro swap!(x, y)
    quote
      $(esc(x)), $(esc(y)) = $(esc(y)), $(esc(x))
    end
end



export isstrictlysorted
function isstrictlysorted(itr, order::Ordering)
    y = iterate(itr)
    y === nothing && return true
    this, state = y
    while true
        y = iterate(itr, state)
        y === nothing && return true
        prev = this
        this, state = y
        lt(order, prev, this) || return false
    end
end
function isstrictlysorted(itr;
                          lt=isless, by=identity,
                          rev::Union{Bool,Nothing}=nothing,
                          order::Ordering=Forward)
    isstrictlysorted(itr, ord(lt,by,rev,order))
end



export bitsign
"""
Inverse of signbit
"""
function bitsign(b::Bool)::Int
    # b ? -1 : +1
    1 - 2 * b
end

function bitsign(b::I)::I where {I <: Signed}
    I(bitsign(isodd(b)))
end



export linear
"""
Linear interpolation
"""
function linear(x0::T, y0::U, x1::T, y1::U, x::T)::U where
        {T <: Number,U <: Number}
    U(x - x1) / U(x0 - x1) * y0 + U(x - x0) / U(x1 - x0) * y1
end



export characteristic
"""
Characteristic function
"""
function characteristic(x0::T, y0::U, x1::T, y1::U, x::T)::U where
        {T <: Number,U <: Number}
    x0 < x < x1 && return U(1)
    x == x0 && return y0
    x == x1 && return y1
    return U(0)
end

end
