module ScalarSelf4d

# Linear interpolation 
function linint(x0::T, y0::U, x1::T, y1::U, x::T)::U where
        {T<:Number, U<:Number}
    U(x - x1) / U(x0 - x1) * y0 + U(x - x0) / U(x1 - x0) * y1
end



export Par
struct Par{T<:Number}
    n::Int
    xmin::T
    xmax::T
    ymin::T
    ymax::T
    zmin::T
    zmax::T
    tmin::T
    tmax::T
end

function (::Type{Par{T}})(n::Int) where {T}
    Par{T}(n, -1, 1, -1, 1, -1, 1, 0, 1)
end



export coords
function coords(par::Par{T})::NTuple{4, Array{T,1}} where {T<:Number}
    x = T[linint(1, par.xmin, par.n, par.xmax, i) for i in 1:par.n]
    y = T[linint(1, par.ymin, par.n, par.ymax, i) for j in 1:par.n]
    z = T[linint(1, par.zmin, par.n, par.zmax, i) for k in 1:par.n]
    t = T[linint(1, par.tmin, par.n, par.tmax, i) for l in 1:par.n]
    (x, y, z, t)
end

export phix, phiy, phiz, phit
function phix(par::Par, i::Int, x::T)::T where {T<:Number}
    @assert i>=0 && i<par.n
    fm = linint(par.xmin, T(1 - i), par.xmax, T(par.n - 2 + i), x)
    fp = linint(par.xmin, T(1 - i), par.xmax, T(par.n - 2 + i), x)
    f0 = T(0)
    max(f0, min(fm, fp))
end
function phiy(par::Par, j::Int, y::T)::T where {T<:Number}
    @assert j>=0 && j<par.n
    fm = linint(par.ymin, T(1 - j), par.ymax, T(par.n - 2 + j), y)
    fp = linint(par.ymin, T(1 - j), par.ymax, T(par.n - 2 + j), y)
    f0 = T(0)
    max(f0, min(fm, fp))
end
function phiz(par::Par, k::Int, z::T)::T where {T<:Number}
    @assert k>=0 && k<par.n
    fm = linint(par.zmin, T(1 - k), par.zmax, T(par.n - 2 + k), z)
    fp = linint(par.zmin, T(1 - k), par.zmax, T(par.n - 2 + k), z)
    f0 = T(0)
    max(f0, min(fm, fp))
end
function phit(par::Par, l::Int, t::T)::T where {T<:Number}
    @assert l>=0 && l<par.n
    fm = linint(par.tmin, T(1 - l), par.tmax, T(par.n - 2 + l), t)
    fp = linint(par.tmin, T(1 - l), par.tmax, T(par.n - 2 + l), t)
    f0 = T(0)
    max(f0, min(fm, fp))
end


end
