function unpack_vector(x::Vector, n::Vector)
    if length(x) != length(n) error("Must be same length") end
    xp = []
    for (ni, xi) in zip(n,x)
        append!(xp, repeat([xi], ni))
    end
    return xp
end

function pack_vector(y_input::Vector, n::Vector)
    yp = copy(y_input)
    if length(yp) != sum(n) error("Cannot unpack") end
    y = []
    for ni in reverse(n)
        yi = sum([pop!(yp) for i=1:ni])
        push!(y, yi)
    end
    reverse!(y)
    return y
end