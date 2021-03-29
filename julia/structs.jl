struct Pauli{T<:Integer}
    #=
    Pauli mask representation of a Pauli string.
    This type is parametric on `T` so that smaller
    sized integers can be used to represent the Pauli
    string, if possible. This also performs a check that
    the given integers constitute a valid Pauli string.
    
    Example:
    Pauli string: "IIXYZYZIYIZ"
              id:  11000001010 -> 1546
               x:  00100000000 -> 256
               y:  00010100100 -> 164
               z:  00001010001 -> 81

    =#
    id::T
    x::T
    y::T
    z::T

    function Pauli{T}(id::T, x::T, y::T, z::T) where T<:Integer
        s_id = bitstring(id)
        s_x = bitstring(x)
        s_y = bitstring(y)
        s_z = bitstring(z)
        f = minimum(map(s -> findfirst('1', s), [s_id, s_x, s_y, s_z]))
        for tup in zip(s_id[f:end], s_x[f:end], s_y[f:end], s_z[f:end])
            if sum(map(c -> parse(Int, c), tup)) > 1
                error("Invalid Pauli: $id, $x, $y, $z")
            end
        end
        new(id, x, y, z)
    end
end