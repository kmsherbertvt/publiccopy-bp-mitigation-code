include("fast_pauli_vec_mult.jl")

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
    phase::T

    function Pauli{T}(id::T, x::T, y::T, z::T, phase::T) where T<:Integer
        s_id = bitstring(id)
        s_x = bitstring(x)
        s_y = bitstring(y)
        s_z = bitstring(z)

        have_ones = filter(s -> '1' in s, [s_id, s_x, s_y, s_z])
        if length(have_ones) != 0
            f = minimum(map(s -> findfirst('1', s), have_ones))

            for tup in zip(s_id[f:end], s_x[f:end], s_y[f:end], s_z[f:end])
                if sum(map(c -> parse(Int, c), tup)) > 1
                    error("Invalid Pauli: $id, $x, $y, $z")
                end
            end
        end
        new(id, x, y, z, phase)
    end
end


function pauli_commute(P::Pauli, Q::Pauli)
    id = xor(P.id, Q.id)
    x = xor(P.x, Q.x)
    y = xor(P.y, Q.y)
    z = xor(P.z, Q.z)
    
    x = (~id)^x
    y = (~id)^y
    z = (~id)^z

    res = 0
    res += count_ones(x)
    res += count_ones(y)
    res += count_ones(z)

    println(res)

    return Bool((res+1)%2)
end


function pauli_string_to_pauli(ps::String)
    l = zeros(Int64, length(ps))
    for (i, c)=enumerate(ps)
        if c == 'I'
            l[i] = 0
        elseif c == 'X'
            l[i] = 1
        elseif c == 'Y'
            l[i] = 2
        elseif c == 'Z'
            l[i] = 3
        else
            error("Invalid character: $c")
        end
    end
    pm = [0, 0, 0, 0]
    pauli_masks(pm, l)
    return Pauli{Int64}(pm..., 0)
end