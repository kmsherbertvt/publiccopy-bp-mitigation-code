include("fast_pauli_vec_mult.jl")

struct Pauli{T<:Unsigned}
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

    function Pauli{T}(x::T, y::T, z::T, phase::T) where T<:Unsigned
        s_x = bitstring(x)
        s_y = bitstring(y)
        s_z = bitstring(z)

        have_ones = filter(s -> '1' in s, [s_x, s_y, s_z])
        if length(have_ones) != 0
            f = minimum(map(s -> findfirst('1', s), have_ones))

            for tup in zip(s_x[f:end], s_y[f:end], s_z[f:end])
                if sum(map(c -> parse(Int, c), tup)) > 1
                    error("Invalid Pauli: $x, $y, $z")
                end
            end
        end
        
        new(~(x|y|z), x, y, z, phase)
    end
    function Pauli{T}(id::T, x::T, y::T, z::T, phase::T) where T<:Unsigned
        return Pauli{T}(x, y, z, phase)
    end

    function Pauli(x::Integer, y::Integer, z::Integer, phase::Integer)
        t = unsigned(typeof(x))
        return Pauli{t}(unsigned(x), unsigned(y), unsigned(z), unsigned(phase))
    end
end


function Base.show(io::IO, P::Pauli) 
    num_qubits = maximum(map(i -> ndigits(i, base=2), [P.x, P.y, P.z]))
    xs = bitstring(P.x)[end-num_qubits+1:end]
    ys = bitstring(P.y)[end-num_qubits+1:end]
    zs = bitstring(P.z)[end-num_qubits+1:end]
    println("Pauli(x=$xs, y=$ys, z=$zs)")
end


function pauli_commute(P::Pauli, Q::Pauli)
    nid = ~(P.id | Q.id)
    x = xor(P.x, Q.x)&nid
    y = xor(P.y, Q.y)&nid
    z = xor(P.z, Q.z)&nid
    
    res = 0
    res += count_ones(x)
    res += count_ones(y)
    res += count_ones(z)

    return Bool(((res÷2)+1)%2)
end


function pauli_string_to_pauli(ps::String, type_out = UInt64)
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
    _type_out = unsigned(type_out)
    return Pauli(
        _type_out(pm[2]),
        _type_out(pm[3]),
        _type_out(pm[4]),
        _type_out(0),
    )
end