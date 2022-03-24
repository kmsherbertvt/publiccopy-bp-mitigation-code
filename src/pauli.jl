import Base
using LinearAlgebra
import IterTools: enumerate

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
        #s_x = bitstring(x)
        #s_y = bitstring(y)
        #s_z = bitstring(z)

        #have_ones = filter(s -> '1' in s, [s_x, s_y, s_z])
        #if length(have_ones) != 0
        #    f = minimum(map(s -> findfirst('1', s), have_ones))

        #    for tup in zip(s_x[f:end], s_y[f:end], s_z[f:end])
        #        if sum(map(c -> parse(Int, c), tup)) > 1
        #            error("Invalid Pauli: $x, $y, $z")
        #        end
        #    end
        #end

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


Base.:(==)(lhs ::Pauli, rhs ::Pauli) = (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.id == rhs.id) && (lhs.phase == rhs.phase)


function num_qubits(P::Pauli)
    return maximum(map(k -> ndigits(k, base=2), [P.x, P.y, P.z]))
end


function pauli_to_axes(P::Pauli{T}, n::Int) where T<:Unsigned
    if num_qubits(P) > n
        error("Needs more qubits")
    end

    l = zeros(Int64, n)

    for i=1:n
        for (ax, p_comp) in zip([1, 2, 3], [P.x, P.y, P.z])
            res = ((p_comp >> (i-1)) & 1)
            if res == 1
                l[i] = ax
            end
        end
    end
    return reverse(l)
end


function pauli_string_to_pauli(ps::String, type_out = UInt64)
    """ This function reads in a Pauli string lexicographically.
    This means that an input "XYZ" correponds to `Z1 Y2 X3`.
    Another way of saying this is that the input string is
    interpreted as little-endian, i.e. the rightmost bit has
    the smallest index.
    """
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
    return pauli_string_to_pauli(l, type_out)
end


function pauli_string_to_pauli(ps::Array{Int64,1}, type_out = UInt64)
    pm = [0, 0, 0, 0]
    _pauli_masks(pm, ps)
    _type_out = unsigned(type_out)
    return Pauli(
        _type_out(pm[2]),
        _type_out(pm[3]),
        _type_out(pm[4]),
        _type_out(0),
    )
end

function pauli_to_pauli_string(P::Pauli{T}, n::Int) where T<:Unsigned
    plist = ["I","X","Y","Z"]
    pax = pauli_to_axes(P,n) .+ 1
    pstr = []
    for el in reverse(pax)
        push!(pstr,plist[el])
    end
    return reverse(join(pstr))
end

function _pauli_masks(res::Array{Int64,1}, pauli_str::Array{Int64,1})
    for (i,ax)=enumerate(reverse(pauli_str))
        res[ax+1] += 2^(i-1)
    end
end


function Base.show(io::IO, P::Pauli)
    num_qubits = maximum(map(i -> ndigits(i, base=2), [P.x, P.y, P.z]))
    xs = bitstring(P.x)[end-num_qubits+1:end]
    ys = bitstring(P.y)[end-num_qubits+1:end]
    zs = bitstring(P.z)[end-num_qubits+1:end]
    if P.phase == 0
        ph = "+1"
    elseif P.phase == 1
        ph = "+i"
    elseif P.phase == 2
        ph = "-1"
    elseif P.phase == 3
        ph = "-i"
    else
        error("invalid phase...")
    end
    print(io, "Pauli(x=$xs, y=$ys, z=$zs, ph=$ph)")
end


function phase_shift(alpha::ComplexF64, i::Integer)
    if i == 0
        return     alpha.re + im*alpha.im
    elseif i == 1
        return  im*alpha.re - alpha.im
    elseif i == 2
        return    -alpha.re - im*alpha.im
    else
        return -im*alpha.re + alpha.im
    end
end


function pauli_phase(pm::Pauli, a::Int64)
    # Compute the phase gamma where P|a> = gamma |b> for
    # a pauli string P and basis state |a>.
    # Convention:
    #   0 -> 1
    #   1 -> +i
    #   2 -> -1
    #   3 -> -i
    # pm = pauli_mask input
    x = count_ones((pm.y | pm.z) & a) % 2
    y = count_ones(pm.y) % 4

    alpha = y
    beta = 2*x

    return UInt8((alpha+beta) % 4)
end


function pauli_apply(pm::Pauli, a::Int64)
    # pm = pauli_mask input
    return xor((pm.x|pm.y),a)
end


function pauli_mult!(pm::Pauli, state::Array{ComplexF64,1}, result::Array{ComplexF64,1})
    N = length(state)
    for i=0:N-1
        j = pauli_apply(pm, i)
        phase = pauli_phase(pm, i) % 4
        r = state[i+1]
        result[j+1] = phase_shift(r, phase)
    end
end


function pauli_commute(P::Pauli, Q::Pauli)
    id = (P.id | Q.id)
    x = (P.x & Q.x)|id
    y = (P.y & Q.y)|id
    z = (P.z & Q.z)|id

    res = 0
    res += count_ones(x)
    res += count_ones(y)
    res += count_ones(z)

    return Bool((res+1)%2)
end


function pauli_product(P::Pauli, Q::Pauli)
    phase = typeof(P.x)(0)

    out_x = (P.z & Q.y) | (Q.z & P.y) | (P.id & Q.x) | (Q.id & P.x)
    phase += 3*(count_ones(P.z & Q.y) - count_ones(Q.z & P.y))

    out_y = (P.x & Q.z) | (Q.x & P.z) | (P.id & Q.y) | (Q.id & P.y)
    phase -= 1*(count_ones(P.x & Q.z) - count_ones(Q.x & P.z))

    out_z = (P.x & Q.y) | (Q.x & P.y) | (P.id & Q.z) | (Q.id & P.z)
    phase += 1*(count_ones(P.x & Q.y) - count_ones(Q.x & P.y))

    return Pauli(out_x, out_y, out_z, phase%4)
end
