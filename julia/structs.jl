using LinearAlgebra

include("fast_pauli_vec_mult.jl")

mutable struct Pauli{T<:Unsigned}
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

function pauli_masks(res::Array{Int64,1}, pauli_str::Pauli)
    res[1] = pauli_str.id
    res[2] = pauli_str.x
    res[3] = pauli_str.y
    res[4] = pauli_str.z
end

function Base.show(io::IO, P::Pauli) 
    num_qubits = maximum(map(i -> ndigits(i, base=2), [P.x, P.y, P.z]))
    xs = bitstring(P.x)[end-num_qubits+1:end]
    ys = bitstring(P.y)[end-num_qubits+1:end]
    zs = bitstring(P.z)[end-num_qubits+1:end]
    print("Pauli(x=$xs, y=$ys, z=$zs)")
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


mutable struct Operator
    paulis::Array{Pauli,1}
    coeffs::Array{ComplexF64,1}
end

function group_inds_by_eq(A::Array)
    l = length(A)
    grps = Dict{typeof(A[1]), Array{Int64,1}}()
    hit_elts = []
    for i=1:l
        el = A[i]
        if el in hit_elts
            append!(grps[el], i)
        else
            grps[el] = [i]
            append!(hit_elts, el)
        end
    end
    return grps
end

function op_simplify(A::Operator, tol::Float64 = 0.0)
    l = length(A.paulis)

    # Get rid of zero coeff ops
    for i=reverse(1:l)
        if abs(A.coeffs[i]) <= tol
            deleteat!(A.coeffs, i)
            deleteat!(A.paulis, i)
        end
    end

    # Bring all phases out of Paulis
    for i=1:l
        ph = A.paulis[i].phase
        A.paulis[i].phase = 0
        alpha = A.coeffs[i]
        A.coeffs[i] = phase_shift(alpha, ph)
    end

    # Sum duplicate paulis
    dict = group_inds_by_eq(A.paulis)
    new_coeffs = Array{ComplexF64,1}()
    new_paulis = Array{Pauli,1}()

    for (pauli, inds) in d
        c = 0.0 + 0.0im
        for i in inds
            c += A.coeffs[i]
        end
        append!(new_paulis, pauli)
        append!(new_coeffs, c)
    end
    
    A.coeffs = new_coeffs
    A.paulis = new_paulis
end


function op_product(A::Operator, B::Operator)
    new_paulis = Array{Pauli,1}()
    new_coeffs = Array{ComplexF64,1}()

    for i=1:length(A.paulis)
        for j=1:length(B.paulis)
            q = pauli_product(A.paulis[i], B.paulis[j])
            append!(new_paulis, q)
            append!(new_coeffs, A.coeffs[i]*B.coeffs[j])
        end
    end
    return Operator(new_paulis, new_coeffs)
end


function pauli_mult!(pm::Array{Int}, state::Array{ComplexF64,1}, result::Array{ComplexF64,1})
    N = length(state)
    for i=0:N-1
        j = pauli_apply(pm, i)
        phase = (pauli_phase(pm, i)+1) % 4
        r = state[i+1]
        result[j+1] = phase_shift(r, phase)
    end
end


function exp_val(A::Operator, state::Array{ComplexF64,1}, tmp::Array{ComplexF64})
    result = 0.0 + 0.0*im
    pm = [0, 0, 0, 0]
    for (c,p)=zip(A.coeffs, A.paulis)
        # This convention is fine since P=dagger(P)
        # Equiv to (<psi|P)|psi> = <psi|(P|psi>)
        pm[1] = p.id; pm[2] = p.x; pm[3] = p.y; pm[4] = p.z;
        pauli_mult!(pm, state, tmp) # tmp <- P.state
        result += phase_shift(c, p.phase)*dot(conj.(tmp), state)
    end
    return result
end