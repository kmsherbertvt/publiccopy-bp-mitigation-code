using LinearAlgebra

include("structs.jl")
include("fast_pauli_vec_mult.jl")


function pauli_commute(p::Array{Int64,1}, q::Array{Int64,1})
    disagrees = 0
    n = length(p)
    for i=1:n
        if p[i] == 0 || q[i] == 0
            continue
        end
        if p[i] != q[i]
            disagrees += 1
        end
    end
    if disagrees % 2 == 0
        return true
    else
        return false
    end
end


function pp(p::Int64, q::Int64)
    #=
    Returns pauli (axis) and phase
    represented by int in [0, 1, 2, 3]
    =#
    if p == q
        return (0, 0)
    end
    if p == 0
        return (q, 0)
    elseif q == 0
        return (p, 0)
    end

    if p == 1
        if q == 2
            return (3, 1)
        else # q == 3
            return (2, 3)
        end
    elseif p == 2
        if q == 1
            return (3, 3)
        else # q == 3
            return (1, 1)
        end
    else # p == 3
        if q == 1
            return (2, 1)
        else # q == 2
            return (1, 3)
        end
    end
end


function pauli_product(p::Array{Int64,1}, q::Array{Int64,1})
    n = length(p)
    phase = 0
    res = Zeros(Int64, n)
    for i=1:n
        new_pauli, new_phase = pp(p[i], q[i])
        res[i] = new_pauli
        phase += new_phase
    end
    phase = phase % 4
    return (res, phase)
end


function exp_val(O::Operator, state::Array{ComplexF64,1})
    res = 0.0 + 0.0im
    n = length(O.paulis[1])
    l = length(O.coeffs)
    tmp = Zeros(ComplexF64, 2^n)
    Pstate = Zeros(ComplexF64, 2^n)

    for i=1:l
        pauli_vec_mult!(Pstate, O.paulis[i], state, tmp)
        Pstate = conj(Pstate)
        res += O.coeffs[i] * dot(Pstate, state)
    end
    return res
end


function commutator(O::Operator, p::Array{Int64,1})
    l = length(O.coeffs)
    n = length(O.paulis[1])

    res = Operator([],[])

    for i=1:l
        q = O.paulis[i]
        if pauli_commute(q, p)
            continue
        end
        r, phase = pauli_product(q, p)
        push!(
            res.coeffs, 
            exp(phase/2*1im*pi)*O.coeffs[i]
        )
        push!(res.paulis, r)
    end
    return res
end