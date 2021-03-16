include("structs.jl")
include("callbacks.jl")


function pauli_commute(p::PauliString, q::PauliString)
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


function pauli_product(p::PauliString, q::PauliString)
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


function commutator(H::Hamiltonian, p::PauliString)
    ...
end