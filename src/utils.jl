using StatsBase

_PAULI_DICT = Dict(0 => "I", 1 => "X", 2 => "Y", 3 => "Z")

"""
    uniform_state(n::Int)

Return the vector representation of the all `|+>` state on `n` qubits.
"""
function uniform_state(n::Int)
    initial_state = ones(ComplexF64, 2^n) / sqrt(2^n)
    initial_state /= norm(initial_state)
    return initial_state
end

function energy_improving(v)
    return all(diff(v) .<= 0)
end

function String(pauli::Pauli{T}; zero_index = false) where T<:UInt
    ax = reverse(pauli_to_axes(pauli, num_qubits(pauli)))
    s = "["
    for (i,a)=enumerate(ax)
        if zero_index
            i -= 1
        end
        if a != 0
            s = s * "$(_PAULI_DICT[a])$i "
        end
    end
    s = strip(s, [' '])
    s = s * "]"
    return s
end

function String(op::Operator)
    s = "Op["
    for (p,c)=zip(op.paulis, op.coeffs)
        term = "($c)$(String(p)) + "
        s = s * term
    end
    s = strip(s, [' ', '+'])
    s = s * "]"
end

function safe_floor(x::Float64, eps=1e-15, delta=1e-8)
    if x <= -delta error("Too negative...") end
    if x <= 0.0
        return eps
    else
        return x
    end
end

function log_mean(x)
	return 10^mean(log10.(safe_floor.(x)))
end

function get_git_id()
    return chop(read(`git rev-parse --short HEAD`, String), tail=2)
end