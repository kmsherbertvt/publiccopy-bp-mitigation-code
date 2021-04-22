using NLopt

include("pauli.jl")
include("operator.jl")
include("simulator.jl")


function VQE(
    hamiltonian::Operator,
    ansatz::Array{Pauli{T},1},
    opt::Opt,
    initial_point::Array{Float64,1},
    num_qubits::Int64,
    initial_state::Union{Nothing,Array{ComplexF64,1}} = nothing # Initial state
) where T<:Unsigned
    tmp = zeros(ComplexF64, 2^num_qubits)
    if initial_state === nothing
        initial_state = zeros(ComplexF64, 2^num_qubits)
        initial_state[1] = 1.0 + 0.0im
    end

    function _cost_fn(x::Vector{Float64})
        state = copy(initial_state)
        pauli_ansatz!(ansatz, x, state, tmp)
        res = real(exp_val(hamiltonian, state, tmp))
        return res
    end

    function cost_fn(x::Vector{Float64}, grad::Vector{Float64})
        if length(grad) > 0
            error("Gradients not supported, yet...")
        end
        return _cost_fn(x)
    end

    opt.lower_bounds = -π
    opt.upper_bounds = +π
    opt.min_objective = cost_fn

    (minf,minx,ret) = optimize(opt, initial_point)

    return (minf, minx, ret)
end
