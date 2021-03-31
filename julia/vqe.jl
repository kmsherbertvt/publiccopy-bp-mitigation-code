using NLopt

include("pauli.jl")
include("operator.jl")
include("simulator.jl")


function VQE(
    hamiltonian::Operator,
    ansatz::Array{Pauli{T},1},
    opt::Opt,
    initial_point::Array{Float64,1},
    num_qubits::Int64
) where T<:Unsigned
    tmp = zeros(ComplexF64, 2^num_qubits)
    state = zeros(ComplexF64, 2^num_qubits)
    state[1] = 1.0 + 0.0im
    function cost_fn(x::Vector, grad::Vector) # grad::Vector?
        #if length(grad) > 0
        #    
        #end
        pauli_ansatz!(ansatz, x, state, tmp)
        res = exp_val(hamiltonian, state, tmp).re
        return res
    end

    opt.lower_bounds = -π
    opt.upper_bounds = +π
    opt.min_objective = cost_fn

    (minf,minx,ret) = optimize(opt, initial_point)

    return (minf, minx, ret)
end