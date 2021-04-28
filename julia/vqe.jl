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
    state = copy(initial_state)

    function _cost_fn(x::Vector{Float64})
        state .= initial_state
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

mutable struct ADAPTHistory
    energy::Array{Float64,1}
    max_grad::Array{Float64,1}
    max_grad_ind::Array{Int64,1}
    grads::Array{Array{Float64, 1}, 1}
    opt_pars::Array{Array{Float64, 1}, 1}
end


function adapt_step!(
        hist::ADAPTHistory,
        comms,
        tmp,
        state,
        hamiltonian,
        opt_pars
        )
    push!(
        hist.grads,
        [imag(exp_val(com, state, tmp)) for com in comms]
        ) # phase?

    push!(
        hist.max_grad_ind,
        argmax(map(x -> abs(x), hist.grads[end]))
    )

    push!(
        hist.max_grad,
        abs(hist.grads[end][hist.max_grad_ind[end]])
    )

    push!(
        hist.energy,
        real(exp_val(hamiltonian, state, tmp))
    )

    push!(
        hist.opt_pars,
        opt_pars
    )
end


function adapt_vqe(
    hamiltonian::Operator,
    pool::Array{Pauli{T},1},
    num_qubits::Int64,
    optimizer::Union{String,Dict},
    callbacks::Array{Function},
    state::Union{Nothing,Array{ComplexF64,1}} = nothing, # Initial state
    tmp::Union{Nothing, Array{ComplexF64,1}} = nothing
) where T<:Unsigned
    hist = ADAPTHistory([], [], [], [], [])

    if optimizer isa String
        opt_dict = Dict("name" => optimizer)
    elseif optimizer isa Dict
        opt_dict = optimizer
    else
        throw(ArgumentError("optimizer should be String or Dict"))
    end

    if tmp === nothing
        tmp = zeros(ComplexF64, 2^num_qubits)
        tmp[1] = 1.0 + 0.0im
    end
    if state === nothing
        state = zeros(ComplexF64, 2^num_qubits)
        state[1] = 1.0 + 0.0im
    end

    comms = Array{Operator, 1}()
    for _op in pool
        op = Operator([_op], [1.0])
        push!(comms, commutator(hamiltonian, op))
    end

    adapt_step!(hist, comms, tmp, state, hamiltonian, [])
    ansatz = Array{Pauli{T}, 1}()

    while true
        for c in callbacks
            if c(hist)
                return hist
            end
        end

        push!(ansatz, pool[hist.max_grad_ind[end]])
        point = vcat(hist.opt_pars[end], [0.0])

        opt = Opt(Symbol(opt_dict["name"]), length(point))
        if haskey(opt_dict,"maxeval")
            opt.maxeval = opt_dict["maxeval"]
        end
        if haskey(opt_dict,"ftol_rel")
            opt.ftol_rel = opt_dict["ftol_rel"]
        end
        if haskey(opt_dict,"xtol_rel")
            opt.xtol_rel = opt_dict["xtol_rel"]
        end

        energy, point, ret = VQE(hamiltonian, ansatz, opt, point, num_qubits, state)#, tmp)  note that VQE doesn't take a tmp, it makes its own
        pauli_ansatz!(ansatz, point, state, tmp)
        adapt_step!(hist, comms, tmp, state, hamiltonian, point)
    end
end
