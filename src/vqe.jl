using HDF5
using NLopt

include("pauli.jl")
include("operator.jl")
include("simulator.jl")
include("fast_grad.jl")

function commuting_vqe(
    hamiltonian::Operator,
    ansatz::Array{Operator,1},
    opt::Opt,
    initial_point::Array{Float64,1},
    num_qubits::Int64,
    initial_state::Union{Nothing,Array{ComplexF64,1}} = nothing, # Initial state
    path = nothing, # Should be a CSV file
    output_state::Union{Nothing,Array{ComplexF64,1}} = nothing
)
    tmp = zeros(ComplexF64, 2^num_qubits)
    tmp1 = similar(tmp)
    tmp2 = similar(tmp)
    if initial_state === nothing
        initial_state = zeros(ComplexF64, 2^num_qubits)
        initial_state[1] = 1.0 + 0.0im
    end
    state = copy(initial_state)
    eval_count = 0

    fn_evals = Vector{Float64}()
    grad_evals = Vector{Float64}()

    function cost_fn(x::Vector{Float64}, grad::Vector{Float64})
        state .= initial_state
        
        unpacked_x = []
        unpacked_ansatz = []

        for (xi, op) in zip(x,ansatz)
            xp = xi*real(op.coeffs)
            append!(unpacked_ansatz, op.paulis)
            append!(unpacked_x, xp)
            pauli_ansatz!(op.paulis, xp, state, tmp)
        end
        
        res = real(exp_val(hamiltonian, state, tmp))

        if output_state !== nothing
            output_state .= state
        end

        if length(grad) > 0
            state .= initial_state

            repeat_lengths = map(o -> length(o.paulis), ansatz)
            unpacked_grad = similar(unpacked_x)

            unpacked_grad = convert(Array{Float64,1},unpacked_grad)
            unpacked_x = convert(Array{Float64,1},unpacked_x)
            unpacked_ansatz = convert(Array{Pauli{UInt64},1},unpacked_ansatz)

            fast_grad!(hamiltonian, unpacked_ansatz, unpacked_x, unpacked_grad, tmp, state, tmp1, tmp2)

            
            inds_begin = accumulate(+,repeat_lengths,init=1)-repeat_lengths
            inds_end = accumulate(+,repeat_lengths)

            grad_list = [unpacked_grad[i1:i2] for (i1,i2) in zip(inds_begin,inds_end)]
            grad .= map(sum, grad_list)
        end

	push!(fn_evals, res)
	append!(grad_evals, grad)

        eval_count += 1

        return res
    end

    opt.lower_bounds = -π
    opt.upper_bounds = +π
    opt.min_objective = cost_fn

    # Test #
    cost_fn(initial_point, similar(initial_point))
    ########

    (minf,minx,ret) = optimize(opt, initial_point)

    #if path != nothing
    #    fid = h5open(path, "w")
    #    fid["fn_evals"] = fn_evals
    #    fid["grad_evals"] = grad_evals
    #    close(fid)
    #end

    return (minf, minx, ret, eval_count)
end

function QAOA(
    hamiltonian::Operator,
    mixers::Array{Operator,1},
    opt::Opt,
    initial_point::Array{Float64,1},
    num_qubits::Int64,
    initial_state::Union{Nothing,Array{ComplexF64,1}} = nothing, # Initial state
    path = nothing, # Should be a CSV file
    output_state::Union{Nothing,Array{ComplexF64,1}} = nothing
)
    # num_pars == 2*length(mixers)+1
    p = length(mixers)
    ansatz = zip(mixers, repeat([hamiltonian], p))
    ansatz = collect(Iterators.flatten(ansatz))
    prepend!(ansatz, [hamiltonian])

    return commuting_vqe(
        hamiltonian,
        ansatz,
        opt,
        initial_point,
        num_qubits,
        initial_state,
        path,
        output_state)
end

mutable struct ADAPTHistory
    energy::Array{Float64,1}
    max_grad::Array{Float64,1}
    max_grad_ind::Array{Int64,1}
    grads::Array{Array{Float64, 1}, 1}
    opt_pars::Array{Array{Float64, 1}, 1}
    paulis::Array{Any,1}
    opt_numevals::Array{Any,1}
end


function adapt_history_dump!(hist::ADAPTHistory, path::String, num_qubits::Int64)
    l = length(hist.energy)
    open(path, "w") do io
        write(io, "layer; energy; max_grad; max_grad_ind; grads; opt_pars; opt_numevals; paulis\n")
        for (i, en, mg, mgi, gr, op, ne, pauli)=zip(1:l, hist.energy, hist.max_grad, hist.max_grad_ind, hist.grads, hist.opt_pars, hist.opt_numevals, hist.paulis)
            if pauli === nothing
                ps = nothing
	    else
		ps = "null"
		#ps = pauli_to_pauli_string(pauli, num_qubits)
	    end

            if length(op) == 0
	        op = nothing
	    end
              

            write(io, "$i; $en; $mg; $mgi; $gr; $op; $ne; $ps\n")
        end
    end
end


function adapt_step!(
        hist::ADAPTHistory,
        comms,
        tmp,
        state,
        hamiltonian,
        opt_pars,
        pauli_chosen,
        numevals
        )
    push!(hist.grads, [imag(exp_val(com, state, tmp)) for com in comms]) # phase?
    push!(hist.max_grad_ind, argmax(map(x -> abs(x), hist.grads[end])))
    push!(hist.max_grad, abs(hist.grads[end][hist.max_grad_ind[end]]))
    push!(hist.energy, real(exp_val(hamiltonian, state, tmp)))
    push!(hist.opt_pars, opt_pars)
    push!(hist.paulis, pauli_chosen)
    push!(hist.opt_numevals, numevals)
end


function adapt_vqe(
    hamiltonian::Operator,
    pool::Array{Pauli{T},1},
    num_qubits::Int64,
    optimizer::Union{String,Dict},
    callbacks::Array{Function};
    initial_parameter::Float64 = 0.0,
    initial_state::Union{Nothing,Array{ComplexF64,1}} = nothing, # Initial state
    path = nothing,
    tmp::Union{Nothing, Array{ComplexF64,1}} = nothing
) where T<:Unsigned
    hist = ADAPTHistory([], [], [], [], [], [], [])

    if path !== nothing
        """ in $path the following files will be written
            $path/layer_*.csv - indexed by integer
            $path/adapt_history.csv
        """
    end

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
    if initial_state === nothing
        initial_state = zeros(ComplexF64, 2^num_qubits)
        initial_state[1] = 1.0 + 0.0im
    end

    comms = Array{Operator, 1}()
    for _op in pool
        op = Operator([_op], [1.0])
        push!(comms, commutator(hamiltonian, op, false))
    end

    state = copy(initial_state)
    adapt_step!(hist, comms, tmp, state, hamiltonian, [], nothing, nothing)
    ansatz = Array{Pauli{T}, 1}()

    layer_count = 0

    while true
        for c in callbacks
            if c(hist)
                if path !== nothing
                    adapt_history_dump!(hist, "$path/adapt_history.csv", num_qubits)
                end
                return hist
            end
        end

        push!(ansatz, pool[hist.max_grad_ind[end]])
        point = vcat(hist.opt_pars[end], [initial_parameter])

        opt = Opt(Symbol(opt_dict["name"]), length(point))

        opt_keys = collect(keys(opt_dict))
        deleteat!(opt_keys,findall(x->x=="name",opt_keys))
        for akey in opt_keys
            setproperty!(opt,Symbol(akey),opt_dict[akey])
        end

	vqe_path = "$path/vqe_layer_$layer_count.h5"

        state .= initial_state
        energy, point, ret, opt_evals = VQE(hamiltonian, ansatz, opt, point, num_qubits, state, vqe_path)
        state .= initial_state
        pauli_ansatz!(ansatz, point, state, tmp)
        adapt_step!(hist, comms, tmp, state, hamiltonian, point,pool[hist.max_grad_ind[end]],opt_evals) # pool operator of the step that just finished

        layer_count += 1
    end
end


function adapt_qaoa(
    hamiltonian::Operator,
    ### CHANGE
    pool::Array{Operator,1},
    ##########
    num_qubits::Int64,
    optimizer::Union{String,Dict},
    callbacks::Array{Function};
    initial_parameter::Float64 = 0.0,
    initial_state::Union{Nothing,Array{ComplexF64,1}} = nothing, # Initial state
    path = nothing,
    tmp::Union{Nothing, Array{ComplexF64,1}} = nothing
)
    hist = ADAPTHistory([], [], [], [], [], [], [])

    if path !== nothing
        """ in $path the following files will be written
            $path/layer_*.csv - indexed by integer
            $path/adapt_history.csv
        """
    end

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
    if initial_state === nothing
        initial_state = zeros(ComplexF64, 2^num_qubits)
        initial_state[1] = 1.0 + 0.0im
    end

    comms = Array{Operator, 1}()
    ##### CHANGE
    for _op in pool
        push!(comms, commutator(hamiltonian, _op, false))
    end
    ############

    state = copy(initial_state)
    adapt_step!(hist, comms, tmp, state, hamiltonian, [], nothing, nothing)

    ##### CHANGE
    ansatz = Array{Operator, 1}()
    ############

    layer_count = 0

    while true
        for c in callbacks
            if c(hist)
                #if path !== nothing
                #    adapt_history_dump!(hist, "$path/adapt_history.csv", num_qubits)
                #end
                return hist
            end
        end

        push!(ansatz, pool[hist.max_grad_ind[end]])
        ##### CHANGE
        point = vcat(hist.opt_pars[end], [initial_parameter, initial_parameter])
        if length(point) == 2
            push!(point, initial_parameter)
        end
        #point = vcat(hist.opt_pars[end], [initial_parameter, initial_parameter])
        ############

        opt = Opt(Symbol(opt_dict["name"]), length(point))
        opt_keys = collect(keys(opt_dict))
        deleteat!(opt_keys,findall(x->x=="name",opt_keys))
        for akey in opt_keys
            setproperty!(opt,Symbol(akey),opt_dict[akey])
        end

	vqe_path = "$path/vqe_layer_$layer_count.h5"

        state .= initial_state

        energy, point, ret, opt_evals = QAOA(hamiltonian, ansatz, opt, point, num_qubits, state, vqe_path, tmp)
        ##### CHANGE
        state = copy(tmp)
        ############
        adapt_step!(hist, comms, tmp, state, hamiltonian, point,pool[hist.max_grad_ind[end]],opt_evals) # pool operator of the step that just finished

        layer_count += 1
    end
end
