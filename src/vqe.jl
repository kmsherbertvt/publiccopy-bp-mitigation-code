using HDF5
using NLopt
using Random
using Distributions

""" Unpacking a vector refers to taking the original vector (`x`) and
repeating each element in a new vector some (potentially variable) number
of times.

If I have an ansatz where parameters are bound together,
e.g. `U(t1) V(t2)` where
`U(t) = e^(-i t A1) e^(-i t A2)`
`V(t) = e^(-i t B1) e^(-i t B2)`
then the parameter vector `[t1, t2]` unpacked according to the repetitions
`[2, 2]` would be `[t1, t1, t2, t2]`.

On the other hand, unpacking a vector undoes this process, where the aggregation
function is assumed to be addition. Hence, packing the vector `[t1, t2, t3, t4]`
according to the reptitions `[2, 2]` would yield the vector `[t1+t2, t3+t4]`.
"""

function unpack_vector(x::Vector, n::Vector; s = nothing)
    """
        unpack_vector(x::Vector, n::Vector; s = nothing)

    ### Inputs
    * x::Vector -- The vector to unpack
    * n::Vector -- The number of times to repeat each element of `x`
    * s::Vector -- The vector used for scaling, defaults to all ones.

    ### Output
    This function acts much like `repeat`, except that each element of
        `x` is repeated according to the corresponding integer in the vector `n`.
        For instance, if `x=[x1,x2]` and `n=[1,2]`, then the output will be
        `[x1,x2,x2]`. If a scale is provided, then the elements of the output
        vector will be scaled according to the elements in `s`. For instance, if
        `s = [2, 3]` in the previous example, the output would instead be
        `[2*x1, 3*x2, 3*x2]`.
    
    """
    if length(x) != length(n) error("Must be same length, x-n") end
    if s === nothing s = ones(Float64, length(x)) end
    if length(x) != length(s) error("Must be same length, x-s") end
    xp = []
    for (ni, xi, si) in zip(n,x,s)
        append!(xp, repeat([xi*si], ni))
    end
    return xp
end

function pack_vector(y_input::Vector, n::Vector)
    """
        pack_vector(y_input::Vector, n::Vector; s_input = nothing)

    ### Inputs
    * y_input::Vector -- The vector to unpack
    * n::Vector -- The number of times to repeat each element of `x`

    ### Output
    This function is intended to be a sort of reverse to packing. Given
        some vector, `y_input`, we want to aggregate (by summation)
        components of the vector according to the elements of `n`. Consider
        `y_input = [y1, y2, y3, y4, y5]` with `n = [2, 1, 2]`. The resulting
        vector would be `[y1+y2, y3, y4+y5]`. Note that there are `2` terms
        in the first element, `1` term in the second element, and `2` terms
        in the third, corresponding to the elements of `n`.
    """
    yp = copy(y_input)
    if length(yp) != sum(n) error("Cannot unpack") end
    y = []
    for ni in reverse(n)
        yi = sum([pop!(yp) for i=1:ni])
        push!(y, yi)
    end
    reverse!(y)
    return y
end

function _cost_fn_vqe(
    x::Vector{Float64}, 
    grad::Vector{Float64}, 
    ansatz,
    hamiltonian,
    fn_evals,
    grad_evals,
    eval_count, 
    state, 
    initial_state, 
    tmp, 
    tmp1, 
    tmp2,
    )
    eval_count += 1
    state .= initial_state
    pauli_ansatz!(ansatz, x, state, tmp)
    res = real(exp_val(hamiltonian, state, tmp))
    if length(grad) > 0
        state .= initial_state
        fast_grad!(hamiltonian, ansatz, x, grad, tmp, state, tmp1, tmp2)
    end

    push!(fn_evals, res)
    append!(grad_evals, grad)

    return res
end


function _cost_fn_commuting_vqe(
    x::Vector{Float64}, 
    grad::Vector{Float64},
    ansatz,
    hamiltonian,
    fn_evals,
    grad_evals,
    eval_count, 
    state, 
    initial_state, 
    tmp, 
    tmp1, 
    tmp2,
    output_state
    )
    # Initialize statevector
    state .= initial_state
    repeat_lengths = map(o -> length(o.paulis), ansatz)
    
    unpacked_x = []
    unpacked_ansatz = []

    if length(x) != length(ansatz) error("Invalid number of pars") end
    l = length(x)
    for i in 1:l
        xi = x[i]
        op = ansatz[i]
        xp = xi*real(op.coeffs)
        append!(unpacked_ansatz, op.paulis)
        append!(unpacked_x, xp)
        pauli_ansatz!(op.paulis, xp, state, tmp)
    end

    state_test = copy(initial_state)
    pauli_ansatz!(unpacked_ansatz, unpacked_x, state_test, tmp)
    if norm(state_test - state) > 1e-8 error("These should match") end
    
    # Update cost, output_state
    res = real(exp_val(hamiltonian, state, tmp))
    if output_state !== nothing
        output_state .= state
    end

    # Compute gradient
    if length(grad) > 0
        state .= initial_state
        unpacked_grad = similar(unpacked_x)

        fast_grad!(hamiltonian, unpacked_ansatz, unpacked_x, unpacked_grad, tmp, state, tmp1, tmp2)
        #finite_difference!(hamiltonian, unpacked_ansatz, unpacked_x, unpacked_grad, state, tmp1, tmp2, 1e-5)

        coeffs = []
        for op in ansatz append!(coeffs, op.coeffs) end
        for c in coeffs
            if abs(c) <= 1e-8 error("Cannot unscale $c") end
        end
        unpacked_grad = unpacked_grad .* coeffs
        grad .= pack_vector(unpacked_grad, repeat_lengths)
    end

    # Cleaning up
    push!(fn_evals, res)
    append!(grad_evals, grad)
    eval_count += 1
    return res
end


"""
    sample_points(hamiltonian, ansatz, initial_state, num_samples; rng = nothing)

Compute samples of the VQE cost function and its gradient components for the given Hamiltonian,
list of generators, and initial state. Optional RNG for determining points to be sampled.
"""
function sample_points(hamiltonian, ansatz, initial_state, num_samples; rng = nothing, dist = nothing, point = nothing, use_norm = false)
    if rng === nothing
        rng = _DEFAULT_RNG
    end
    eval_count = 0
    num_pars = length(ansatz)

    result_energy = Array{Float64,1}()
    result_grads = Array{Float64,1}()

    grad = zeros(Float64, num_pars)
    x = zeros(Float64, num_pars)
    psi_1 = similar(initial_state)
    psi_2 = similar(initial_state)
    psi_3 = similar(initial_state)
    psi_4 = similar(initial_state)

    for _=1:num_samples
        if (dist === nothing) && (point === nothing)
            x .= rand(rng, Uniform(-pi, +pi), num_pars)
        else
            x .= rand(rng, Uniform(-pi, +pi), num_pars)
            x /= norm(x)
            x *= dist
            x += point
        end
        _cost_fn_commuting_vqe(x, grad, ansatz, hamiltonian, result_energy, result_grads, eval_count, psi_4, initial_state, psi_1, psi_2, psi_3, nothing)
    end

    min_energy = get_ground_state(hamiltonian)
    result_errors = result_energy .- min_energy
    result_relative_errors = result_errors ./ get_energy_gap(hamiltonian)

    if use_norm
        if (length(result_grads) % num_pars) != 0
            error("Invalid length encountered")
        end
        _result_grads = similar(result_grads)
        _result_grads = [
            norm(result_grads[((k-1)*num_pars+1):(num_pars*k)])
            for k=1:num_samples
        ]
        result_grads = _result_grads
    end

    return (result_energy, result_grads, result_errors, result_relative_errors)
end


"""
    VQE(...)

Run the VQE algorithm.

# Arguments

- `hamiltonian::Operator`: Hamiltonian to minimize.
- `ansatz::Array{Pauli{T},1}`: List of generators to use.
- `opt::Union{Opt,String}`: Numerical to use, or string to specify `NLopt` optimizer.
- `initial_point::Array{Float64,1}`: Initial point to use for optimization.
- `num_qubits::Int64`: Number of qubits.
- `initial_state::Union{Nothing,Array{ComplexF64,1}} = nothing`: Initial state for the ansatz.
- `path = nothing; # Should be a CSV file`: (Unused)
- `rand_range = (-π,+π)`: (Unused, use `sample_points` instead)
- `num_samples = 50`: (Unused, use `sample_points` instead)
"""
function VQE(
    hamiltonian::Operator,
    ansatz::Array{Pauli{T},1},
    opt::Union{Opt,String},
    initial_point::Array{Float64,1},
    num_qubits::Int64,
    initial_state::Union{Nothing,Array{ComplexF64,1}} = nothing, # Initial state
    path = nothing; # Should be a CSV file
    rand_range = (-π,+π),
    num_samples = 500
    ) where T<:Unsigned
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

    function cost_fn(x, grad)
        return _cost_fn_vqe(x, grad, ansatz, hamiltonian, fn_evals, grad_evals, eval_count, state, initial_state, tmp, tmp1, tmp2)
    end

    if opt !== "random_sampling"
        opt.lower_bounds = -π
        opt.upper_bounds = +π
        opt.min_objective = cost_fn

        (minf,minx,ret) = optimize(opt, initial_point)
    else
        k = length(ansatz)
        grad = zeros(Float64, k)
        d = Uniform(-π,+π)
        for _=1:num_samples
            x = rand(d, k)
            cost_fn(x, grad)
        end
        (minf, minx, ret)=(nothing, nothing, nothing)
    end

    #if path != nothing
    #    fid = h5open(path, "w")
    #    fid["fn_evals"] = fn_evals
    #    fid["grad_evals"] = grad_evals
    #    close(fid)
    #end

    return (minf, minx, ret, eval_count)
end


"""
    commuting_vqe(...)

Similar to `VQE`. Here, generators in the ansatz which are sums of Pauli strings
are allowed, however, it is assumed that the terms in each generator mutually commute.

This is useful for running QAOA and Fermionic anstaze.
"""
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

    function cost_fn(x, grad)
        return _cost_fn_commuting_vqe(x, grad, ansatz, hamiltonian, fn_evals, grad_evals, eval_count, state, initial_state, tmp, tmp1, tmp2, output_state)
    end

    opt.lower_bounds = -π
    opt.upper_bounds = +π
    opt.min_objective = cost_fn

    cost_fn(initial_point, similar(initial_point))

    (minf,minx,ret) = optimize(opt, initial_point)

    if output_state !== nothing
        cost_fn(minx, similar(minx))
    end

    return (minf, minx, ret, eval_count)
end

function qaoa_ansatz(
    hamiltonian::Operator,
    mixers::Array{Operator,1}
)
    # num_pars == 2*length(mixers)+1
    p = length(mixers)
    ansatz = zip(mixers, repeat([hamiltonian], p))
    ansatz = collect(Iterators.flatten(ansatz))
    #prepend!(ansatz, [hamiltonian])
    return ansatz
end

"""
    QAOA(mixers::Array{Operator,1}, ...)

Run the QAOA algorithm

# Arguments that differ from `VQE`
- `mixers::Array{Operator,1}`: Mixers to use, interleaved with the cost Hamiltonian.
"""
function QAOA(
    hamiltonian::Operator,
    mixers::Array{Operator,1},
    opt::Union{Opt, String},
    initial_point::Array{Float64,1},
    num_qubits::Int64,
    initial_state::Union{Nothing,Array{ComplexF64,1}} = nothing, # Initial state
    path = nothing, # Should be a CSV file
    output_state::Union{Nothing,Array{ComplexF64,1}} = nothing
)
    ansatz = qaoa_ansatz(hamiltonian, mixers)
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

"""
    QAOA(p::Int, ...)

Run the QAOA algorithm with a specified number of layers `p`. Uses the `X_1 + ... X_n` mixer.

# Arguments that differ from `VQE`
- `p::Int`: The number of times to repeat the cost function and mixer in the ansatz.
"""
function QAOA(
    hamiltonian::Operator,
    p::Int,
    opt::Opt,
    initial_point::Array{Float64,1},
    num_qubits::Int64,
    initial_state::Union{Nothing,Array{ComplexF64,1}} = nothing, # Initial state
    path = nothing, # Should be a CSV file
    output_state::Union{Nothing,Array{ComplexF64,1}} = nothing
)
    mixers = [qaoa_mixer(num_qubits) for i=1:p]
    return QAOA(
        hamiltonian,
        mixers,
        opt,
        initial_point,
        num_qubits,
        initial_state = initial_state,
        path = path,
        output_state = output_state
    )
end

"""
    ADAPTHistory

Stores various relevant quantities when running the ADAPT-VQE algorithm.
"""
mutable struct ADAPTHistory
    energy::Array{Float64,1}
    max_grad::Array{Float64,1}
    max_grad_ind::Array{Int64,1}
    grads::Array{Array{Float64, 1}, 1}
    opt_pars::Array{Array{Float64, 1}, 1}
    paulis::Array{Any,1}
    opt_numevals::Array{Any,1}
    opt_state::Array{Any, 1}
end


function Base.firstindex(hist::ADAPTHistory)
    return 1
end


function Base.lastindex(hist::ADAPTHistory)
    return length(hist.energy)
end


function Base.length(hist::ADAPTHistory)
    return Base.lastindex(hist)
end


function Base.getindex(hist::ADAPTHistory, i::Int)
    1<=i<=lastindex(hist) || throw(BoundsError(hist, i))
    return Dict(f => getfield(hist, f)[i] for f in fieldnames(ADAPTHistory))
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
        numevals,
        grad_state = nothing
        )
    
    if grad_state === nothing
        grad_state = state
    end
    push!(hist.grads, [abs(exp_val(com, grad_state, tmp)) for com in comms]) # phase?
    push!(hist.max_grad_ind, argmax(map(x -> abs(x), hist.grads[end])))
    push!(hist.max_grad, abs(hist.grads[end][hist.max_grad_ind[end]]))
    push!(hist.energy, real(exp_val(hamiltonian, state, tmp)))
    push!(hist.opt_pars, opt_pars)
    push!(hist.paulis, pauli_chosen)
    push!(hist.opt_numevals, numevals)
    push!(hist.opt_state, copy(state))
end


"""
    adapt_vqe(...)

Run the ADAPT-VQE algorithm.

# Arguments (that are different from `VQE`)
- `pool::Array{Pauli{T},1}`: List of Pauli operators to use in the pool.
- `callbacks::Array{Function}`: List of callbacks to run at each layer of the algorithm.
- `initial_parameter::Float64 = 0.0`: Initial parameter to use when initializing a new layer.
- `initial_state::Union{Nothing,Array{ComplexF64,1}} = nothing`: Initial state to use for the ansatz.
"""
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
    hist = ADAPTHistory([], [], [], [], [], [], [], [])

    #if path !== nothing
    #    """ in $path the following files will be written
    #        $path/layer_*.csv - indexed by integer
    #        $path/adapt_history.csv
    #    """
    #end

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
                #if path !== nothing
                #    adapt_history_dump!(hist, "$path/adapt_history.csv", num_qubits)
                #end
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


function make_opt(optimizer, point)
    if optimizer isa String
        opt_dict = Dict("name" => optimizer)
    elseif optimizer isa Dict
        opt_dict = optimizer
    else
        throw(ArgumentError("optimizer should be String or Dict"))
    end

    opt = Opt(Symbol(opt_dict["name"]), length(point))
    opt_keys = collect(keys(opt_dict))
    deleteat!(opt_keys,findall(x->x=="name",opt_keys))
    for akey in opt_keys
        setproperty!(opt,Symbol(akey),opt_dict[akey])
    end
    return opt
end


"""
    adapt_qaoa(...)

Run the ADAPT-QAOA algorithm.

# Arguments (that are different from `VQE`)
- `pool::Array{Pauli{T},1}`: List of Pauli operators to use in the pool.
- `callbacks::Array{Function}`: List of callbacks to run at each layer of the algorithm.
- `initial_parameter::Float64 = 0.0`: Initial parameter to use when initializing a new layer.
- `initial_state::Union{Nothing,Array{ComplexF64,1}} = nothing`: Initial state to use for the ansatz.
"""
function adapt_qaoa(
    hamiltonian::Operator,
    pool::Array{Operator,1},
    num_qubits::Int64,
    optimizer::Union{String,Dict},
    callbacks::Array{Function};
    initial_parameter::Float64 = 0.0,
    initial_state::Union{Nothing,Array{ComplexF64,1}} = nothing, # Initial state
    path = nothing,
    tmp::Union{Nothing, Array{ComplexF64,1}} = nothing
)
    #### Initialization
    hist = ADAPTHistory([], [], [], [], [], [], [], [])

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
        push!(comms, commutator(hamiltonian, _op, false))
    end
    mixers = Array{Operator, 1}()
    layer_count = 0

    state = initial_state
    point = Vector{Float64}()
    optimal_point = Vector{Float64}()
    opt_evals = nothing

    while true
        #### Some book-keeping of variables
        grad_state = state
        pauli_ansatz!(hamiltonian.paulis, real(hamiltonian.coeffs)*initial_parameter, grad_state, tmp)
        if length(hist.max_grad_ind) !== 0
            chosen_op = pool[hist.max_grad_ind[end]]
        else
            chosen_op = nothing
        end

        adapt_step!(hist, comms, tmp, state, hamiltonian, optimal_point, chosen_op, opt_evals, grad_state)

        #### Check Convergence
        for c in callbacks
            if c(hist)
                return hist
            end
        end

        #### Get new operator
        push!(mixers, pool[hist.max_grad_ind[end]])

        #### Update point
        _init_beta = 0.0
        _init_gamma = 0.0
        push!(point, _init_beta)
        push!(point, _init_gamma)
        if length(point) != 2*length(mixers) error("Invalid point size") end

        output_state = similar(initial_state)
        min_energy, optimal_point, _, opt_evals = QAOA(hamiltonian, mixers, make_opt(optimizer, point), point, num_qubits, copy(initial_state), nothing, output_state)
        state .= output_state
        point .= optimal_point
    end
end
