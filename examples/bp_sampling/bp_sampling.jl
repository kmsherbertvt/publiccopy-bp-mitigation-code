using Distributed

println("staring script..."); flush(stdout)
@everywhere using Pkg
@everywhere Pkg.activate("../../")
@everywhere Pkg.instantiate()
@everywhere using AdaptBarren

@everywhere using LinearAlgebra
@everywhere using Test
@everywhere using DataFrames
@everywhere using StatsBase
@everywhere using Distributions
@everywhere using IterTools
@everywhere using NLopt
@everywhere using Random
@everywhere using CSV

#@everywhere Random.seed!(42)
#@everywhere rng = MersenneTwister(14)

# Hyperparameters
@everywhere num_samples = 20
@everywhere num_point_samples = 500
@everywhere max_num_qubits = 8
@everywhere max_grad = 1e-4
@everywhere vqe_sampling_depths = [20, 40, 60]
@everywhere opt_alg = "LD_LBFGS"
@everywhere opt_dict = Dict("name" => opt_alg, "maxeval" => 1500)


@everywhere function analyze_results!(res_dict, ham, opt_states)
    gse = get_ground_state(ham)
    gap = get_energy_gap(ham)
    ham_vec = real(diagonal_operator_to_vector(ham))
    res_dict["energy_errors"] = res_dict["energies"] .- gse
    res_dict["approx_ratio"] = res_dict["energies"] ./ gse
    res_dict["relative_error"] = abs.(res_dict["energies"]) ./ (gap)
    res_dict["ground_state_overlaps"] = map(s -> ground_state_overlap(ham_vec, s), opt_states)

    res_dict["final_energy_errors"] = res_dict["energy_errors"][end]
    res_dict["final_approx_ratio"] = res_dict["approx_ratio"][end]
    res_dict["final_relative_error"] = res_dict["relative_error"][end]
    res_dict["final_ground_state_overlaps"] = res_dict["ground_state_overlaps"][end]

    return res_dict
end


@everywhere function main_adapt(n, ham, pool::Array{Pauli{T},1}) where T<:Unsigned
    callbacks = Function[MaxGradientStopper(max_grad), DeltaYStopper()]
    initial_state = uniform_state(n)

    res = adapt_vqe(ham, pool, n, opt_dict, callbacks; initial_state=initial_state)
    res_dict = Dict{Any,Any}(
        "energies" => res.energy,
        "ansatz" => map(p -> Operator(p), filter(x -> x !== nothing, res.paulis)),
        "max_grads" => res.max_grad,
        "opt_pars" => res.opt_pars,
    )
    analyze_results!(res_dict, ham, res.opt_state)
    return res_dict
end


@everywhere function main_adapt_qaoa(n, ham, pool::Array{Operator,1})
    callbacks = Function[MaxGradientStopper(max_grad), DeltaYStopper()]
    initial_state = uniform_state(n)

    res = adapt_qaoa(ham, pool, n, opt_dict, callbacks; initial_parameter=1e-2, initial_state=initial_state)
    res.paulis
    mixers = map(p->Operator(p),filter(p -> p !== nothing,res.paulis))
    ansatz = qaoa_ansatz(ham, mixers)
    res_dict =Dict{Any,Any}(
        "energies" => res.energy,
        "ansatz" => ansatz,
        "max_grads" => res.max_grad,
        "opt_pars" => res.opt_pars,
    )
    analyze_results!(res_dict, ham, res.opt_state)
    return res_dict
end


@everywhere function main_vqe(n, ham, ansatz::Array{Pauli{T},1}, rng) where T<:Unsigned
    initial_state = uniform_state(n)
    psi = copy(initial_state)
    initial_point = rand(rng, Uniform(-pi, +pi), length(ansatz))

    min_en, opt_pt, _, _ = VQE(ham, ansatz, make_opt(opt_dict, initial_point), initial_point, n, initial_state)
    pauli_ansatz!(ansatz, opt_pt, psi, similar(initial_state))

    res_dict = Dict{Any,Any}(
        "energies" => [min_en],
        "ansatz" => map(p->Operator(p),ansatz),
        "max_grads" => Array{Float64,1}(),
        "opt_pars" => [opt_pt],
    )
    analyze_results!(res_dict, ham, [psi])
    return res_dict
end


@everywhere function run_instance(seed, n, method, d)
    rng = MersenneTwister(seed)
    hamiltonian = random_regular_max_cut_hamiltonian(n, n-1; rng=rng, weighted=true)
    initial_state = uniform_state(n)

    if method == "adapt_vqe_2l"
        pool = two_local_pool(n)
        res = main_adapt(n, hamiltonian, pool)
    elseif method == "adapt_qaoa_2l"
        pool = map(p -> Operator([p], [1.0]), two_local_pool(n))
        push!(pool, qaoa_mixer(n))
        res = main_adapt_qaoa(n, hamiltonian, pool)
    elseif method == "qaoa"
        pool = [qaoa_mixer(n)]
        res = main_adapt_qaoa(n, hamiltonian, pool)
    elseif method == "vqe"
        ansatz = random_two_local_ansatz(n, d; rng=rng)
        res = main_vqe(n, hamiltonian, ansatz, rng)
    else
        error("unrecognized method: $method")
    end
    
    sampled_energy, sampled_grads = sample_points(hamiltonian, res["ansatz"], initial_state, num_point_samples; rng=rng)

    return Dict(
        "result_dict" => res,
        "sampled_energies" => sampled_energy,
        "sampled_grads" => sampled_grads,
        "pars_dict" => Dict("seed" => seed, "method" => method, "n" => n, "d" => d),
    )
end

# Debug
#run_instance(42, 3, "adapt_vqe_2l", 20)
#run_instance(42, 3, "adapt_qaoa_2l", 20)
#run_instance(42, 3, "qaoa", 20)
#run_instance(42, 3, "vqe", 20)


# Parallel Info
println("Num procs: $(nprocs())"); flush(stdout)
println("Num workers: $(nworkers())"); flush(stdout)

# Set up problem instances
instances = []
for seed=1:num_samples
    for method=["adapt_vqe_2l", "adapt_qaoa_2l", "qaoa", "vqe"]
        if method == "vqe"
            append!(instances, [[seed, method, d] for d in vqe_sampling_depths])
        else
            append!(instances, [[seed, method, missing]])
        end
    end
end
println("There are $(length(instances)) problem instances for each number of qubits"); flush(stdout)

# Main Loop
for n=4:2:max_num_qubits
    println("Starting n=$n qubits"); flush(stdout)
    t_0 = time()
    results = pmap(inst -> run_instance(inst[1], n, inst[2], inst[3]), instances)
    println("Dumping"); flush(stdout)
    df_res   = DataFrame(n=[], method=[], d=[], seed=[], 
        energies=[], 
        max_grads=[], 
        opt_pars=[], 
        energy_errors=[], 
        approx_ratio=[], 
        relative_error=[],
        ground_state_overlaps=[],
        final_energy_errors=[],
        final_approx_ratio=[],
        final_relative_error=[],
        final_ground_state_overlaps=[]
        )
    df_grads = DataFrame(n=[], method=[], d=[], seed=[],
        grad=[]
        )
    df_ens   = DataFrame(n=[], method=[], d=[], seed=[],
        en=[]
        )
    for res in results
        _res_dict = res["result_dict"]
        pop!(_res_dict, "ansatz")
        _pars_dict = res["pars_dict"]
        _energies = res["sampled_energies"]
        _grads = res["sampled_grads"]

        push!(df_res, merge(_pars_dict, _res_dict))
        for e in _energies
            push!(df_ens, merge(Dict("en" => e), _pars_dict))
        end
        for g in _grads
            push!(df_grads, merge(Dict("grad" => g), _pars_dict))
        end
    end

    CSV.write("data_res_$(n).csv", df_res)
    CSV.write("data_en_$(n).csv", df_ens)
    CSV.write("data_grad_$(n).csv", df_grads)

    println("Finished n=$n qubits in $(time()-t_0) seconds"); flush(stdout)
end