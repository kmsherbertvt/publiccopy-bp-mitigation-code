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
@everywhere num_samples = 50
@everywhere num_point_samples = 200
@everywhere max_num_qubits = 12
@everywhere max_grad = 1e-6
@everywhere max_adapt_layers = 50
@everywhere vqe_sampling_depths = vcat(1:10,10:5:50,60:10:100,150:50:400)
@everywhere adapt_sampling_depths = vcat(1:10,10:5:50,60:10:100,150:50:400)
@everywhere ball_sampling_radii = collect(0:0.2:5)
@everywhere opt_alg = "LD_LBFGS"
@everywhere opt_dict = Dict("name" => opt_alg, "maxeval" => 1500)
@everywhere data_path_prefix = "data/data"
@everywhere data_path_suffix = "csv"

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
    callbacks = Function[MaxGradientStopper(max_grad), DeltaYStopper(), ParameterStopper(max_adapt_layers)]
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
    callbacks = Function[MaxGradientStopper(max_grad), DeltaYStopper(), ParameterStopper(max_adapt_layers)]
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


# Debug
#run_instance(42, 3, "adapt_vqe_2l", 20)
#run_instance(42, 3, "adapt_qaoa_2l", 20)
#run_instance(42, 3, "qaoa", 20)
#run_instance(42, 3, "vqe", 20)


@everywhere function run_instance(seed, n, method, d)
    rng = MersenneTwister(seed)
    hamiltonian = random_regular_max_cut_hamiltonian(n, n-1; rng=rng, weighted=true)
    initial_state = uniform_state(n)

    t_0 = time()
    if method == "adapt_vqe_2l"
        pool = two_local_pool(n)
        append!(pool, one_local_pool_from_axes(n, [1,2,3]))
        res = main_adapt(n, hamiltonian, pool)
    elseif method == "adapt_qaoa_2l"
        pool = two_local_pool(n)
        append!(pool, one_local_pool_from_axes(n, [1,2,3]))
        pool = map(p -> Operator([p], [1.0]), pool)
        push!(pool, qaoa_mixer(n))
        res = main_adapt_qaoa(n, hamiltonian, pool)
    elseif method == "qaoa"
        pool = [qaoa_mixer(n)]
        res = main_adapt_qaoa(n, hamiltonian, pool)
    elseif method == "vqe"
        ansatz = random_two_local_ansatz(n, d; rng=rng)
        res = main_vqe(n, hamiltonian, ansatz, rng)
        res["ansatz"] = map(p->Operator(p),ansatz)
    else
        error("unrecognized method: $method")
    end
    res["sim_dur"] = time() - t_0
    
    if method === "vqe"
        _depths = [d]
    else
        ans_len = length(res["ansatz"])
        _depths = copy(adapt_sampling_depths)
        _depths = vcat(filter(x -> x<ans_len, _depths), ans_len)
    end
    # Whole space sampling
    t_0 = time()
    sample_pairs = [(dp, sample_points(hamiltonian, res["ansatz"][1:dp], initial_state, num_point_samples; rng=rng)...) for dp=_depths]
    sampled_energies_list = [ens for (_, ens, _)=sample_pairs]
    sampled_grads_list = [grads for (_, _, grads)=sample_pairs]
    sampled_depths = [dp for (dp, _, _)=sample_pairs]
    res["samp_dur"] = time() - t_0

    # Ball sampling
    t_0 = time()
    optimal_point = res["opt_pars"][end]
    ball_sample_pairs = [(rp, sample_points(hamiltonian, res["ansatz"], initial_state, Int(floor(sqrt(num_point_samples))); rng=rng, dist=rp, point=optimal_point)...) for rp=ball_sampling_radii]
    ball_sampled_energies_list = [ens for (_, ens, _)=ball_sample_pairs]
    ball_sampled_grads_list = [grads for (_, _, grads)=ball_sample_pairs]
    ball_sampled_rads = [rp for (rp, _, _)=ball_sample_pairs]
    res["ball_samp_dur"] = time() - t_0

    return Dict(
        "result_dict" => res,
        "sampled_energies_list" => sampled_energies_list,
        "sampled_grads_list" => sampled_grads_list,
        "sampled_depths" => sampled_depths,
        "ball_sampled_energies_list" => ball_sampled_energies_list,
        "ball_sampled_grads_list" => ball_sampled_grads_list,
        "ball_sampled_rads" => ball_sampled_rads,
        "pars_dict" => Dict("seed" => seed, "method" => method, "n" => n, "d" => d),
    )
end

# Parallel Info
println("Num procs: $(nprocs())"); flush(stdout)
println("Num workers: $(nworkers())"); flush(stdout)

# Set up problem problem_instances
problem_instances = []
for seed=1:num_samples
    for method=["adapt_vqe_2l", "adapt_qaoa_2l", "qaoa", "vqe"]
        if method == "vqe"
            append!(problem_instances, [[seed, method, d] for d in vqe_sampling_depths])
        else
            append!(problem_instances, [[seed, method, missing]])
        end
    end
end
println("There are $(length(problem_instances)) problem problem_instances for each number of qubits"); flush(stdout)

# Main Loop
for n=4:2:max_num_qubits
    println("Starting n=$n qubits"); flush(stdout)
    t_0 = time()
    results = pmap(inst -> run_instance(inst[1], n, inst[2], inst[3]), problem_instances)
    println("Dumping"); flush(stdout)
    df_res   = DataFrame(n=[], method=[], d=[], seed=[], 
        energies=[], 
        sim_dur=[],
        samp_dur=[],
        ball_samp_dur=[],
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
    df_grads_ball = DataFrame(n=[], method=[], rad=[], seed=[],
        grad=[]
        )
    df_ens_ball   = DataFrame(n=[], method=[], rad=[], seed=[],
        en=[]
        )
    for res in results
        _res_dict = res["result_dict"]
        pop!(_res_dict, "ansatz")
        _pars_dict = res["pars_dict"]

        push!(df_res, merge(_pars_dict, _res_dict))
        # Whole sampling
        _l = length(res["ball_sampled_grads_list"])
        for (d,_energies,_grads)=zip(res["sampled_depths"], res["sampled_energies_list"], res["sampled_grads_list"])
            for e in _energies
                _d_out = merge(Dict("en" => e), _pars_dict)
                _d_out["d"] = d
                push!(df_ens, _d_out)
            end
            for g in _grads
                _d_out = merge(Dict("grad" => g), _pars_dict)
                _d_out["d"] = d
                push!(df_grads, _d_out)
            end
        end

        # Ball sampling
        """ This is taking a while, this could potentially be improved in terms
        of performance by doing something like `df_tmp = DataFrame(rad=res["ball_sampled_rads"], ...)`
        and then `df_tmp["n"] = n`, etc., then concatenating then DataFrames.
        """
        _l = length(res["ball_sampled_grads_list"])
        for (rad,_energies,_grads)=zip(res["ball_sampled_rads"], res["ball_sampled_energies_list"], res["ball_sampled_grads_list"])
            for e in _energies
                _d_out = merge(Dict("en" => e), _pars_dict)
                _d_out["rad"] = rad
                delete!(_d_out, "d")
                push!(df_ens_ball, _d_out)
            end
            for g in _grads
                _d_out = merge(Dict("grad" => g), _pars_dict)
                _d_out["rad"] = rad
                delete!(_d_out, "d")
                push!(df_grads_ball, _d_out)
            end
        end
    end

    CSV.write("$(data_path_prefix)_res_$(n).$(data_path_suffix)", df_res)
    CSV.write("$(data_path_prefix)_en_$(n).$(data_path_suffix)", df_ens)
    CSV.write("$(data_path_prefix)_grad_$(n).$(data_path_suffix)", df_grads)

    CSV.write("$(data_path_prefix)_ball_en_$(n).$(data_path_suffix)", df_ens_ball)
    CSV.write("$(data_path_prefix)_ball_grad_$(n).$(data_path_suffix)", df_grads_ball)

    println("Finished n=$n qubits in $(time()-t_0) seconds"); flush(stdout)
end
