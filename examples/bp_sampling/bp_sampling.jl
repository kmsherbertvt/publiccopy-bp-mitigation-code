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
@everywhere using IterTools
@everywhere using NLopt
@everywhere using Random
@everywhere using CSV

@everywhere Random.seed!(42)
@everywhere rng = MersenneTwister(14)

# Hyperparameters
@everywhere num_samples = 20
@everywhere opt_alg = "LD_LBFGS"
@everywhere opt_dict = Dict("name" => opt_alg, "maxeval" => 1500)
@everywhere max_p = 10
@everywhere max_pars = 2*max_p
@everywhere max_grad = 1e-4
@everywhere path="test_data"
@everywhere n_min = 4
@everywhere n_max = 14
@everywhere num_grad_samples = 500

# Main functions
""" These should take hyperparameter inputs and return dictionary
outputs with results for analysis and sampling
"""

@everywhere function analyze_results!(res_dict, ham)
    gse = get_ground_state(ham)
    gap = get_energy_gap(ham)
    res_dict["energy_errors"] = res_dict["energies"] .- gse
    res_dict["approx_ratio"] = res_dict["energies"] ./ gse
    res_dict["relative_error"] = abs(res_dict["energies"]) ./ (gap)
    res_dict["ground_state_overlaps"] = map(s -> ground_state_overlap(ham, s), res_dict["opt_states"])
    return res_dict
end


@everywhere function main_adapt(n, ham, pool::Array{Pauli{T},1}) where T<:Unsigned
    callbacks = []
    initial_state = uniform_state(n)

    res = adapt_vqe(ham, pool, n, opt_dict, callbacks; initial_state=initial_state)
    res_dict = Dict(
        "energies" => res.energy,
        "ansatz" => res.paulis,
        "max_grads" => res.max_grad,
        "opt_pars" => res.opt_pars,
        "opt_states" => res.opt_state
    )
    analyze_results!(res_dict, ham)
    return res_dict
end

@everywhere function main_adapt_qaoa(n, ham, pool::Array{Operator,1})
    callbacks = []
    initial_state = uniform_state(n)

    res = adapt_qaoa(ham, pool, n, opt_dict, callbacks; initial_parameter=1e-2, initial_state=initial_state)
    res_dict =Dict(
        "energies" => res.energy,
        "ansatz" => qaoa_ansatz(ham, res.paulis),
        "max_grads" => res.max_grad,
        "opt_pars" => res.opt_pars,
        "opt_states" => res.opt_state
    )
    analyze_results!(res_dict, ham)
    return res_dict
end

@everywhere function main_vqe(n, ham, ansatz::Array{Pauli{T},1}, rng) where T<:Unsigned
    initial_state = uniform_state(n)
    psi = copy(initial_state)
    initial_point = rand(rng, Uniform(-pi, +pi), length(ansatz))

    min_en, opt_pt, _, _ = VQE(ham, ansatz, make_opt(opt_dict, initial_point), initial_point, n, initial_state=initial_state)
    pauli_ansatz!(ansatz, opt_pt, psi, similar(initial_state))

    res_dict = Dict(
        "energies" => [min_en],
        "ansatz" => ansatz,
        "max_grads" => Array{Float64,1}(),
        "opt_pars" => [opt_pt],
        "opt_states" => [psi]
    )
    analyze_results!(res_dict, ham)
    return res_dict
end


# Main sampling
""" This should take in inputs based on results from (ADAPT-)VQE
runs and then return statistics on the sampling.
"""
@everywhere function run_sampling()
    VQE(num_samples=num_grad_samples)
end

# Main function
@everywhere function run_instance(seed, n, method)
    rng = MersenneTwister(seed)
    hamiltonian = random_regular_max_cut_hamiltonian(n, n-1; rng=rng, weighted=true)
    initial_state = uniform_state(n)



    sample_points(hamiltonian, ansatz, initial_state, num_samples; rng=rng)
end

# Parallel Debug
println("Num procs: $(nprocs())")
println("Num workers: $(nworkers())")

# Main Loop
global df = DataFrame(seed=[], alg=[], layer=[], err=[], n=[], overlap=[], time=[], rel_err=[], apprx=[])
