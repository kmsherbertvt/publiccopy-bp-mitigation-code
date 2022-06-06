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
@everywhere function main_adapt(n, ham, pool)
    pool = Array{Pauli{T},1}() ...
    callbacks = []
    initial_state = uniform_state(n)

    res = adapt_vqe(ham, pool, n, opt_dict, callbacks; initial_state=initial_state)
    return energies, ansatz
end

@everywhere function main_adapt_qaoa(n, ham, pool)
    pool = Array{Operator,1}() ...
    callbacks = []
    initial_state = uniform_state(n)

    res = adapt_qaoa(ham, pool, n, opt_dict, callbacks; initial_parameter=1e-2, initial_state=initial_state)
    return energies, ansatz
end

@everywhere function main_vqe(n, ham, ansatz, rng)
    ansatz = Array{Pauli{T},1}() ...
    initial_state = uniform_state(n)
    initial_point = rand(rng, Uniform(-pi, +pi), length(ansatz))

    res = VQE(ham, ansatz, make_opt(opt_dict, initial_point), initial_point, n, initial_state=initial_state)
    return energies, ansatz
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
