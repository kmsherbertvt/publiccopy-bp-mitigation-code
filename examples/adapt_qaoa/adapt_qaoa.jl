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


@everywhere function run_adapt_qaoa(seed, pool_name, n)
    t_0 = time()
    d = n-1
    hamiltonian = random_regular_max_cut_hamiltonian(n, d; seed=seed)

    pool = Vector{Operator}()
    push!(pool, qaoa_mixer(n))
    if pool_name == "2l"
        append!(pool, map(p -> Operator([p], [1.0]), two_local_pool(n)))
    elseif pool_name == "qaoa"
        1
    else
        error("Invalid pool: $pool")
    end

    ham_vec = real(diagonal_operator_to_vector(hamiltonian))
    gse = get_ground_state(hamiltonian)
    gap = get_energy_gap(hamiltonian)

    initial_state = ones(ComplexF64, 2^n) / sqrt(2^n)
    initial_state /= norm(initial_state)
    callbacks = Function[ ParameterStopper(max_pars)]

    result = adapt_qaoa(hamiltonian, pool, n, opt_dict, callbacks; initial_parameter=1e-2, initial_state=initial_state, path=path)
    t_f = time()
    dt = t_f - t_0
    println("Finished n=$n, seed=$seed with alg=$pool_name in $dt seconds"); flush(stdout)

    num_layers = length(result)
    df = DataFrame(seed=[], alg=[], layer=[], err=[], n=[], overlap=[], time=[], rel_err=[])
    for k = 1:num_layers
        d = result[k]
        en_err = safe_floor(d[:energy]-gse)
        rel_en_err = en_err / gap
        gse_overlap = ground_state_overlap(ham_vec, d[:opt_state])
        push!(df, Dict(
            :seed=>seed, 
            :alg=>pool_name, 
            :layer=>k, 
            :err=>en_err,
            :rel_err=>rel_en_err,
            :n=>n, 
            :overlap=>gse_overlap, 
            :time=>dt
            ))
    end
    return df
end

# Parallel Debug
println("Num procs: $(nprocs())")
println("Num workers: $(nworkers())")

# Main Loop
global df = DataFrame(seed=[], alg=[], layer=[], err=[], n=[], overlap=[], time=[], rel_err=[])
for n=4:2:n_max
    fn_inputs = collect(product(1:num_samples, ["2l", "qaoa"], [n]))
    println("starting simulations..."); flush(stdout)
    #results = map(i -> run_adapt_qaoa(i...), fn_inputs)
    t_0 = time()
    results = pmap(i -> run_adapt_qaoa(i...), fn_inputs)
    t_f = time()
    dt = t_f - t_0
    println("The walltime for the n=$n samples was $dt seconds")

    println("concatenating results..."); flush(stdout)
    global df = vcat(df, results...)
    println("dumping data..."); flush(stdout)
    CSV.write("data.csv", df)
end
