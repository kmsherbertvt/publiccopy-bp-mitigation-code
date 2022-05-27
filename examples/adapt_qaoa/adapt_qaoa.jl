using Distributed

println("staring script..."); flush(stdout)
@everywhere using Pkg
@everywhere Pkg.activate("../../")
@everywhere Pkg.instantiate()
@everywhere using AdaptBarren

@everywhere using LinearAlgebra
@everywhere using Test
@everywhere using DataFrames
@everywhere using StatsPlots
@everywhere using StatsBase
@everywhere using IterTools
@everywhere using NLopt
@everywhere using Random
@everywhere using CSV
@everywhere using ProgressBars

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
@everywhere n_max = 10


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

    initial_state = ones(ComplexF64, 2^n) / sqrt(2^n)
    initial_state /= norm(initial_state)
    callbacks = Function[ ParameterStopper(max_pars)]

    result = adapt_qaoa(hamiltonian, pool, n, opt_dict, callbacks; initial_parameter=1e-2, initial_state=initial_state, path=path)
    t_f = time()
    dt = t_f - t_0
    println("Finished n=$n, seed=$seed with alg=$pool_name in $dt seconds"); flush(stdout)

    num_layers = length(result)
    df = DataFrame(seed=[], alg=[], layer=[], err=[], n=[], overlap=[], time=[])
    for k = 1:num_layers
        d = result[k]
        en_err = safe_floor(d[:energy]-gse)
        gse_overlap = ground_state_overlap(ham_vec, d[:opt_state])
        push!(df, Dict(:seed=>seed, :alg=>pool, :layer=>k, :err=>en_err, :n=>n, :overlap=>gse_overlap, :time=>dt))
    end
end

# Parallel Debug
println("Num procs: $(nprocs())")
println("Num workers: $(nworkers())")

# Main Loop
fn_inputs = collect(product(1:num_samples, ["2l", "qaoa"], 4:2:n_max))
println("starting simulations..."); flush(stdout)
#results = map(i -> run_adapt_qaoa(i...), fn_inputs)
results = pmap(i -> run_adapt_qaoa(i...), fn_inputs)

println("concatenating results..."); flush(stdout)
df = vcat(results...)
println("dumping data..."); flush(stdout)
CSV.write("data.csv", df)

#println("Plotting..."); flush(stdout)
### Plotting
#using Plots; gr()
#
#for n_=n_min:2:n_max
#    df_n = filter(:n => n -> n==n_, df)
#    n = n_
#    plot()
#    @df filter(:alg => ==("QAOA"), df_n) plot!(:layer, :err, group=:seed, color=:blue, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=false)
#    @df filter(:alg => ==("ADAPT"), df_n) plot!(:layer, :err, group=:seed, color=:red, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=false)
#    savefig("test_qaoa_comp_$n.pdf")
#
#    plot()
#    gdf_n = groupby(df_n, [:layer, :alg])
#    @df combine(gdf_n, :err => (x -> 10^mean(log10.(safe_floor.(x)))) => :err_mean) plot(:layer, :err_mean, group=:alg, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=true)
#    savefig("test_qaoa_comp_mean_$n.pdf")
#end
#