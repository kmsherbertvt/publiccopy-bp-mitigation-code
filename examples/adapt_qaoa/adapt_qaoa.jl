println("staring script..."); flush(stdout)
using LinearAlgebra
using Test
using DataFrames
using AdaptBarren
using StatsPlots
using StatsBase
using NLopt
using Random
using CSV
using ProgressBars

Random.seed!(42)
rng = MersenneTwister(14)

# Hyperparameters
num_samples = 20
opt_alg = "LD_LBFGS"
opt_dict = Dict("name" => opt_alg, "maxeval" => 1500)
max_p = 10
max_pars = 2*max_p
max_grad = 1e-4
path="test_data"
n_min = 4
n_max = 18


function run_adapt_qaoa(n, hamiltonian, pool_name)
    pool = Vector{Operator}()
    push!(pool, qaoa_mixer(n))
    if pool_name == "2l"
        append!(pool, map(p -> Operator([p], [1.0]), two_local_pool(n)))
    elseif pool_name == "qaoa"
        1
    else
        error("Invalid pool: $pool")
    end

    initial_state = ones(ComplexF64, 2^n) / sqrt(2^n)
    initial_state /= norm(initial_state)
    callbacks = Function[ ParameterStopper(max_pars)]

    result = adapt_qaoa(hamiltonian, pool, n, opt_dict, callbacks; initial_parameter=1e-2, initial_state=initial_state, path=path)

    return result
end

# Main Loop
df = DataFrame(seed=[], alg=[], layer=[], err=[], n=[], overlap=[])

println("Starting simulations..."); flush(stdout)
for n in ProgressBar(n_min:2:n_max, printing_delay=0.1)
    for i in ProgressBar(1:num_samples, printing_delay=0.1)
        d = n-1
        println("Starting n=$n seed=$i"); flush(stdout)
        t_0 = time()
        hamiltonian = random_regular_max_cut_hamiltonian(n, d)
        ham_vec = real(diagonal_operator_to_vector(hamiltonian))
        gse = get_ground_state(hamiltonian)

        println("Starting ADAPT-QAOA, sample=$i, n=$n"); flush(stdout)
        t_0 = time()
        hist_adapt = run_adapt_qaoa(n, hamiltonian, "2l");
        t_f = time()
        dt = t_f - t_0
        println("ADAPT-QAOA took $dt seconds on sample=$i, n=$n"); flush(stdout)

        println("Starting QAOA, sample=$i, n=$n"); flush(stdout)
        t_0 = time()
        hist_qaoa = run_adapt_qaoa(n, hamiltonian, "qaoa");
        t_f = time()
        dt = t_f - t_0
        println("QAOA took $dt seconds on sample=$i, n=$n"); flush(stdout)

        for (alg,hist)=zip(["ADAPT","QAOA"],[hist_adapt,hist_qaoa])
            num_layers = length(hist)
            for k = 1:num_layers
                d = hist[k]
                println(d)
                en_err = safe_floor(d[:energy]-gse)
                gse_overlap = ground_state_overlap(ham_vec, d[:opt_state])
                push!(df, Dict(:seed=>i, :alg=>alg, :layer=>k, :err=>en_err, :n=>n, :overlap=>gse_overlap))
            end
        end
        CSV.write("data.csv", df)

        t_f = time()
        dt = t_f - t_0
        println("Finished n=$n seed=$i in $dt seconds"); flush(stdout)
    end
end

println("Done with simulations, dumping data..."); flush(stdout)
CSV.write("data.csv", df)
println("Plotting..."); flush(stdout)

### Plotting
using Plots; gr()

for n_=n_min:2:n_max
    df_n = filter(:n => n -> n==n_, df)
    n = n_
    plot()
    @df filter(:alg => ==("QAOA"), df_n) plot!(:layer, :err, group=:seed, color=:blue, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=false)
    @df filter(:alg => ==("ADAPT"), df_n) plot!(:layer, :err, group=:seed, color=:red, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=false)
    savefig("test_qaoa_comp_$n.pdf")

    plot()
    gdf_n = groupby(df_n, [:layer, :alg])
    @df combine(gdf_n, :err => (x -> 10^mean(log10.(safe_floor.(x)))) => :err_mean) plot(:layer, :err_mean, group=:alg, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=true)
    savefig("test_qaoa_comp_mean_$n.pdf")
end
