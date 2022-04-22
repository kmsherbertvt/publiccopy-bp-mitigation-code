using LinearAlgebra
using Test
using DataFrames
using AdaptBarren
using StatsPlots
using NLopt
using Random
using CSV
using ProgressBars

Random.seed!(42)
rng = MersenneTwister(14)

function mean(x) return sum(x)/length(x) end
function safe_floor(x::Float64, eps=1e-15, delta=1e-8)
    if x <= -delta error("Too negative...") end
    if x <= 0.0
        return eps
    else
        return x
    end
end

# Hyperparameters
num_samples = 20
opt_alg = "LD_LBFGS"
opt_dict = Dict("name" => opt_alg, "maxeval" => 1500)
max_p = 40
max_pars = 2*max_p
max_grad = 1e-4
path="test_data"
n_min = 4
n_max = 12


function run_adapt_qaoa(n, hamiltonian, pool_name)
    ground_state_energy = minimum(real(diag(operator_to_matrix(hamiltonian))))

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

    adapt_qaoa_energy = last(result.energy)
    en_err = adapt_qaoa_energy - ground_state_energy

    return result, (result.energy - repeat([ground_state_energy], length(result.energy)))
end

# Main Loop
results_qaoa = []
results_adapt = []
df = DataFrame(seed=[], alg=[], layer=[], err=[], n=[])

lk = ReentrantLock()
println("Starting simulations..."); flush(stdout)
Threads.@threads for i in ProgressBar(1:num_samples, printing_delay=0.1)
    for n=reverse(n_min:2:n_max)
        d = n-1
        hamiltonian = random_regular_max_cut_hamiltonian(n, d)
        gse = minimum(real(diag(operator_to_matrix(hamiltonian))))
        println("Starting ADAPT-QAOA, sample=$i, n=$n"); flush(stdout)
        t_0 = time()
        hist_adapt, _res_adapt = run_adapt_qaoa(n, hamiltonian, "2l");
        t_f = time()
        dt = t_f - t_0
        println("ADAPT-QAOA took $dt seconds on sample=$i, n=$n"); flush(stdout)

        println("Starting QAOA, sample=$i, n=$n"); flush(stdout)
        t_0 = time()
        hist_qaoa, _res_qaoa = run_adapt_qaoa(n, hamiltonian, "qaoa");
        t_f = time()
        dt = t_f - t_0
        println("QAOA took $dt seconds on sample=$i, n=$n"); flush(stdout)

        lock(lk) do
            #loop over results in adapt and append to df
            for (k,en)=enumerate(hist_adapt.energy)
                push!(df, Dict(:seed=>i, :alg=>"ADAPT", :layer=>k+1, :err=>safe_floor(en-gse), :n=>n))
            end
            #loop over results in qaoa and append to df
            for (k,en)=enumerate(hist_qaoa.energy)
                # double the number of parameters since k here measures p
                push!(df, Dict(:seed=>i, :alg=>"QAOA", :layer=>k+1, :err=>safe_floor(en-gse), :n=>n))
            end

            # Also collect here
            push!(results_adapt, _res_adapt)
            push!(results_qaoa, _res_qaoa)
            CSV.write("data.csv", df)
        end
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
