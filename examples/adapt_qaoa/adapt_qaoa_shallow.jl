println("staring script..."); flush(stdout)
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
max_p = 20
max_pars = 2*max_p
max_grad = 1e-4
path="test_data"
n_min = 4
n_max = 18


function run_qaoa(n, hamiltonian)
    energy_result = []
    ground_state_energy = minimum(real(diag(operator_to_matrix(hamiltonian))))
    for current_p in range(2,max_p)
        mixers = repeat([qaoa_mixer(n)], current_p)
        initial_point = rand(rng, Float64, 2*current_p)
        opt = Opt(Symbol(opt_alg), length(initial_point))
        opt.maxeval = opt_dict["maxeval"]
        initial_state = ones(ComplexF64, 2^n) / sqrt(2^n)
        result = QAOA(hamiltonian, mixers, opt, initial_point, n, initial_state)
        qaoa_energy = result[1]
        en_err = qaoa_energy - ground_state_energy
        push!(energy_result, qaoa_energy)
    end
    return energy_result
end

function run_adapt_qaoa(n, hamiltonian)
    ground_state_energy = minimum(real(diag(operator_to_matrix(hamiltonian))))

    pool = two_local_pool(n)
    pool = map(p -> Operator([p], [1.0]), pool)
    push!(pool, qaoa_mixer(n))

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
for n in ProgressBar(n_min:2:n_max, printing_delay=0.1)
    for i in ProgressBar(1:num_samples, printing_delay=0.1)
        d = n-1
        println("Starting n=$n seed=$i"); flush(stdout)
        t_0 = time()
        hamiltonian = random_regular_max_cut_hamiltonian(n, d)
        gse = minimum(real(diag(operator_to_matrix(hamiltonian))))
        _res_qaoa = run_qaoa(n, hamiltonian);
        hist_adapt, _res_adapt = run_adapt_qaoa(n, hamiltonian);

        lock(lk) do
            #loop over results in adapt and append to df
            for (k,en)=enumerate(hist_adapt.energy)
                push!(df, Dict(:seed=>i, :alg=>"ADAPT", :layer=>k+1, :err=>safe_floor(en-gse), :n=>n))
            end
            #loop over results in qaoa and append to df
            for (k,en)=enumerate(_res_qaoa)
                # double the number of parameters since k here measures p
                push!(df, Dict(:seed=>i, :alg=>"QAOA", :layer=>2*k, :err=>safe_floor(en-gse), :n=>n))
            end

            # Also collect here
            push!(results_adapt, _res_adapt)
            push!(results_qaoa, _res_qaoa)
            CSV.write("data_shallow.csv", df)
        t_f = time()
        dt = t_f - t_0
        println("Finished n=$n seed=$i in $dt seconds"); flush(stdout)
        end
    end
end

println("Done with simulations, dumping data..."); flush(stdout)
CSV.write("data_shallow.csv", df)
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
