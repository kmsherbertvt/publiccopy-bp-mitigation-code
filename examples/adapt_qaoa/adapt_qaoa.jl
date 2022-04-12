using LinearAlgebra
using Test
using DataFrames
using AdaptBarren
using StatsPlots
using NLopt
using Random
using ProgressBars

Random.seed!(42)
rng = MersenneTwister(14)

# Hyperparameters
num_samples = 20
opt_alg = "LD_LBFGS"
opt_dict = Dict("name" => opt_alg, "maxeval" => 1500)
max_p = 15
max_pars = 2*max_p
max_grad = 1e-4
path="test_data"

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

n = 6
d = 5

lk = ReentrantLock()
println("Starting simulations...")
Threads.@threads for i in ProgressBar(1:num_samples, printing_delay=0.1)
    hamiltonian = random_regular_max_cut_hamiltonian(n, d)
    gse = minimum(real(diag(operator_to_matrix(hamiltonian))))
    _res_qaoa = run_qaoa(n, hamiltonian);
    hist_adapt, _res_adapt = run_adapt_qaoa(n, hamiltonian);

    lock(lk) do
        #loop over results in adapt and append to df
        for (k,en)=enumerate(hist_adapt.energy)
            push!(df, Dict(:seed=>i, :alg=>"ADAPT", :layer=>k+1, :err=>en-gse, :n=>n))
        end
        #loop over results in qaoa and append to df
        for (k,en)=enumerate(_res_qaoa)
            # double the number of parameters since k here measures p
            push!(df, Dict(:seed=>i, :alg=>"QAOA", :layer=>2*k, :err=>en-gse, :n=>n))
        end

        # Also collect here
        push!(results_adapt, _res_adapt)
        push!(results_qaoa, _res_qaoa)
    end

end
println("Done with simulations, plotting...")

function safe_floor(x::Float64, eps=1e-15, delta=1e-8)
    if x <= -delta error("Too negative...") end
    if x <= 0.0
        return eps
    else
        return x
    end
end

### Plotting
using Plots; gr()
plot()
@df filter(:alg => ==("QAOA"), df) plot!(:layer, :err, group=:seed, color=:blue, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=false)
@df filter(:alg => ==("ADAPT"), df) plot!(:layer, :err, group=:seed, color=:red, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=false)
savefig("test_qaoa_comp.pdf")

plot()
gdf = groupby(df, [:layer, :alg])
function mean(x) return sum(x)/length(x) end
@df combine(gdf, :err => (x -> 10^mean(log10.(safe_floor.(x)))) => :err_mean) plot(:layer, :err_mean, group=:alg, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=false)
savefig("test_qaoa_comp_mean.pdf")