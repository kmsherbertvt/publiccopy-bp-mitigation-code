using LinearAlgebra
using Test
using AdaptBarren
using NLopt
using Random
using ProgressBars

rng = MersenneTwister(14)

function _id_x_str(n::Int, k::Int)
    s = repeat(["I"], n)
    s[k] = "X"
    return join(s)
end

function qaoa_mixer(n::Int)
    paulis = [pauli_string_to_pauli(_id_x_str(n, k)) for k in range(1,n)]
    coeffs = repeat([1.0], n)
    return Operator(paulis, coeffs)
end

num_samples = 20
n = 6
d = 3
opt_alg = "LD_LBFGS"
max_p = 20
max_pars = 2*max_p+1
max_grad = 1e-4
path="test_data"

function run_qaoa(hamiltonian)
    #hamiltonian = random_regular_max_cut_hamiltonian(n, d)
    energy_result = []
    ground_state_energy = minimum(real(diag(operator_to_matrix(hamiltonian))))
    for current_p in range(2,max_p)
        mixers = repeat([qaoa_mixer(n)], current_p)
        initial_point = rand(rng, Float64, 2*current_p+1)
        opt = Opt(Symbol(opt_alg), length(initial_point))
        initial_state = ones(ComplexF64, 2^n) / sqrt(2^n)
        result = QAOA(hamiltonian, mixers, opt, initial_point, n, initial_state)
        qaoa_energy = result[1]
        en_err = qaoa_energy - ground_state_energy
        push!(energy_result, en_err)
    end
    return energy_result
end

function run_adapt_qaoa(hamiltonian)
    #hamiltonian = random_regular_max_cut_hamiltonian(n, d)
    ground_state_energy = minimum(real(diag(operator_to_matrix(hamiltonian))))

    pool = two_local_pool(n)
    pool = map(p -> Operator([p], [1.0]), pool)
    push!(pool, qaoa_mixer(n))

    initial_state = ones(ComplexF64, 2^n) / sqrt(2^n)
    initial_state /= norm(initial_state)
    callbacks = Function[ParameterStopper(max_pars), MaxGradientStopper(max_grad)]

    result = adapt_qaoa(hamiltonian, pool, n, opt_alg, callbacks; initial_parameter=1e-2, initial_state=initial_state, path=path)

    adapt_qaoa_energy = last(result.energy)
    en_err = adapt_qaoa_energy - ground_state_energy

    return result.energy - repeat([ground_state_energy], length(result.energy))
end

results_qaoa = []
results_adapt = []

lk = ReentrantLock()

function safe_floor(x, eps = 1e-5, delta = 1e-10)
    if x <= -eps error("Too negative: $x < -$eps") end
    if x <= 0.0 return delta end
    return x
end

Threads.@threads for i in ProgressBar(1:num_samples, printing_delay=0.1)
    hamiltonian = random_regular_max_cut_hamiltonian(n, d)
    _res_qaoa = run_qaoa(hamiltonian);
    _res_adapt = run_adapt_qaoa(hamiltonian);

    _res_qaoa = map(safe_floor, _res_qaoa)
    _res_adapt = map(safe_floor, _res_adapt)

    lock(lk) do
        push!(results_adapt, _res_adapt)
        push!(results_qaoa, _res_qaoa)
    end
end

### Plotting
using Plots; pgfplotsx()

for res in results_adapt
    plot!(1:length(res), res, c=:blue, yaxis=:log)
end

for res in results_qaoa
        plot!(1:length(res), res, c=:red, yaxis=:log)
end

savefig("test_qaoa_comp.pdf")
