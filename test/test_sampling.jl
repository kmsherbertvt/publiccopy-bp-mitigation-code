using LinearAlgebra
using StatsBase
using Test
using Random
using AdaptBarren
using Distributions
using Optim
using LineSearches


@testset "Test Cost Function and Gradient Sampling" begin
    rng = MersenneTwister(42);
    n = 4
    k = 10
    num_samples = 50
    pool = map(p -> Operator([p], [1.0]), two_local_pool(n))
    push!(pool, qaoa_mixer(n))
    ansatz = sample(pool, k; replace = true)
    hamiltonian = random_regular_max_cut_hamiltonian(n, n-1; weighted = true)
    initial_state = uniform_state(n)

    point = rand(rng, Uniform(-pi, +pi), length(ansatz))
    points_to_sample = Vector{Vector{Float64}}()
    append!(points_to_sample, [])

    l_min = []
    l_max = []

    for _=1:2
        result_energies, result_gradients, _, _ = sample_points(hamiltonian, ansatz, initial_state, num_samples; rng=rng, dist=0.0, point=point)

        delta = maximum(result_energies) - minimum(result_energies)
        @test abs(delta) <= 1e-6

        push!(l_min, minimum(result_energies))
        push!(l_max, maximum(result_energies))
    end

    @test abs(l_min[1] - l_min[2]) <= 1e-6
    @test abs(l_max[1] - l_max[2]) <= 1e-6
end

@testset "Point sampling gradient minimum" begin
    for seed=1:200
        rng = MersenneTwister(seed)
        num_samples = 1
        n = 4
        use_norm = true

        pool = two_local_pool(n)
        append!(pool, one_local_pool_from_axes(n, [1,2,3]))
        pool = map(p -> Operator([p], [1.0]), pool)
        push!(pool, qaoa_mixer(n))

        hamiltonian = random_regular_max_cut_hamiltonian(n, n-1; weighted = true, rng=rng)

        initial_state = uniform_state(n)
        dist = 0.0
        opt = Optim.LBFGS(
            m=100,
            alphaguess=LineSearches.InitialStatic(alpha=0.5),
            linesearch=LineSearches.HagerZhang()
        )
        callbacks = Function[MaxGradientStopper(1e-6), DeltaYStopper(), ParameterStopper(50)]

        adapt_hist = adapt_qaoa(
            hamiltonian,
            pool,
            n,
            opt,
            callbacks;
            initial_parameter=1e-2,
            initial_state=initial_state
        )
        
        mixers = map(p->Operator(p),filter(p -> p !== nothing, adapt_hist.paulis))
        ansatz = qaoa_ansatz(hamiltonian, mixers)
        point = adapt_hist.opt_pars[end]

        _, result_grads, _, _ = sample_points(hamiltonian, ansatz, initial_state, num_samples; dist=dist, point=point, use_norm=use_norm, rng=rng)
        @test maximum(result_grads) <= 1e-3
    end
end
