using LinearAlgebra
using StatsBase
using Test
using Random
using AdaptBarren
using Distributions

rng = MersenneTwister(42);

@testset "Test Cost Function and Gradient Sampling" begin
    n = 4
    k = 10
    num_samples = 50
    pool = map(p -> Operator([p], [1.0]), two_local_pool(n))
    push!(pool, qaoa_mixer(n))
    ansatz = sample(pool, k; replace = true)
    hamiltonian = random_regular_max_cut_hamiltonian(n, n-1; weighted = true)
    initial_state = uniform_state(n)

    point = rand(rng, Uniform(-pi, +pi), length(ansatz))

    for _=1:2
        result_energies, result_gradients = sample_points(hamiltonian, ansatz, initial_state, num_samples; rng=rng, dist=0.0, point=point)

        @test length(result_energies) == num_samples
        @test length(result_gradients) == num_samples * k

        delta = maximum(result_energies) - minimum(result_energies)
        @test abs(delta) <= 1e-6
    end
end