using LinearAlgebra
using StatsBase
using Test
using Random
using AdaptBarren

@testset "Test Cost Function and Gradient Sampling" begin
    n = 4
    k = 10
    num_samples = 50
    pool = map(p -> Operator([p], [1.0]), two_local_pool(n))
    push!(pool, qaoa_mixer(n))
    ansatz = sample(pool, k; replace = true)
    hamiltonian = random_regular_max_cut_hamiltonian(n, n-1; weighted = true)
    initial_state = uniform_state(n)

    result_energies, result_gradients = sample_points(hamiltonian, ansatz, initial_state, num_samples)

    @test length(result_energies) == num_samples
    @test length(result_gradients) == num_samples * k
end