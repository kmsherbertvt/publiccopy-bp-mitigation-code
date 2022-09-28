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

    points_to_sample = Vector{Vector{Float64}}()
    append!(points_to_sample, [rand(rng, Uniform(-pi, +pi), length(ansatz))])

    l_min = []
    l_max = []

    for _=1:2
        result_energies, result_gradients, _, _ = sample_points(hamiltonian, ansatz, initial_state, num_samples; rng=rng, dist=0.0, point=points_to_sample)

        delta = maximum(result_energies) - minimum(result_energies)
        @test abs(delta) <= 1e-6

        push!(l_min, minimum(result_energies))
        push!(l_max, maximum(result_energies))
    end

    @test abs(l_min[1] - l_min[2]) <= 1e-6
    @test abs(l_max[1] - l_max[2]) <= 1e-6
end