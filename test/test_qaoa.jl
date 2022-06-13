using LinearAlgebra
using Test
using AdaptBarren
using NLopt
using Random

rng = MersenneTwister(14)

@testset "QAOA Test" begin
    n = 4
    d = 3
    p = 50

    for _ in range(1,10)
        hamiltonian = random_regular_max_cut_hamiltonian(n, d)
        ground_state_energy = minimum(real(diag(operator_to_matrix(hamiltonian))))

        mixers = repeat([qaoa_mixer(n)], p)
        initial_point = rand(rng, Float64, 2*p)
        #opt = Opt(:LN_COBYLA, length(initial_point))
        opt = Opt(:LD_LBFGS, length(initial_point))
        initial_state = ones(ComplexF64, 2^n) / sqrt(2^n)
        result = QAOA(hamiltonian, mixers, opt, initial_point, n, initial_state)
        qaoa_energy = result[1]

        en_err = abs(ground_state_energy - qaoa_energy)
        @test en_err <= 1e-3
    end
end

@testset "ADAPT-QAOA Test" begin
   n = 6
   d = 3
   opt = "LD_LBFGS"
   path="test_data"

   for _ in range(1,10)
       hamiltonian = random_regular_max_cut_hamiltonian(n, d)
       ground_state_energy = minimum(real(diag(operator_to_matrix(hamiltonian))))

       pool = two_local_pool(n)
       pool = map(p -> Operator([p], [1.0]), pool)

       initial_state = ones(ComplexF64, 2^n) / sqrt(2^n)
       initial_state /= norm(initial_state)
       callbacks = Function[ParameterStopper(100), MaxGradientStopper(1e-8)]

       result = adapt_qaoa(hamiltonian, pool, n, opt, callbacks; initial_parameter=1e-2, initial_state=initial_state, path=path)

       adapt_qaoa_energy = last(result.energy)
       en_err = abs(ground_state_energy - adapt_qaoa_energy)
       @test en_err <= 1e-3
   end
end


@testset "Reproduce Linghua's Result" begin
    n = 6
    d = 3
    p = 30

    for _ in range(1,10)
        hamiltonian = random_regular_max_cut_hamiltonian(n, d)
        ground_state_energy = minimum(real(diag(operator_to_matrix(hamiltonian))))
        opt = "LD_LBFGS"
        #opt = "LN_NELDERMEAD"

        pool = two_local_pool(n)
        pool = map(p -> Operator([p], [1.0]), pool)
        #push!(pool, [qaoa_mixer(n)])
        push!(pool, qaoa_mixer(n))

        initial_state = ones(ComplexF64, 2^n) / sqrt(2^n)
        initial_state /= norm(initial_state)
        callbacks = Function[ParameterStopper(50), MaxGradientStopper(1e-8)]

        path=nothing

        result = adapt_qaoa(hamiltonian, pool, n, opt, callbacks; initial_parameter=1e-2, initial_state=initial_state, path=path)

        adapt_qaoa_energy = last(result.energy)
        en_err = abs(ground_state_energy - adapt_qaoa_energy)
        @test en_err <= 1e-3
    end
end
