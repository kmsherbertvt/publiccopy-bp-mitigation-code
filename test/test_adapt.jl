using LinearAlgebra
using Test
using AdaptBarren

@testset "ADAPT Random Diagonal" begin
    for _=1:10
        for n in [4, 6]
            operator = random_regular_max_cut_hamiltonian(n, 3)
            op_simplify!(operator)
            hamiltonian = operator_to_matrix(operator)
            ground_state_energy = minimum(real(diag(hamiltonian)))

            pool = two_local_pool(n)
            optimizer = "LD_LBFGS"
            callbacks = Function[ParameterStopper(200), MaxGradientStopper(1e-10)]
            
            state = ones(ComplexF64, 2^n)
            state /= norm(state)

            result = adapt_vqe(operator, pool, n, optimizer, callbacks; initial_parameter=1e-2, initial_state=state)

            en_adapt = last(result.energy)

            en_err = abs(ground_state_energy - en_adapt)
            @test en_err <= 1e-8
        end
    end
end
