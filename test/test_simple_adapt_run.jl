using LinearAlgebra
using Random
using Test
using AdaptBarren

@testset "ADAPT Simple Run" begin
    n = 4
    operator = random_regular_max_cut_hamiltonian(n, n-1)
    op_simplify!(operator)
    hamiltonian = operator_to_matrix(operator)
    ground_state_energy = minimum(real(diag(hamiltonian)))

    pool = two_local_pool(n)
    optimizer = "LD_LBFGS"
    callbacks = Function[ParameterStopper(10), MaxGradientStopper(1e-10)]
            
    state = rand(ComplexF64, 2^n)
    state /= norm(state)

    result = adapt_vqe(operator, pool, n, optimizer, callbacks; initial_parameter=0.0, initial_state=state)

    en_adapt = last(result.energy)

    en_err = abs(ground_state_energy - en_adapt)

    println("Energy")
    println(result.energy)
    println("\n\n\n")

    println("Result")
    println(result)
    println("\n\n\n")

    #for (k,d)=enumerate(result)
    #    println(d)
    #end
end
