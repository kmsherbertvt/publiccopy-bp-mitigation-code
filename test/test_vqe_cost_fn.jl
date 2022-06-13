using Random
using Test
using LinearAlgebra
using AdaptBarren


@testset "Handmade  QAOA Cost Fn Example" begin
    """ Corresponds to the output shown in `test_vqe_cost_fn_sister_script_4.wls`
    """
    n = 2
    hamiltonian = qaoa_mixer(2)
    hz = Operator(
        [
            pauli_string_to_pauli("IZ"),
            pauli_string_to_pauli("ZI"),
        ],
        [0.5, 0.4]
    )
    ansatz = [hz]
    x = [1.1]

    ### Test
    expected_cost = 0.890601 # From Mathematica Notebook
    expected_state = [0.100177-0.152637im,0.362941+0.0400854im,0.544412-0.0601281im,0.400706+0.610547im] 
    expected_grad = [-1.28396]
    initial_state = [1+0im,2+0im,3+0im,4+0im]
    
    # other vars
    initial_state /= norm(initial_state)
    grad = similar(x)
    fn_evals = Vector{Float64}()
    grad_evals = Vector{Float64}()
    eval_count = 0

    state = similar(initial_state)
    tmp = zeros(ComplexF64, 2^n)
    tmp1 = similar(tmp)
    tmp2 = similar(tmp)
    output_state = similar(state)

    ans = _cost_fn_commuting_vqe(
        x, #
        grad, #
        ansatz,
        hamiltonian,
        fn_evals,
        grad_evals,
        eval_count, 
        state,
        initial_state,
        tmp,
        tmp1,
        tmp2,
        output_state
    )

    @test abs(ans - expected_cost) <= 1e-3
    @test norm(expected_state - output_state) <= 1e-3
    @test norm(grad - expected_grad) <= 1e-3
end