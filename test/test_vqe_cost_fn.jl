using Random
using Test
using LinearAlgebra
using AdaptBarren

@testset "Handmade  QAOA Cost Fn Example" begin
    n = 4
    hamiltonian = Operator(
        [
            pauli_string_to_pauli("IIII"),
            pauli_string_to_pauli("IZII"),
            pauli_string_to_pauli("ZIII"),
            pauli_string_to_pauli("ZZII"),

            pauli_string_to_pauli("ZIZI"),
            pauli_string_to_pauli("ZZZZ"),
            pauli_string_to_pauli("IIZZ"),
        ],
        [0.1, 0.4, 0.4, 0.3, 0.4, 0.5, 0.6]
    )
    mixers = [
        Operator([pauli_string_to_pauli("XZXY")], [1.0]),
        qaoa_mixer(4)
    ]
    beta = [1.1, 3.3]
    gamma = [1.5, 1.6, 1.7]

    ansatz = qaoa_ansatz(hamiltonian, mixers)

    initial_state = ones(ComplexF64, 2^n)
    initial_state /= norm(initial_state)

    ### Test
    expected_cost = 0.241157 # From Mathematica Notebook
    expected_state = [0.355111-0.0851575im, -0.16343-0.22583im, 0.0360796-0.0627358im, 0.0847592+0.112516im, -0.277738-0.172374im, -0.0193867-0.260217im, -0.0818478-0.224706im, -0.206698-0.138988im, -0.140029+0.269187im, -0.225121+0.0648942im, 0.072486+0.113428im, 0.157399+0.0641457im, 0.261241-0.210772im, -0.255521-0.0995391im, -0.137949-0.123217im, 0.171185-0.15596im]
    expected_grad = [-0.00400984, 0.70968, 0.259633, 0.316023, 0]
    x = [gamma[1], beta[1], gamma[2], beta[2], gamma[3]]
    grad = similar(x)
    
    # other vars
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
    @test isapprox(grad, expected_grad)
end







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

    @show ans
    @show output_state
    @show expected_grad
    @show grad

    @test abs(ans - expected_cost) <= 1e-3
    @test norm(expected_state - output_state) <= 1e-3
    @test norm(grad - expected_grad) <= 1e-3
end