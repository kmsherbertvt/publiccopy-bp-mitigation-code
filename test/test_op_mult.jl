using Random
using Test
using StatsBase
using LinearAlgebra

using AdaptBarren

@testset "Handmade Ham Op Mult" begin
    n = 2
    ham = Operator(
        [pauli_string_to_pauli("XX"), pauli_string_to_pauli("ZI")], 
        [1.0+0.0im, 1.0+0.0im]
    )

    init_state = zeros(ComplexF64, 2^n)
    init_state[4] = 1.0 + 0.0im

    expected_result = zeros(ComplexF64, 2^n)
    expected_result[1] = 1.0 + 0.0im
    expected_result[4] = -1.0 + 0.0im

    actual_result = copy(init_state)
    tmp1 = similar(actual_result)
    tmp2 = similar(actual_result)
    ham_state_mult!(ham, actual_result, tmp1, tmp2)

    @test norm(actual_result - expected_result) <= 1e-4
end

@testset "Handmade Example 2" begin
    H = Operator(
        [pauli_string_to_pauli("X"), pauli_string_to_pauli("Y"), pauli_string_to_pauli("Z")],
        [1.0+0.0im, 2.0+0.0im, 3.0+0.0im]
    )
    H_mat = [3.0+0.0im +1.0-2.0im 
             1.0+2.0im -3.0+0.0im]
    @test norm(operator_to_matrix(H) - H_mat) <= 1e-10

    psi = [1.0+0.0im
           0.0+2.0im]
    psi_final = [7.0+2.0im
                 1.0-4.0im]
    @test norm(H_mat * psi - psi_final) <= 1e-10

    psi_test = copy(psi)
    tmp1 = similar(psi_test)
    tmp2 = similar(psi_test)
    ham_state_mult!(H, psi_test, tmp1, tmp2)
    @test norm(psi_test - psi_final) <= 1e-10
end

@testset "Handmade example on 2 qubits" begin
    H = Operator(map(pauli_string_to_pauli, ["II", "IZ", "ZI", "ZZ"]), map(ComplexF64, [1.0, 2.0, 3.0, 4.0]))
    H_out = matrix_to_operator(operator_to_matrix(H))

    for (c,p) in zip(H_out.coeffs, H_out.paulis)
        println("C_alpha = $c, P_alpha = $p")
    end

    psi = [
        1.0 + 0.0im
        0.0 + 0.0im
        1.0 + 0.0im
        0.0 + 0.0im
    ]
    psi_final = [
        10.0 + 0.0im
        0.0 + 0.0im
        -4.0 + 0.0im
        0.0 + 0.0im
    ]

    tmp1 = similar(psi)
    tmp2 = similar(psi)

    ham_state_mult!(H, psi, tmp1, tmp2)

    @test norm(psi - psi_final) <= 1e-10
end

@testset "Random Ham Op Mult" begin
    for _ = 1:5
        for n=1:3
            mat = rand(ComplexF64, 2^n, 2^n)
            mat = mat + conj(transpose(mat))
            ham = matrix_to_operator(mat)

            psi_init = rand(ComplexF64, 2^n)
            psi_final = mat * psi_init

            tmp1 = similar(psi_final)
            tmp2 = similar(psi_final)
            ham_state_mult!(ham, psi_init, tmp1, tmp2)

            @test norm(psi_final - psi_init) <= 1e-10
        end
    end
end
