include("operator.jl")
include("pauli.jl")

using Random
using Test
using StatsBase
using LinearAlgebra

Random.seed!(42)

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

@testset "Random Ham Op Mult" begin
    n = 2
    mat = rand(ComplexF64, 2^n, 2^n)
    ham = matrix_to_operator(mat)

    init_state = rand(ComplexF64, 2^n)
    init_state /= norm(init_state)

    expected_result = mat * init_state

    actual_result = copy(init_state)
    tmp1 = similar(actual_result)
    tmp2 = similar(actual_result)
    ham_state_mult!(ham, actual_result, tmp1, tmp2)

    @test norm(actual_result - expected_result) <= 1e-4
end