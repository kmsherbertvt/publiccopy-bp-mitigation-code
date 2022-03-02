using Random
using Test
using StatsBase
using LinearAlgebra

using AdaptBarren

Random.seed!(42)

@testset "Handmade conversion test" begin
    actual_mat = ComplexF64[1+0im 0+0im 0+0im 0-1im;
                     0+0im -1+0im 0+1im 0+0im;
                     0+0im 0-1im -1+0im 0+0im;
                     0+1im 0+0im 0+0im 1+0im]
    actual_op = Operator([pauli_string_to_pauli("XY"), pauli_string_to_pauli("ZZ")], [1.0, 1.0])
    test_op = matrix_to_operator(actual_mat)
    test_mat = operator_to_matrix(actual_op)
    op_chop!(test_op, 1e-10)

    @test norm(actual_mat - test_mat) <= 1e-10
end

@testset "Ham Op Convert" begin
    for _=1:5
        for n=2:4
            mat = rand(ComplexF64, 2^n, 2^n)
            ham = matrix_to_operator(mat)

            mat_new = operator_to_matrix(ham)

            @test norm(mat - mat_new) <= 1e-5
        end
    end
end

@testset "Preserves hermiticity" begin
    for _=1:5
        for n=2:5
            mat = rand(ComplexF64, 2^n, 2^n)
            mat = mat + transpose(conj(mat))
            op = matrix_to_operator(mat)
            @test norm(imag.(op.coeffs)) <= 1e-10
            
            mat_new = operator_to_matrix(op)

            @test norm(mat_new - transpose(conj(mat_new))) <= 1e-10
        end
    end
end
