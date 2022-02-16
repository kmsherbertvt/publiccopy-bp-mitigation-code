using Random
using Test
using StatsBase
using LinearAlgebra

using AdaptBarren

Random.seed!(42)

@testset "Ham Op Convert" begin
    for _=1:5
        n = 4
        mat = rand(ComplexF64, 2^n, 2^n)
        ham = matrix_to_operator(mat)

        mat_new = operator_to_matrix(ham)

        @test norm(mat - mat_new) <= 1e-5
    end
end

@testset "Test preserves hermiticity" begin
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
