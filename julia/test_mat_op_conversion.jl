include("operator.jl")
include("pauli.jl")

using Random
using Test
using StatsBase
using LinearAlgebra

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