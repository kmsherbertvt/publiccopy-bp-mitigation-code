using Random
import LinearAlgebra: norm
using Test
using BenchmarkTools

include("fast_pauli_vec_mult.jl")
include("simulator.jl")

@testset "Test simulator" begin
    d = 5
    for n=2:5
        for _=1:10
            axes = [[rand(0:3) for i=1:n] for j=1:d]
            theta = [pi*rand() for j=1:d]
            expected = pauli_ansatz(axes, theta)

            actual = zeros(ComplexF64, 2^n)
            tmp1 = zeros(ComplexF64, 2^n)
            tmp2 = zeros(ComplexF64, 2^n)
            pauli_ansatz_new!(axes, theta, actual, tmp1, tmp2)

            @test norm(actual-expected)≈0.0
        end
    end
end

@testset "Test multiplication" begin
    for n=2:5
        init = rand(ComplexF64, 2^n)
        init /= norm(init)
        for _=1:10
            axes=[rand(0:3) for i=1:n]
            expected = pauli_str(axes) * init

            result = zeros(ComplexF64, 2^n)
            tmp = zeros(ComplexF64, 2^n)
            pauli_vec_mult!(result, axes, init, tmp)

            @test norm(result-expected)≈0.0
        end
    end
end