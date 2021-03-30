using Random
import LinearAlgebra: norm, abs
using Test
using BenchmarkTools

include("fast_pauli_vec_mult.jl")
include("simulator.jl")


@testset "Test empty ansatz: New" begin
    d = 5
    for n=2:5
        for _=1:10
            axes = [[rand(0:3) for i=1:n] for j=1:d]
            theta = [0.0 for j=1:d]

            # Initial state is vac
            actual = zeros(ComplexF64, 2^n)
            actual[1] = 1.0 + 0.0im
            tmp = zeros(ComplexF64, 2^n)
            pauli_ansatz!(axes, theta, actual, tmp)

            vac = zeros(ComplexF64, 2^n)
            vac[1] = 1.0 + 0.0im

            @test norm(actual-vac)≈0.0 || "Actual: $actual"
        end
    end
end


@testset "Test empty anastz: Old" begin
    d = 5
    for n=2:5
        for _=1:10
            axes = [[rand(0:3) for i=1:n] for j=1:d]
            theta = [0.0 for j=1:d]
            expected = pauli_ansatz(axes, theta)

            vac = zeros(ComplexF64, 2^n)
            vac[1] = 1.0 + 0.0im
            @test norm(vac-expected)≈0.0
        end
    end
end

@testset "Test simulator" begin
    d = 5
    for n=2:5
        for _=1:10
            axes = [[rand(0:3) for i=1:n] for j=1:d]
            theta = [pi*rand() for j=1:d]
            expected = pauli_ansatz(axes, theta)

            actual = zeros(ComplexF64, 2^n)
            actual[1] = 1.0 + 0.0im
            tmp = zeros(ComplexF64, 2^n)
            pauli_ansatz!(axes, theta, actual, tmp)

            @test norm(actual)≈1.0 || "Actual not norm'd: $actual"
            @test norm(expected)≈1.0 || "Expected not norm'd: $expected"
            @test norm(actual-expected) <= 1e-15
        end
    end
end