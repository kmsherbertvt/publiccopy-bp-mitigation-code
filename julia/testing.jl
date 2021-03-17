using Random
import LinearAlgebra: norm
using Test
using BenchmarkTools

include("fast_pauli_vec_mult.jl")
include("simulator.jl")


@testset "Empty ansatz old" begin
    d = 5
    for n=2:5
        vac = zeros(ComplexF64, 2^n)
        vac[1] = 1.0 + 0.0im
        for _=1:10
            axes = [[rand(0:3) for i=1:n] for j=1:d]
            theta = [0.0 for j=1:d]

            # Initial state is vac
            expected = pauli_ansatz(axes, theta)

            @test norm(expected-vac)≈0.0
        end
    end

end

@testset "Empty ansatz new" begin
    d = 5
    for n=2:5
        vac = zeros(ComplexF64, 2^n)
        vac[1] = 1.0 + 0.0im
        for _=1:10
            axes = [[rand(0:3) for i=1:n] for j=1:d]
            theta = [0.0 for j=1:d]

            # Initial state is vac
            actual = zeros(ComplexF64, 2^n)
            actual[1] = 1.0 + 0.0im
            tmp1 = zeros(ComplexF64, 2^n)
            tmp2 = zeros(ComplexF64, 2^n)
            pauli_ansatz_new!(axes, theta, actual, tmp1, tmp2)

            @test norm(actual-vac)≈0.0
        end
    end

end



@testset "Test simulator consistency" begin
    d = 5
    for n=2:5
        for _=1:10
            axes = [[rand(0:3) for i=1:n] for j=1:d]
            theta = [pi*rand() for j=1:d]

            # Initial state is vac
            expected = pauli_ansatz(axes, theta)

            # Initial state is vac
            actual = zeros(ComplexF64, 2^n)
            actual[1] = 1.0 + 0.0im
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