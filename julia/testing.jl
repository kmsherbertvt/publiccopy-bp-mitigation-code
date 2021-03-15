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
            pauli_ansatz_new!(axes, theta, actual, tmp1)

            @test norm(actual-expected)≈0.0
        end
    end
end

@testset "Test multiplication" begin
    for n=2:5
        for _=1:40
            init = rand(ComplexF64, 2^n)
            init /= norm(init)
            #@test norm(init)≈1.0
            
            axes=[rand(0:3) for i=1:n]
            expected = pauli_str(axes) * init
            pauli_vec_mult!(init, axes)

            @test norm(init-expected)≈0.0 || "failed on $axes"
            #@test norm(abs.(init)-abs.(expected))≈0.0
            #@test norm(init)≈1.0
        end
    end
end

@testset "Hamming weight" begin
    @test hamming_weight(83) == 4
    @test hamming_weight(5458) == 6
end

@testset "Pauli Masks" begin
    pauli = [1, 0, 3, 2, 0, 1, 0]
    pm = pauli_masks(pauli)

    #@test pm[1] == [0, 1, 0, 0, 1, 0, 1]
    @test pm[1] == Int(0b1000010)
    @test pm[2] == Int(0b0001000)
    @test pm[3] == Int(0b0010000)

    @test pauli_masks([0, 3, 3, 0, 3])[3] == Int(0b01101)
    @test pauli_masks([0, 3, 3, 0, 3])[1] == Int(0)
    @test pauli_masks([0, 3, 3, 0, 3])[2] == Int(0)
end

@testset "Pauli Apply" begin
    test_cases = [
        # pauli, input state, expected output state
        [[0, 3, 3, 0], 3, 3],
        [[0, 3, 3, 0], 5, 5],
        [[0, 3, 3, 0], 8, 8],
    ]
    for tc=test_cases
        pauli, state, expected = tc
        pm = pauli_masks(pauli)
        @test pauli_apply(pm, state) == expected
    end
end

@testset "Phase Shift" begin
    for _=1:10
        for k=0:3
            a = rand(ComplexF64)
            b = phase_shift(a, k)
        
            ap = a*exp(2*pi*1im*k/4)

            @test b≈ap
        end
    end
end

@testset "Pauli Phase" begin
    test_sets = [
        # pauli, state, phase
        [[1, 0, 3, 2, 0, 1, 0], Int(0b0110100), 3],
        [[1, 1, 2, 1, 0, 3, 0, 2, 1, 2], Int(0b1111111111), 3], # (-i)(-1)(-i)(-i) = (i^3)
        [[1, 1, 0, 1, 0, 3, 0, 2, 1, 2], Int(0b1111111111), 0], # (+1)(-1)(-i)(-i) = (-1)(i)(i) = (-1)(-1)
        [[0, 3, 0, 3, 3, 0], Int(0b111111), 2],
        [[0, 3, 0, 3, 3, 0], Int(0b101111), 0],
        [[0, 3, 3, 0, 3], Int(0b11111), 2],
    ]
    for trip=test_sets
        pauli, state, phase = trip
        pm = pauli_masks(pauli)
        @test pauli_phase(pm, state) == phase
    end
end

@testset "Invert phase" begin
    @test invert_phase(0) == 0
    @test invert_phase(1) == 3
    @test invert_phase(3) == 1
    @test invert_phase(2) == 2
end