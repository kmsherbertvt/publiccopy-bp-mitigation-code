using Test
using LinearAlgebra
using AdaptBarren

@testset "Test MAX-2-SAT Hamiltonian" begin
    A = [1 0 ; 0 1 ; 1 1 ; 0 0]
    P = SATProblem(A)
    @test P.n == 4
    @test P.m == 2
    expected_vec = reverse([1,0,0,-1,-1,0,0,1,1,0,0,-1,-1,0,0,1]) .+ 1
    actual_vec = Int.(real(diag(operator_to_matrix(max_1_2_sat_ham(P)))))
    @test norm(actual_vec .- expected_vec) <= 1e-4
end

@testset "Test MAX-3-SAT Hamiltonian" begin
    A = [1 1 ; 1 1 ; 1 0 ; 0 1]
    P = SATProblem(A)
    @test P.n == 4
    @test P.m == 2
    expected_vec = reverse([6, 0, 0, -2, 3, -1, -1, -1, 3, -1, -1, -1, 0, -2, -2, 0]) .+ 2
    actual_vec = Int.(real(diag(operator_to_matrix(max_1_3_sat_ham(P)))))
    @test norm( actual_vec .- expected_vec) <= 1e-4
end