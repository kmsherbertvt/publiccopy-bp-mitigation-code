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

@testset "Distinct Columns" begin
    A = [1 0 ; 0 1 ; 1 1 ; 0 0]
    @test distinct_columns(A) == true

    A = [1 1 ; 0 1 ; 1 1 ; 0 0]
    @test distinct_columns(A) == true

    A = [1 1 ; 1 1 ; 1 1 ; 0 0]
    @test distinct_columns(A) == false

    A = [0 1 ; 0 1]
    @test distinct_columns(A) == true

    A = [1 1 ; 0 1]
    @test distinct_columns(A) == true

    A = [1 1 ; 1 1]
    @test distinct_columns(A) == false
end

@testset "Test Validity of Random Instances" begin
    for k=[2, 3]
        for n=4:10
            m = 3
            for _=1:10
                A = random_k_sat_instance(n, m, k)
                @test distinct_columns(A) == true
            end
        end
    end
end

@testset "Try generation of random Hamiltonians" begin
    for k=[2, 3]
        for n=4:6
            for _=1:5
                m = 3
                h = random_k_sat_hamiltonian(n, m, k)
            end
        end
    end
end