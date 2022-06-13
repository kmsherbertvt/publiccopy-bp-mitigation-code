using LinearAlgebra
using AdaptBarren
using Test

@testset "Test Clumping" begin
    x = [
        0.0, 1e-15, 2e-15,
        1.0, 1.0 + 1e-15,
        1.5,
        1.6,
        1.7, 1.7 + 1e-15
        ]
    expected_result = [
        [0.0, 1e-15, 2e-15],
        [1.0, 1.0 + 1e-15],
        [1.5],
        [1.6],
        [1.7, 1.7 + 1e-15]
        ]
    
    actual_result = clump_degenerate_values(x)

    for (a,b)=zip(actual_result, expected_result)
        @test norm(a .- b) < 1e-15
    end
end