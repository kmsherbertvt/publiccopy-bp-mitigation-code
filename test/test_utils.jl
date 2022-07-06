using LinearAlgebra
using Test
using AdaptBarren

@testset "Test Decreasing Energy util" begin
    @test true == energy_improving(reverse(1:10))
    @test false == energy_improving([5,4,3,4,2,1,0])
    @test true == energy_improving([5,4,3,2,2,1,0])
end