using AdaptBarren
using Glob
using Test
using Random

Random.seed!(123)

@testset "AdaptBarren.jl" begin
    for f in glob("test_*.jl")
        include(f)
    end
end
