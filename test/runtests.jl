using AdaptBarren
using Glob
using Test

@testset "AdaptBarren.jl" begin
    for f in glob("test_*.jl")
        include(f)
    end
end
