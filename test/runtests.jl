using AdaptBarren
using Test

test_files = ["test_adapt.jl",
    "test_fast_grad.jl",
    "test_mat_op_conversion.jl",
    "test_op_mult.jl",
    "test_operators.jl",
    "test_pauli_evolve.jl",
    "test_vqe.jl"]

@testset "AdaptBarren.jl" begin
    for f in test_files
        include(f)
    end
end