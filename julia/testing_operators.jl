using Random
import LinearAlgebra: norm, abs
using Test
using BenchmarkTools

include("structs.jl")


@testset "Commuting operators" begin
    p1 = pauli_string_to_pauli("IXYZYI")
    p2 = pauli_string_to_pauli("IXXIZY")
    p3 = pauli_string_to_pauli("ZZIYXX")
    p4 = pauli_string_to_pauli("IIIIII")

    for P in [p1, p2, p3]
        @test pauli_commute(P, p4)
    end
end