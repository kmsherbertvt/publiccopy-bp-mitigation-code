using Random
import LinearAlgebra: norm, abs
using Test
using BenchmarkTools

include("structs.jl")


@testset "Basic id tests" begin
    P = pauli_string_to_pauli("IIIX")
    ID = pauli_string_to_pauli("IIII")
    @test pauli_commute(P, ID) || "Didn't commute with id: $P"

    P = pauli_string_to_pauli("IIXX")
    ID = pauli_string_to_pauli("IIII")
    @test pauli_commute(P, ID) || "Didn't commute with id: $P"

    P = pauli_string_to_pauli("XXXX")
    ID = pauli_string_to_pauli("III")
    @test pauli_commute(P, ID) || "Didn't commute with id: $P"
end

@testset "Involved id tests" begin

    p1 = pauli_string_to_pauli("IXYZYI")
    p2 = pauli_string_to_pauli("IXXIZY")
    p3 = pauli_string_to_pauli("ZZIYXX")
    p4 = pauli_string_to_pauli("IIIIII")

    for P in [p1, p2, p3]
        @test pauli_commute(P, p4) || "Didn't commute with id: $P"
    end
end


@testset "Handworked tests" begin
    # Handworked tests
    @test false == pauli_commute(
        pauli_string_to_pauli("IXIZYIZYI"),
        pauli_string_to_pauli("XIYXIYZIY")
    )

    @test true == pauli_commute(
        pauli_string_to_pauli("IXIZYIXYI"),
        pauli_string_to_pauli("XIYXIYZIY")
    )

    @test false == pauli_commute(
        pauli_string_to_pauli("XIZYIZ"),
        pauli_string_to_pauli("XXXIYZ")
    )

    @test true == pauli_commute(
        pauli_string_to_pauli("IXIZYIY"),
        pauli_string_to_pauli("IXXXIYZ")
    )
end

@testset "Randomized tests" begin
    # Nontrivial tests
    for n=1:5
        for _=1:5
            Ps = String(rand(['I', 'X', 'Y', 'Z'], n))
            for _=1:5
                Qs = String(rand(['I', 'X', 'Y', 'Z'], n))

                disagrees = 0
                for (p,q)=zip(Ps,Qs)
                    if p == 'I' || q == 'I'
                        continue
                    end
                    if p != q
                        disagrees += 1
                    end
                end
                actual = Bool((disagrees+1) % 2) # commutes?
                P = pauli_string_to_pauli(Ps)
                Q = pauli_string_to_pauli(Qs)

                @test pauli_commute(P, Q) == actual
            end
        end
    end
end