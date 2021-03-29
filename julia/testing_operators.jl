using Random
import LinearAlgebra: norm, abs
using Test
using BenchmarkTools

include("structs.jl")


#@testset "Basic id tests" begin
#    P = pauli_string_to_pauli("IIIX")
#    ID = pauli_string_to_pauli("IIII")
#    @test pauli_commute(P, ID) || "Didn't commute with id: $P"
#
#    P = pauli_string_to_pauli("IIXX")
#    ID = pauli_string_to_pauli("IIII")
#    @test pauli_commute(P, ID) || "Didn't commute with id: $P"
#
#    P = pauli_string_to_pauli("XXXX")
#    ID = pauli_string_to_pauli("III")
#    @test pauli_commute(P, ID) || "Didn't commute with id: $P"
#end
#
#@testset "Involved id tests" begin
#
#    p1 = pauli_string_to_pauli("IXYZYI")
#    p2 = pauli_string_to_pauli("IXXIZY")
#    p3 = pauli_string_to_pauli("ZZIYXX")
#    p4 = pauli_string_to_pauli("IIIIII")
#
#    for P in [p1, p2, p3]
#        @test pauli_commute(P, p4) || "Didn't commute with id: $P"
#    end
#end
#
#
#@testset "Handworked tests" begin
#    # Handworked tests
#    @test false == pauli_commute(
#        pauli_string_to_pauli("IXIZYIZYI"),
#        pauli_string_to_pauli("XIYXIYZIY")
#    )
#
#    @test true == pauli_commute(
#        pauli_string_to_pauli("IXIZYIXYI"),
#        pauli_string_to_pauli("XIYXIYZIY")
#    )
#
#    @test false == pauli_commute(
#        pauli_string_to_pauli("XIZYIZ"),
#        pauli_string_to_pauli("XXXIYZ")
#    )
#
#    @test true == pauli_commute(
#        pauli_string_to_pauli("IXIZYIY"),
#        pauli_string_to_pauli("IXXXIYZ")
#    )
#end
#
#@testset "Randomized tests" begin
#    # Nontrivial tests
#    for n=1:5
#        for _=1:5
#            Ps = String(rand(['I', 'X', 'Y', 'Z'], n))
#            for _=1:5
#                Qs = String(rand(['I', 'X', 'Y', 'Z'], n))
#
#                disagrees = 0
#                for (p,q)=zip(Ps,Qs)
#                    if p == 'I' || q == 'I'
#                        continue
#                    end
#                    if p != q
#                        disagrees += 1
#                    end
#                end
#                actual = Bool((disagrees+1) % 2) # commutes?
#                P = pauli_string_to_pauli(Ps)
#                Q = pauli_string_to_pauli(Qs)
#
#                @test pauli_commute(P, Q) == actual
#            end
#        end
#    end
#end


@testset "Randomized identity" begin
    for n=2:5
        for _=1:10
            id = pauli_string_to_pauli(String(repeat(['I'], n)))
            p = pauli_string_to_pauli(String(rand(['I', 'X', 'Y', 'Z'], n)))
            @test pauli_product(id, p) == p
            @test pauli_product(p, id) == p
        end
    end
end

#=
-------------------------------------------------------
    THIS FUNCTION IS SLOW AND JUST FOR SELF-CONSISTENCY
    IT SHOULD NOT BE USED FOR ACTUAL COMPUTATIONS
-------------------------------------------------------
=#

function easy_pp(p::Int64, q::Int64)
    #=
    Returns pauli (axis) and phase
    represented by int in [0, 1, 2, 3]
    =#
    if p == q
        return (0, 0)
    end
    if p == 0
        return (q, 0)
    elseif q == 0
        return (p, 0)
    end

    if p == 1
        if q == 2
            return (3, 1)
        else # q == 3
            return (2, 3)
        end
    elseif p == 2
        if q == 1
            return (3, 3)
        else # q == 3
            return (1, 1)
        end
    else # p == 3
        if q == 1
            return (2, 1)
        else # q == 2
            return (1, 3)
        end
    end
end

#=
-------------------------------------------------------
    THIS FUNCTION IS SLOW AND JUST FOR SELF-CONSISTENCY
    IT SHOULD NOT BE USED FOR ACTUAL COMPUTATIONS
-------------------------------------------------------
=#

function easy_pauli_product(p::Array{Int64,1}, q::Array{Int64,1})
    n = length(p)
    phase = 0
    res = zeros(Int64, n)
    for i=1:n
        new_pauli, new_phase = easy_pp(p[i], q[i])
        res[i] = new_pauli
        phase += new_phase
    end
    phase = phase % 4
    return (res, phase)
end

#=
-------------------------------------------------------
    THIS FUNCTION IS SLOW AND JUST FOR SELF-CONSISTENCY
    IT SHOULD NOT BE USED FOR ACTUAL COMPUTATIONS
-------------------------------------------------------
=#

function pauli_str_to_axes(s::String)
    res = Array{Int64,1}()
    for c in s
        if c == 'I'
            append!(res, 0)
        elseif c == 'X'
            append!(res, 1)
        elseif c == 'Y'
            append!(res, 2)
        elseif c == 'Z'
            append!(res, 3)
        end
    end
    return res
end

#=
-------------------------------------------------------
    THIS FUNCTION IS SLOW AND JUST FOR SELF-CONSISTENCY
    IT SHOULD NOT BE USED FOR ACTUAL COMPUTATIONS
-------------------------------------------------------
=#

function pauli_axes_to_str(ax::Array{Int64, 1})
    res = Array{Char,1}()
    for c in ax
        if c == 0
            append!(res, 'I')
        elseif c == 1
            append!(res, 'X')
        elseif c == 2
            append!(res, 'Y')
        elseif c == 3
            append!(res, 'Z')
        end
    end
    return String(res)
end


@testset "Nilpotent paulis" begin
    for n=2:5
        for _=1:4
            ps = String(rand(['I', 'X', 'Y', 'Z'], n))
            _P = pauli_str_to_axes(ps)
            P = pauli_string_to_pauli(ps)
            R = pauli_product(P, P)
            
            @test [R.x, R.y, R.z, R.phase] == [0, 0, 0, 0]
        end
    end
end


@testset "Completely randomized" begin
    for n=2:5
        for _=1:4
            ps = String(rand(['I', 'X', 'Y', 'Z'], n))
            _P = pauli_str_to_axes(ps)
            P = pauli_string_to_pauli(ps)
            for _=1:4
                qs = String(rand(['I', 'X', 'Y', 'Z'], n))
                _Q = pauli_str_to_axes(qs)
                Q = pauli_string_to_pauli(qs)

                R = pauli_product(P, Q)

                _R, _phase = easy_pauli_product(_P, _Q)
                R_prime = pauli_string_to_pauli(pauli_axes_to_str(_R))

                @test [R_prime.x, R_prime.y, R_prime.z, _phase] == [R.x, R.y, R.z, R.phase]
            end
        end
    end
end