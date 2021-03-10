import IterTools: enumerate


function get_kth_bit(n::Int64, k::Int64)
    return ((n & (1 << (k - 1))) >> (k - 1))
end


function pauli_phase_appl(ax::Int64, k::Int64, b::Int64)
    if ax == 1
        return 1.0 + 0.0im
    elseif ax == 2
        b_k = get_kth_bit(b, k)
        return 0.0 + (-1.0)^b_k * 1.0im
    elseif ax == 3
        b_k = get_kth_bit(b, k)
        return (-1.0)^b_k * 1.0 + 0.0im
    elseif ax == 0
        return 1.0 + 0.0im
    end
end


function pauli_modify_bitstring(ax::Int64, k::Int64, b::Int64)
    # Save comparison of ax == 0 since loop excludes this possibility
    if ax == 3
        return b
    end
        return xor(b, 1<<(k-1))
end


function pauli_vec_mult!(psi_new::Array{ComplexF64,1}, axes, psi::Array{ComplexF64,1}, tmp::Array{ComplexF64,1})
    tmp .= psi
    n = length(axes)
    N = 2^n
    weight = 0
    for (kp,p)=enumerate(reverse(axes))
        k = kp
        if p == 0
            continue
        end
        weight += 1
        for j=0:N-1
            new_bit = pauli_modify_bitstring(p, k, j)
            new_phase = pauli_phase_appl(p, k, j)
            psi_new[new_bit+1] = tmp[j+1]*new_phase
        end
        tmp .= psi_new
    end
    if weight == 0
        psi_new .= psi
    end
end


#using Random
#import LinearAlgebra: norm
#using Test
#include("simulator.jl")
#
#@testset "Test multiplication" begin
#    for n=2:5
#        vac = zeros(ComplexF64, 2^n)
#        vac[1] = 1.0 + 0.0im
#        for _=1:10
#            axes=[rand(0:3) for i=1:n]
#            expected = pauli_str(axes) * vac
#
#            result = zeros(ComplexF64, 2^n)
#            tmp = zeros(ComplexF64, 2^n)
#            pauli_vec_mult!(result, axes, vac, tmp)
#
#            @test norm(result-expected)≈0.0
#        end
#    end
#end