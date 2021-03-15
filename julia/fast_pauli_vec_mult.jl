import IterTools: enumerate
import DataStructures: SortedSet


function get_kth_bit(n::Int64, k::Int64)
    return ((n & (1 << (k - 1))) >> (k - 1))
end


function phase_shift(alpha::ComplexF64, i::Int64)
    if i == 0
        return alpha
    elseif i == 1
        return -alpha.im + im*alpha.re
    elseif i == 2
        return -alpha
    elseif i == 3
        return alpha.im - im*alpha.re
    end
end


function pauli_masks(res::Array{Int64,1}, pauli_str::Array{Int64,1})
    for (i,ax)=enumerate(reverse(pauli_str))
        res[ax+1] += 2^(i-1)
    end
    return res
end


function hamming_weight(a::Int64)
    res = 0
    for k=0:64-1 # could probably stop earlier...?
        res += get_kth_bit(a, k)
    end
    return res
end


function pauli_phase(pm::Array{Int64, 1}, a::Int64)
    # Compute the phase gamma where P|a> = gamma |b> for
    # a pauli string P and basis state |a>.
    # Convention:
    #   0 -> 1
    #   1 -> +i
    #   2 -> -1
    #   3 -> -i
    # pm = pauli_mask input
    x = hamming_weight((pm[3] | pm[4]) & a) % 2
    y = hamming_weight(pm[3]) % 4

    alpha = y
    beta = 2*x

    return (alpha+beta) % 4
end


function pauli_apply(pm::Array{Int64, 1}, a::Int64)
    # pm = pauli_mask input
    return xor((pm[2]|pm[3]),a)
end


function invert_phase(i::Int64)
    # equivalent to compl conj
    return mod(i-2*(i%2),4)
end


function pauli_vec_mult!(
        psi::Array{ComplexF64,1}, 
        axes, 
        pm::Union{Array{Int64,1},Nothing}=nothing,
        hit_bits::Union{SortedSet{Int64},Nothing}=nothing
        )
    n = length(axes)
    N = 2^n

    if hit_bits === nothing
        hit_bits = SortedSet{Int64}()
    end
    if pm === nothing
        pm = [0, 0, 0, 0]
    end
    for i=0:N-1
        if in(i,hit_bits)
            continue
        end
        pm[1] = 0; pm[2] = 0; pm[3] = 0; pm[4] = 0
        pauli_masks(pm, axes)
        j = pauli_apply(pm, i)
        gamma = pauli_phase(pm, i)
        gamma_inv = invert_phase(gamma) 
        
        jp1 = j+1
        ip1 = i+1
        # Apply phases
        psi[ip1] = phase_shift(psi[ip1], gamma)
        # Swap coeffs
        if i != j
            psi[jp1] = phase_shift(psi[jp1], gamma_inv)
            psi[ip1], psi[jp1] = psi[jp1], psi[ip1]
        end

        push!(hit_bits, i)
        push!(hit_bits, j)
    end
    empty!(hit_bits)
end