import IterTools: enumerate


function get_kth_bit(n::Int64, k::Int64)
    return ((n & (1 << (k - 1))) >> (k - 1))
end


function pauli_phase_appl(ax::Int64, k::Int64, b::Int64)
    if ax == 1
        return 0
    elseif ax == 2
        b_k = get_kth_bit(b, k)
        if b_k == 0
            return 1
        else
            return 3
        end
    elseif ax == 3
        b_k = get_kth_bit(b, k)
        if b_k == 0
            return 0
        else
            return 2
        end
    elseif ax == 0
        return 0
    end
end


function pauli_modify_bitstring(ax::Int64, k::Int64, b::Int64)
    # Save comparison of ax == 0 since loop excludes this possibility
    if ax == 3
        return b
    end
        return xor(b, 1<<(k-1))
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
            psi_new[new_bit+1] = phase_shift(tmp[j+1], new_phase)
        end
        tmp .= psi_new
    end
    if weight == 0
        psi_new .= psi
    end
end