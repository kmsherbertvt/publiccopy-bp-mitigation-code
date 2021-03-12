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


function pauli_masks(pauli_str::Array{Int64,1})
    res = [0, 0, 0, 0]
    for (i,ax)=enumerate(reverse(pauli_str))
        res[ax+1] += 2^(i-1)
    end
    deleteat!(res, 1)
    return res
end


function hamming_weight(a::Int64)
    res = 0
    for k=0:64-1 # could probably stop earlier...?
        res += get_kth_bit(a, k)
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
    px, py, pz = pm
    x = hamming_weight((py | pz) & a) % 2
    y = hamming_weight(py) % 4

    alpha = y
    beta = 2*x

    return (alpha+beta) % 4
end


function pauli_apply(pm::Array{Int64, 1}, a::Int64)
    # pm = pauli_mask input
    return xor((pm[1]|pm[2]),a)
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
            new_elt = phase_shift(tmp[j+1], new_phase)
            psi_new[new_bit+1] = new_elt
        end
        tmp .= psi_new
    end
    if weight == 0
        psi_new .= psi
    end
end