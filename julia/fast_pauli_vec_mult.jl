import IterTools: enumerate
import DataStructures: SortedSet



function phase_shift(alpha::ComplexF64, i::UInt8)
    if i == 0
        return     alpha.re + im*alpha.im
    elseif i == 1
        return  im*alpha.re - alpha.im 
    elseif i == 2
        return    -alpha.re - im*alpha.im
    else
        return -im*alpha.re + alpha.im 
    end
end


function pauli_masks(res::Array{Int64,1}, pauli_str::Array{Int64,1})
    for (i,ax)=enumerate(reverse(pauli_str))
        res[ax+1] += 2^(i-1)
    end
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
    x = count_ones((pm[3] | pm[4]) & a) % 2
    y = count_ones(pm[3]) % 4

    alpha = y
    beta = 2*x

    return UInt8((alpha+beta) % 4)
end


function pauli_apply(pm::Array{Int64, 1}, a::Int64)
    # pm = pauli_mask input
    return xor((pm[2]|pm[3]),a)
end


function invert_phase(i::Int64)
    # equivalent to compl conj
    return mod(i-2*(i%2),4)
end