using LinearAlgebra
using TensorOperations
using DataStructures
using LoopVectorization

include("fast_pauli_vec_mult.jl")


function get_pauli(i::Int64)
    if i == 0
        return [1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im]
    elseif i == 1
        return [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im]
    elseif i == 2
        return [0.0+0.0im 0.0-1.0im; 0.0+1.0im 0.0+0.0im]
    elseif i == 3
        return [1.0+0.0im 0.0+0.0im; 0.0+0.0im -1.0+0.0im]
    end
end


function pauli_str(axes::Array{Int64})
    return foldl(kron, map(get_pauli, axes))
end


function mcp_g_list(n::Int64)
    result = Array{Array{Int64, 1}, 1}()
    for i = 1:n
        l = zeros(Int64, n)
        l[i] = 2
        push!(result, l)
    end
    for i in 1:n-1
        l = zeros(Int64, n)
        l[i] = 3
        l[i+1] = 2
        push!(result, l)
    end
    return result
end


function pauli_ansatz(
        axes::Array{Array{Int64,1},1}, 
        pars::Array{Float64,1}, 
        current_state::Union{Array{ComplexF64,1},Nothing} = nothing
        )
    num_pars = length(axes)
    num_qubits = length(axes[1])
    if current_state === nothing
        current_state = zeros(ComplexF64, 2^num_qubits)
        current_state[1] = 1.0+0.0im
    end
    if length(pars) != num_pars
        throw(ExceptionError())
    end
    for (theta,ax)=zip(pars,axes)
        mat = pauli_str(ax)
        mat *= -1.0im*sin(theta)
        c = cos(theta)
        for i=1:2^num_qubits
            mat[i,i] += c
        end
        current_state = mat*current_state
    end
    return current_state
end


function pauli_ansatz!(
        axes::Union{Array{Array{Int64,1},1},Array{Pauli,1}}, 
        pars::Array{Float64,1}, 
        result::Array{ComplexF64,1}, # pre-alloc # also initial state
        tmp::Array{ComplexF64,1}, # pre-alloc
        )
    num_pars = length(axes)
    num_qubits = Int(log2(length(result)))
    pm = [0, 0, 0, 0]
    N = length(result)
    D = length(pars)
    for d=1:D
        theta = pars[d]
        c = cos(theta)
        s = sin(theta)
        
        tmp .= c .* result

        pauli_masks(pm, axes[d])
        for i=0:N-1
            j = pauli_apply(pm, i)
            phase = UInt8((pauli_phase(pm, i)+1) % 4)

            @inbounds r = result[i+1]
            @inbounds tmp[j+1] -= phase_shift(r*s, phase)
        end
        pm[1] = 0; pm[2] = 0; pm[3] = 0; pm[4] = 0
        result .= tmp
    end
end

