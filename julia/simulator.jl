using LinearAlgebra
using TensorOperations

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
        mat::Union{Array{ComplexF64,2},Nothing} = nothing,
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
        @tensor begin
            current_state[i] = mat[i,j]*current_state[j]
        end
    end
    return current_state
end


function pauli_ansatz_new!(
        axes::Array{Array{Int64,1},1}, 
        pars::Array{Float64,1}, 
        result::Union{Array{ComplexF64,1},Nothing}, # pre-alloc # also initial state
        tmp::Union{Array{ComplexF64,1},Nothing}, # pre-alloc
        )
    num_pars = length(axes)
    num_qubits = length(axes[1])
    if length(pars) != num_pars
        throw(ExceptionError())
    end

    for (theta,ax)=zip(pars,axes)
        c = cos(theta)
        s = sin(theta)
        
        tmp .= result
        pauli_vec_mult!(result, ax)
        tmp *= c
        result *= -1.0im*s
        result += tmp
    end
end

