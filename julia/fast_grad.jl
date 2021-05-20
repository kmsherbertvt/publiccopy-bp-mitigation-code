include("operator.jl")
include("pauli.jl")
include("simulator.jl")


function fast_grad!(
        ham::Operator,
        ansatz::Vector{Pauli{T}},
        pars::Vector{Float64},
        result::Vector{Float64},
        psi::Vector{ComplexF64},
        sigma::Vector{ComplexF64}, # Initial state goes here!
        tmp1::Vector{ComplexF64},
        tmp2::Vector{ComplexF64}
    ) where T <: Unsigned

    N = length(ansatz)

    # psi <- |psi_1>
    psi .= sigma
    pauli_rotation!(ansatz[1], pars[1], psi, tmp1)

    # sigma <- |phi>
    pauli_ansatz!(ansatz, pars, sigma, tmp1)

    # sigma <- H|sigma>
    ham_state_mult!(ham, sigma, tmp1, tmp2)

    # sigma <- |sigma_1>
    pauli_ansatz!(reverse(ansatz[2:N]), -reverse(pars[2:N]), sigma, tmp1)

    for k=1:N
        # Compute grad
        # tmp1 <- P_k|psi_k>
        pauli_mult!(ansatz[k], psi, tmp1)
        result[k] = 2.0 * real(-1.0im * dot(sigma, tmp1))

        # Exit loop since N+1 element doesn't exist
        if k == N break end
        pauli_rotation!(ansatz[k+1], pars[k+1], sigma, tmp1)
        pauli_rotation!(ansatz[k+1], pars[k+1], psi, tmp1)
    end
end


function finite_difference!(
        ham::Operator,
        ansatz::Vector{Pauli{T}},
        pars::Vector{Float64},
        result::Vector{Float64},
        initial_state::Vector{ComplexF64},
        tmp1::Vector{ComplexF64},
        tmp2::Vector{ComplexF64},
        eps::Float64 = 1e-8
    ) where T <: Unsigned

    N = length(ansatz)

    for i=1:N
        eps_vec = zeros(Float64, length(pars))
        eps_vec[i] = eps

        # f(p+eps)
        tmp2 .= initial_state
        pauli_ansatz!(ansatz, pars + eps_vec, tmp2, tmp1)
        fn_plus = real(exp_val(ham, tmp2, tmp1))
        
        # f(p-eps)
        tmp2 .= initial_state
        pauli_ansatz!(ansatz, pars - eps_vec, tmp2, tmp1)
        fn_minu = real(exp_val(ham, tmp2, tmp1))

        # Compute grad
        result[i] = (fn_plus - fn_minu) / (2*eps)
    end
end