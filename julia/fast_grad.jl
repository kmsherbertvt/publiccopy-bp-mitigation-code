include("operator.jl")
include("pauli.jl")
include("simulator.jl")


function fast_grad!(
        ham::Operator,
        ansatz::Vector{Pauli{T}},
        pars::Vector{Float64},
        result::Vector{Float64},
        sigma::Vector{ComplexF64},
        tmp1::Vector{ComplexF64},
        tmp2::Vector{ComplexF64},
        psi::Vector{ComplexF64}
    ) where T <: Unsigned

    N = length(ansatz)

    # Construct:
    #   sigma <- |sigma_N>
    #   psi <- |psi_N>
    pauli_ansatz!(ansatz, pars, sigma, tmp1)
    psi .= sigma
    ham_state_mult!(ham, sigma, tmp1, tmp2)

    for i=reverse(1:N)
        # Compute grad
        # tmp1 <- P_N|psi_N>
        pauli_mult!(ansatz[i], psi, tmp1)
        e_val = dot(sigma, tmp1)
        grad = 2.0 * real(-1.0im * e_val)
        result[i] = grad

        # Unfold to time-reversed states
        pauli_rotation!(ansatz[i], -pars[i], sigma, tmp1)
        pauli_rotation!(ansatz[i], -pars[i], psi, tmp1)
    end
end