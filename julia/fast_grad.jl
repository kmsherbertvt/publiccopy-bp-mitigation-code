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

    # I think something is going wrong with the initial state being mutated
    # as the algorithm progresses...

    N = length(ansatz)

    # First we want to assign
    #   sigma <- |sigma_N>
    #   psi <- |psi_N>
    # these can be done simultaneously

    # sigma <- |phi>
    pauli_ansatz!(ansatz, pars, sigma, tmp1)

    # psi <- |phi>
    psi .= sigma

    # sigma <- H|sigma>
    ham_state_mult!(ham, sigma, tmp1, tmp2)

    # Now we'll start the main loop, starting at i=N
    for i=reverse(1:N)
        # Compute grad
        #  tmp1 <- P_N|psi_N>
        pauli_mult!(ansatz[i], psi, tmp1)
        #  e_val <- factor*<sigma|P_N|psi>
        e_val = dot(sigma, tmp1)
        grad = 2.0 * real(-1.0im * e_val)
        result[i] = grad

        # Unfold to time-reversed states
        #  sigma <- exp(-im*theta_i*P_i)|sigma>
        pauli_rotation!(ansatz[i], -pars[i], sigma, tmp1)
        #  psi <- exp(-im*theta_i*P_i)|psi>
        pauli_rotation!(ansatz[i], -pars[i], psi, tmp1)
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