using Test
using StatsBase
using LinearAlgebra

include("operator.jl")
include("pauli.jl")
include("simulator.jl")
include("pools.jl")

include("fast_grad.jl")

@testset "Random Gradient" begin
    # Idea: Random ansatz, random Hamiltonian, random parameters
    # Compute gradient using FD and with `fast_grad!`
    n = 4
    N = 10
    eps = 1e-8
    tol = 1e-8

    _pool = vcat(minimal_complete_pool(n), two_local_pool(n))
    _h = rand(ComplexF64, 2^n, 2^n)
    _h += conj(_h)
    _h /= 2.0

    ham = matrix_to_operator(_h)
    pars = rand(Float64, N)
    initial_state = zeros(ComplexF64, 2^n)
    initial_state[1] = 1.0 + 0.0im
    ansatz = [sample(_pool) for i=1:N]

    # Construction using `fast_grad!`
    result_fg = zeros(Float64, N)
    psi   = zeros(ComplexF64, 2^n)
    initial_state = zeros(ComplexF64, 2^n)
    initial_state[1] = 1.0 + 0.0im
    tmp1  = zeros(ComplexF64, 2^n)
    tmp2  = zeros(ComplexF64, 2^n)
    fast_grad!(ham, ansatz, pars, result_fg, psi, initial_state, tmp1, tmp2)

    # Construction using finite difference
    initial_state = zeros(ComplexF64, 2^n)
    initial_state[1] = 1.0 + 0.0im
    result_fd = zeros(Float64, N)
    tmp1  = zeros(ComplexF64, 2^n)
    tmp2  = zeros(ComplexF64, 2^n)
    finite_difference!(ham, ansatz, pars, result_fd, initial_state, tmp1, tmp2, eps)

    # Test difference!
    @test norm(result_fd - result_fg) <= tol
end