using Random
using Test
using StatsBase
using LinearAlgebra

include("operator.jl")
include("pauli.jl")
include("simulator.jl")
include("pools.jl")
include("spin_chains.jl")

include("fast_grad.jl")

Random.seed!(42)

@testset "Handmade example" begin
    # Idea: Random ansatz, random Hamiltonian, random parameters
    # Compute gradient using FD and with `fast_grad!`
    n = 2
    N = 3
    eps = 1e-8
    tol = 1e-6

    ham = Operator([pauli_string_to_pauli("ZZ")],[1.0+0.0im])
    pars = [0.1, 0.2, 0.3]
    initial_state = zeros(ComplexF64, 2^n)
    initial_state[1] = 1.0 + 0.0im
    ansatz = [
        pauli_string_to_pauli("XX"),
        pauli_string_to_pauli("IY"),
        pauli_string_to_pauli("ZI")
    ]

    # Construction using `fast_grad!`
    result_fg = zeros(Float64, N)
    psi   = zeros(ComplexF64, 2^n)
    tmp1  = zeros(ComplexF64, 2^n)
    tmp2  = zeros(ComplexF64, 2^n)
    fast_grad!(ham, ansatz, pars, result_fg, psi, copy(initial_state), tmp1, tmp2)

    # Construction using finite difference
    result_fd = zeros(Float64, N)
    tmp1  = zeros(ComplexF64, 2^n)
    tmp2  = zeros(ComplexF64, 2^n)
    finite_difference!(ham, ansatz, pars, result_fd, copy(initial_state), tmp1, tmp2, eps)

    # Test difference!
    @show result_fd
    @show result_fg
    @test norm(result_fd - result_fg) <= tol
end

@testset "Random Gradient" begin
    # Idea: Random ansatz, random Hamiltonian, random parameters
    # Compute gradient using FD and with `fast_grad!`
    n = 4
    N = 10
    eps = 1e-8
    tol = 1e-6

    _pool = vcat(minimal_complete_pool(n), two_local_pool(n))
    ham = xyz_model(n, rand(Float64), rand(Float64), rand(Float64), true)
    pars = rand(Float64, N)
    initial_state = rand(ComplexF64, 2^n)
    initial_state /= norm(initial_state)
    ansatz = [sample(_pool) for i=1:N]

    # Construction using `fast_grad!`
    result_fg = zeros(Float64, N)
    psi   = zeros(ComplexF64, 2^n)
    tmp1  = zeros(ComplexF64, 2^n)
    tmp2  = zeros(ComplexF64, 2^n)
    fast_grad!(ham, ansatz, pars, result_fg, psi, copy(initial_state), tmp1, tmp2)

    # Construction using finite difference
    result_fd = zeros(Float64, N)
    tmp1  = zeros(ComplexF64, 2^n)
    tmp2  = zeros(ComplexF64, 2^n)
    finite_difference!(ham, ansatz, pars, result_fd, copy(initial_state), tmp1, tmp2, eps)

    # Test difference!
    println(result_fd)
    println(result_fg)
    @test norm(result_fd - result_fg) <= tol
end