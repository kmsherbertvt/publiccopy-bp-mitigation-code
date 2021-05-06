using Test
using StatsBase
using LinearAlgebra

include("operator.jl")
include("pauli.jl")
include("simulator.jl")
include("pools.jl")
include("spin_chains.jl")

include("fast_grad.jl")

@testset "Random Gradient" begin
    # Idea: Random ansatz, random Hamiltonian, random parameters
    # Compute gradient using FD and with `fast_grad!`
    n = 4
    N = 10
    eps = 1e-8
    tol = 1e-8

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