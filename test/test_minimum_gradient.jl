using LinearAlgebra
using Distributions
using Random
using Test
using AdaptBarren

rng = MersenneTwister(42)

@testset "Test Optimizer Minimum Gradient" begin
    # Set up problem Hamiltonian
    n = 2
    psi_0 = ones(ComplexF64,2^n);
    psi_0[2] = -1.0 + 0.0im
    psi_0[3] = -1.0 + 0.0im
    psi_0 /= norm(psi_0)
    initial_state = copy(psi_0)
    en_0 = -2.0
    ham = Operator([Pauli(1, 0, 0, 0), Pauli(2, 0, 0, 0)], [1.0+0.0im, 1.0+0.0im])

    # Set up ansatz
    k = 5
    ans = random_two_local_ansatz(n, k, [0,1,2,3]; rng=rng)
    ans = map(Operator, ans)
    dist = Uniform(-pi/10,+pi/10)
    known_minimum = zeros(Float64, k)
    initial_point = rand(rng, dist, k)

    # Optimize
    opt = make_opt(Dict("name" => "LD_LBFGS", "maxeval" => 100000), initial_point)
    vqe_en, vqe_pt, _, _ = commuting_vqe(ham, ans, opt, initial_point, n, psi_0)

    # Test VQE Success
    @test abs(vqe_en - en_0) <= 1e-8
    #@show vqe_pt
    #@show psi_0
    #@show initial_state

    # Set up gradient computation
    grad = similar(vqe_pt)
    state = similar(psi_0)
    tmp1 = similar(psi_0)
    tmp2 = similar(psi_0)
    tmp3 = similar(psi_0)
    output_state = similar(psi_0)
    fn_evals = []
    grad_evals = []
    eval_count = 0

    # Accumulate gradient norms for later testing
    grad_norms = []

    # Run gradient computation
    cost_fn_res = _cost_fn_commuting_vqe(vqe_pt, grad, ans, ham, fn_evals, grad_evals, eval_count, state, initial_state, tmp1, tmp2, tmp3, output_state)
    push!(grad_norms, norm(grad))
    @test abs(cost_fn_res - en_0) <= 1e-8
    @test norm(grad) <= 1e-8
    #@show grad_evals
    
    # Run another gradient computation at another point
    cost_fn_res = _cost_fn_commuting_vqe(known_minimum, grad, ans, ham, fn_evals, grad_evals, eval_count, state, initial_state, tmp1, tmp2, tmp3, output_state)
    push!(grad_norms, norm(grad))
    @test abs(cost_fn_res - en_0) <= 1e-8
    @test norm(grad) <= 1e-8
    #@show grad_evals

    # Run a non-zero gradient point
    off_minimum = ones(Float64, k)
    cost_fn_res = _cost_fn_commuting_vqe(off_minimum, grad, ans, ham, fn_evals, grad_evals, eval_count, state, initial_state, tmp1, tmp2, tmp3, output_state)
    push!(grad_norms, norm(grad))
    #@show grad_evals

    # Test consistency with sample_points gradient collection
    #@show grad_norms
    points_to_sample = Vector{Vector{Float64}}()
    append!(points_to_sample, [vqe_pt])
    append!(points_to_sample, [known_minimum])
    append!(points_to_sample, [off_minimum])
    #@show points_to_sample
    #@show typeof(points_to_sample)
    _, sampled_grads, _, _ = sample_points(ham, ans, initial_state, 300; rng=rng, dist=nothing, point=points_to_sample, use_norm=true)
    #@show sampled_grads

    # Test that sampling points with given method matches the output of sampling them individually
    @test sampled_grads == grad_norms
end