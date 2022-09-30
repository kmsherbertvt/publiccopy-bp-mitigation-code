using LinearAlgebra
using StatsBase
using Test
using Random
using AdaptBarren
using Distributions
using Optim
using LineSearches
using Retry

@testset "Point sampling gradient minimum VQE RUN" begin
    for seed=1:500
        rng = MersenneTwister(seed)
        num_samples = 1
        n = 4
        d = 10
        use_norm = true
        ham = random_regular_max_cut_hamiltonian(n, n-1; weighted = true, rng=rng)
        initial_state = uniform_state(n)
        dist = 0.0
        opt = Optim.LBFGS(
            m=100,
            alphaguess=LineSearches.InitialStatic(alpha=0.5),
            linesearch=LineSearches.HagerZhang()
        )
        #opt = Dict("name" => "LD_LBFGS", "maxeval" => 5000)

        ansatz = random_two_local_ansatz(n, d; rng=rng)
        op_ans = collect(map(Operator, ansatz))
        
        (initial_point, min_en, opt_pt) = @repeat 10 try
            initial_point = rand(rng, Uniform(-pi, +pi), length(ansatz))
            min_en, opt_pt, _, _ = commuting_vqe(ham, op_ans, opt, copy(initial_point), n, copy(initial_state), nothing, nothing, true)
            (initial_point, min_en, opt_pt)
        catch err
            @retry if true end
        end

        _, result_grads, _, _ = sample_points(ham, op_ans, copy(initial_state), num_samples; dist=dist, point=opt_pt, use_norm=use_norm, rng=rng)
        @test maximum(result_grads) <= 1e-3
    end
end