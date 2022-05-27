using LinearAlgebra
using Random
using Test
using DataFrames
using AdaptBarren

@testset "ADAPT Simple Run" begin
    n = 2
    operator = random_regular_max_cut_hamiltonian(n, n-1)
    
    gse = get_ground_state(operator)
    ham_vec = real(diagonal_operator_to_vector(operator))

    pool = two_local_pool(n)
    optimizer = "LD_LBFGS"
    callbacks = Function[ParameterStopper(10), MaxGradientStopper(1e-10)]
            
    state = ones(ComplexF64, 2^n)
    state /= norm(state)

    result = adapt_vqe(operator, pool, n, optimizer, callbacks; initial_parameter=0.0, initial_state=state)

    en_adapt = last(result.energy)

    en_err = abs(gse - en_adapt)

    df = DataFrame(layer=[], err=[], n=[], overlap=[])
    num_layers = length(result)
    for k = 1:num_layers
        d = result[k]
        #println(d)
        en_err = safe_floor(d[:energy]-gse)
        gse_overlap = ground_state_overlap(ham_vec, d[:opt_state])
        push!(df, Dict(:layer=>k, :err=>en_err, :n=>n, :overlap=>gse_overlap))
    end
    #println("\n\n\n")
    #println(df)
    #println("\n\n\n")
    #println(result.energy)
    #println("\n\n\n")
    for (i,st)=enumerate(result.opt_state)
        #println("Layer $i: $st")
    end
end
