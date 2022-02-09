using Distributed

@everywhere using Random
@everywhere using StatsBase
@everywhere using LinearAlgebra
@everywhere using Erdos
@everywhere using Glob
@everywhere using AdaptBarren
@everywhere Random.seed!(42)


@everywhere function run_experiment(n, seed, pool_name, depth = nothing)
    Random.seed!(seed)
    # Construct pool
    if pool_name == "nchoose2local"
        pool = two_local_pool(n)
    elseif pool_name == "mcp2local"
        pool = minimal_complete_pool(n)
    end

    # Define graph and Hamiltonian
    graph = Erdos.random_regular_graph(n, 3)
    ham = max_cut_hamiltonian(graph)

    # Define optimizer
    optimizer = "LD_LBFGS"

    # Callbacks
    callbacks = Function[
        ParameterStopper(200),
        MaxGradientStopper(1e-6)
        ]

    # Define path for data storage
    path = "./bps/maxcut-full/$n/$pool_name/$seed"
    mkpath(path)
    #rm.(glob("$path/*.csv"))

    # Store exact result
    ham_vec = diagonal_operator_to_vector(ham)
    min_ind = argmin(real(ham_vec))
    min_val = real(ham_vec[min_ind])
    min_bs = bitstring(min_ind)[64-n+1:end]
    open("$path/exact_result.csv", "w") do io
        write(io, "min_ind; min_val; min_bs\n")
        write(io, "$min_ind; $min_val; $min_bs\n")
    end

    # Allocate arrays
    state = ones(ComplexF64, 2^n)
    state /= norm(state)

    if depth === nothing
        # Run ADAPT-VQE
        result = adapt_vqe(ham, pool, n, optimizer, callbacks, initial_parameter=1e-10, initial_state=state, path=path)

        n_layers = length(result.paulis)

        for i=1:n_layers
            # Just to make sure this bug doesn't happen again
            state = ones(ComplexF64, 2^n)
            state /= norm(state)

            ansatz = result.paulis[2:i] # The first entry is nothing
            if length(ansatz) == 0
                continue
            end
            ansatz = map(x -> x, ansatz)

            VQE(
                ham,
                ansatz,
                "random_sampling",
                zeros(Float64, i), # This isn't actually used in random sampling"
                n,
                state,
                "$path/rand_layer_$i.h5"#;
                #rand_range=(-pi, +pi),
                #num_samples=500
            )
        end
    else
	ansatz = sample(pool, depth; replace=true)
        state = ones(ComplexF64, 2^n)
        state /= norm(state)
        VQE(
            ham,
            ansatz,
            "random_sampling",
            zeros(Float64, depth), # This isn't actually used in random sampling"
            n,
            state,
            "$path/sampans_layer_$depth.h5"#;
            #rand_range=(-pi, +pi),
            #num_samples=500
        )
    end
end

inputs = []

NMAX = parse(Int,ARGS[1])
SEEDS = parse(Int,ARGS[2])

for seed=1:SEEDS
    for n=4:2:NMAX
	for d=[2, 5, 10, 20, 40, 60, 80, 100, 200, 300]
            for pool_name in ["nchoose2local"]
                push!(inputs, [n, seed, pool_name, d])
                push!(inputs, [n, seed, pool_name])
            end
	end
    end
end

#result = run_experiment(inputs[1]...)

result = pmap(i -> run_experiment(i...), inputs)
