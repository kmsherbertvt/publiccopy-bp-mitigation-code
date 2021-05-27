using Distributed
@everywhere using Random
@everywhere using LinearAlgebra
@everywhere using Erdos
@everywhere using Glob


@everywhere include("operator.jl")
@everywhere include("pauli.jl")
@everywhere include("vqe.jl")
@everywhere include("pools.jl")
@everywhere include("callbacks.jl")
@everywhere include("qaoa_hamiltonians.jl")


@everywhere Random.seed!(42)


@everywhere function run_experiment(n, seed, pool_name)
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
    path = "/home/gbarron/data/bps/maxcut/$n/$pool_name/$seed"
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

    # Run ADAPT-VQE
    result = adapt_vqe(ham, pool, n, optimizer, callbacks, initial_parameter=1e-10, state=state, path=path)

    # Do gradient sampling on resulting ansatz
    ansatz = result.paulis
end

inputs = []

for n=[4, 6, 8, 10, 12, 14, 16, 18, 20]
    for seed=1:40
        for pool_name in ["nchoose2local"]
            push!(inputs, [n, seed, pool_name])
            #@time run_experiment(n, seed, pool_name)
        end
    end
end

pmap(i -> run_experiment(i...), inputs)