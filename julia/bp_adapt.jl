using Random
using LinearAlgebra
using Erdos
using Glob


include("operator.jl")
include("pauli.jl")
include("vqe.jl")
include("pools.jl")
include("callbacks.jl")
include("qaoa_hamiltonians.jl")


Random.seed!(42)


function run_experiment(n, seed, pool_name)
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
    callbacks = Function[ParameterStopper(20)]

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
    result = adapt_vqe(ham, pool, n, optimizer, callbacks, initial_parameter=1e-5, state=state, path=path)
end


for n=[4]
    for seed=1:2
        for pool_name in ["nchoose2local", "mcp2local"]
            run_experiment(n, seed, pool_name)
        end
    end
end