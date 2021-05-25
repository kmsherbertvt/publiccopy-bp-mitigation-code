using Random
using LinearAlgebra
using StatsBase
using Erdos
using Glob


include("operator.jl")
include("pauli.jl")
include("vqe.jl")
include("pools.jl")
include("callbacks.jl")
include("qaoa_hamiltonians.jl")


Random.seed!(42)

pool_of_hope = [
    "YXYYZIYZXY",
    "ZZZYYIYXII",
    "YYXYIIZYZY",
    "XYIXXXXYZY",
    "YZZYIXIXYI",
    "XZYYYZXIII",
    "XIIZIXXZZY",
    "YYIXYXXXII", 
    "ZZXZYYIZZY", 
    "IXZXXXXYII", 
    "IIZYXIIIII", 
    "IYIZZYXIZY", 
    "XYXYXIXXYI", 
    "ZIZIIZYYZY", 
    "IXXIXZZIYI", 
    "IZIZIYYZZY", 
    "YXYZZXXXZY", 
    "ZYIYIYXYXY"
]

pool_of_hope = map(s -> pauli_string_to_pauli(s), pool_of_hope)


function random_hamiltonian(n, k)
    pauli_strings = []
    support = k * n^4
    while length(pauli_strings) <= support
        new_string = String(sample(['I', 'X', 'Y', 'Z'], 10, replace=true))
        if !(new_string in pauli_strings)
            push!(pauli_strings, new_string)
        end
    end
    coeffs = rand(Float64, support)
    paulis = map(s -> pauli_string_to_pauli(s), pauli_strings)
    return Operator(paulis, coeffs)
end


function run_experiment(n, seed, pool_name)
    n = 10
    Random.seed!(seed)
    pool = pool_of_hope

    # Define Hamiltonian
    ham = random_hamiltonian(n, 1)

    # Define optimizer
    optimizer = "LD_LBFGS"

    # Callbacks
    callbacks = Function[ParameterStopper(2000)]

    # Define path for data storage
    path = "/home/gbarron/data/mcp_random_vlad/$n"
    mkpath(path)
    #rm.(glob("$path/*.csv"))

    # Allocate arrays
    state = zeros(ComplexF64, 2^n)
    state[1] = 1.0 + 0.0im

    # Run ADAPT-VQE
    result = adapt_vqe(ham, pool, n, optimizer, callbacks, initial_parameter=1e-5, state=state, path=path)

    # Do gradient sampling on resulting ansatz
    ansatz = result.paulis
end


for n=[10]
    println("Num qubits: $n")
    for seed=[1]
        for pool_name in ["asdf"]
            @time run_experiment(n, seed, pool_name)
        end
    end
end
