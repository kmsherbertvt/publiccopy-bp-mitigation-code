using LinearAlgebra
using Test

include("vqe.jl")
include("spin_chains.jl")
include("operator.jl")
include("pools.jl")
include("callbacks.jl")


@testset "ADAPT Random Diagonal" begin
    for _=1:10
        for n=4:6
            println("starting test")
            v = rand(2^n)
            ground_state_energy = minimum(v)
            hamiltonian = Diagonal(v)
            operator = matrix_to_operator(hamiltonian)
            op_simplify!(operator)
            pool = two_local_pool(n)
            optimizer = "LN_COBYLA"
            callbacks = Function[ParameterStopper(10)]

            result = adapt_vqe(operator, pool, n, optimizer, callbacks)
            @show result
        end
    end
end