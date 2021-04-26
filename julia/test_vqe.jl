using LinearAlgebra
using Test

include("vqe.jl")
include("spin_chains.jl")


function two_local_ansatz(n::Int64, d::Int64=2)
    res = Array{String,1}()
    for _=1:d
        for i=1:n
            for j=1:n
                if i<=j continue end
                l = split('I'^n, "")
                l[i] = "X"
                l[j] = "Y"
                s = string(l...)
                push!(res, s)
            end
        end
    end
    return res
end


@testset "XXZ VQE" begin
    for L=4:5
        for Jxy=range(0.5, 2.0, length=2)
            for Jz=range(0.3, 1.0, length=2)
                for PBCs in [true]
                    xxzmat = xxz_matrix(L,Jxy,Jz,PBCs)
                    evals = eigvals(xxzmat)
                    xxzham = xxz_model(L,Jxy,Jz,PBCs)

                    ansatz_ops = map(pauli_string_to_pauli, two_local_ansatz(L, 2));

                    init_state = zeros(ComplexF64,2^L);
                    init_state[1] = 1.0;

                    tmp_state = zeros(ComplexF64,2^L)
                    #print("init state energy: ",exp_val(xxzham,init_state,tmp_state),'\n')

                    opt = Opt(:LN_COBYLA, length(ansatz_ops))
                    init_pt = ones(Float64,length(ansatz_ops))

                    vqe_results = VQE(xxzham,ansatz_ops,opt,init_pt,L,init_state)

                    #println("energy difference exact - vqe = ",evals[1]-vqe_results[1])
                    @test abs(evals[1]-vqe_results[1]) <= 1e-8
                end
            end
        end
    end
end