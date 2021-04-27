include("pauli.jl")


function two_local_pool(n::Int64, axes=[1,2,3])
    pool = Array{Pauli,1}()

    for pair in Iterators.product(ntuple(i->1:n, 2)...)
        i,j = pair
        for pair2 in Iterators.product(ntuple(i->axes, 2)...)
            a,b = pair2
            l = zeros(Int64, n)
            l[i] = a
            l[j] = b
            pauli = pauli_string_to_pauli(l)
            push!(pool, pauli)
        end
    end

    return pool
end