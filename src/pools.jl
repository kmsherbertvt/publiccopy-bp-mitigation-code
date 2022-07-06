using IterTools

function minimal_complete_pool(n::Int64)
    res = Array{String,1}(["ZY","YI"])
    for i=3:n
        tempres = Array{String,1}()
        for pstr in res
            push!(tempres,"Z"*pstr)
        end
        push!(tempres,"Y"*("I"^(i-1)))
        push!(tempres,"IY"*("I"^(i-2)))
        res = copy(tempres)
    end
    return map(pauli_string_to_pauli,res)
end

function one_local_pool_from_axes(n::Int64, axes::Vector{Int64})
    pool = Array{Pauli{UInt64},1}()
    for i=1:n
        for a=axes
            p_new = pauli_indices_to_pauli(n, [(i, a)])
            if !(p_new in pool)
                push!(pool, p_new)
            end
        end
    end
    return pool
end

function two_local_pool_from_pairs(n::Int64, pairs::Vector{Tuple{Int64, Int64}}; include_reverses = true)
    pool = Array{Pauli{UInt64},1}()
    for (i,j)=product(1:n,1:n)
        if (!include_reverses) & (i>j) continue end
        for (a,b)=pairs
            p_new = pauli_indices_to_pauli(n, [(i, a), (j, b)])
            if !(p_new in pool)
                push!(pool, p_new)
            end
        end
    end
    return pool
end

function two_local_pool(n::Int64, axes=[0,1,2,3])
    pool = Array{Pauli{UInt64},1}()

    for pair in Iterators.product(ntuple(i->1:n, 2)...)
        i,j = pair
        if i < j
            for pair2 in Iterators.product(ntuple(i->axes, 2)...)
                a,b = pair2
                if a == b == 0
                    continue
                end
                l = zeros(Int64, n)
                l[i] = a
                l[j] = b
                pauli = pauli_string_to_pauli(l)
                push!(pool, pauli)
            end
        end
    end

    return pool
end


function random_two_local_ansatz(n::Int64, k::Int64, axes=[0,1,2,3]; rng=_DEFAULT_RNG)
    return sample(rng, two_local_pool(n, axes), k; replace=true)
end