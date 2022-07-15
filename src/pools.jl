using IterTools

"""
    minimal_complete_pool(n::Int64)

Return the minimal complete pool on `n` qubits corresponding
with the `V` pool in the qubit-ADAPT paper.
"""
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

"""
    one_local_pool_from_axes(n::Int64, axes::Vector{Int64})

Return the pool on `n` qubits with single qubit operators corresponding
to each element of `axes`.

Has size `n * |axes|`.
"""
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

"""
    two_local_pool_from_pairs(n::Int64, pairs::Vector{Tuple{Int64, Int64}}; include_reverses = true)

Return the pool on `n` qubits with all two qubit Pauli strings defined in `pairs`.

# Examples
The `pairs` vector `[(1,1), (2,3)]` corresponds to all `X_i X_j` and `Y_i Z_j` terms acting on all
pairs of qubits. If `include_reverses` is `true`, then terms like `Y_j Z_i` are included.
"""
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


"""
    two_local_pool(n::Int64, axes=[0,1,2,3])

Returns the two local pool on `n` qubits using elements of `axes` as terms in the
Pauli strings.
"""
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


"""
    random_two_local_ansatz(n::Int64, k::Int64, axes=[0,1,2,3]; rng=_DEFAULT_RNG)

Produce a pool on `n` qubits consisting of `k` randomly sampled Pauli strings
that are two local and determined by the elements of `axes`.
"""
function random_two_local_ansatz(n::Int64, k::Int64, axes=[0,1,2,3]; rng=_DEFAULT_RNG)
    return sample(rng, two_local_pool(n, axes), k; replace=true)
end