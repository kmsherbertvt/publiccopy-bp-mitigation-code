using Random
using Erdos


_DEFAULT_RNG = MersenneTwister(1234);


function _id_x_str(n::Int, k::Int)
    s = repeat(["I"], n)
    s[k] = "X"
    return join(s)
end

function qaoa_mixer(n::Int)
    paulis = [pauli_string_to_pauli(_id_x_str(n, k)) for k in range(1,n)]
    coeffs = repeat([1.0], n)
    return Operator(paulis, coeffs)
end


"""
    max_cut_hamiltonian(n::Int, edges::Vector{Tuple{Int, Int, T}}) where T<:Real

Return the max cut Hamiltonian acting on `n` qubits which corresponds to the edge
set `edges`.

# Examples

Elements of the edge set look like `(1,2,5.0)` corresponding to an edge between
vertices `1` and `2` with edge weight `5.0`.
"""
function max_cut_hamiltonian(n::Int, edges::Vector{Tuple{Int, Int, T}}) where T<:Real
    operator = Operator([], [])

    for (i,j,w)=edges
        l = zeros(Int64, n)

        push!(operator.paulis, pauli_string_to_pauli(l))
        push!(operator.coeffs, -w/2.0)

        l[i] = 3
        l[j] = 3

        push!(operator.paulis, pauli_string_to_pauli(l))
        push!(operator.coeffs, +w/2.0)
    end
    return operator
end


function get_random_unweighted_graph_edges(n::Int, k::Int; rng = _DEFAULT_RNG)
    seed = abs(rand(rng, Int64) % 10^7)
    g = random_regular_graph(n, k, Network, seed=seed)
    return [(i,j,1.0) for (i,j) in Erdos.edges(g)]
end


function randomize_edge_weights!(v::Vector{Tuple{Int, Int, T}}; rng = _DEFAULT_RNG) where T<:Number
    for i=1:length(v)
        a, b, _ = v[i]
        v[i] = (a, b, rand(rng, Float64))
    end
end


"""
    random_regular_max_cut_hamiltonian(n::Int, k::Int; rng = _DEFAULT_RNG, weighted = true)

Return a random Hamiltonian for a max cut problem on `n` qubits.

The corresponding graph is degree `k`. If an RNG is provided, this will be used to sample
the graph and edge weights. If `weighted` is true, the edge weights will be randomly sampled
from the uniform distribution `U(0,1)`.
"""
function random_regular_max_cut_hamiltonian(n::Int, k::Int; rng = _DEFAULT_RNG, weighted = true)
    v = get_random_unweighted_graph_edges(n, k; rng=rng)
    if weighted
        randomize_edge_weights!(v; rng=rng)
    end
    return max_cut_hamiltonian(n, v)
end
