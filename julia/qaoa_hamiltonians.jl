using Erdos

include("operator.jl")


function max_cut_hamiltonian(g::Network)
    if !has_edge_property(g, "weight")
        error("Graph must be weighted (potentially all equal weights)")
    end
    operator = Operator([], [])
    n = length(vertices(g))

    for e in edges(g)
        w = edge_property(g, e)["weight"]
        i, j = e

        l = zeros(Int64, n)

        push!(operator.paulis, pauli_string_to_pauli(l))
        push!(operator.coeffs, -w/2.0)

        l[i] = 3
        l[j] = 3

        push!(operator.paulis, pauli_string_to_pauli(l))
        push!(operator.coeffs, +w/2.0)
    end

    op_simplify!(operator)
    return operator
end


function random_regular_max_cut_hamiltonian(n::Int, k::Int; seed=-1)
    g = random_regular_graph(n, k, Network)
    weights = EdgeMap(g, e -> rand())
    add_edge_property!(g, "weight", weights)
    return max_cut_hamiltonian(g)
end