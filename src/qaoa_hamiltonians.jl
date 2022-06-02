using Random
using Erdos


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


function max_cut_hamiltonian(g)
    #if has_edge_property(g, "weight")
    #    error("Graph must be unweighted")
    #end
    operator = Operator([], [])
    n = length(Erdos.vertices(g))

    for e in Erdos.edges(g)
        w = rand(Float64)
        #w = 1.0
        i, j = e

        l = zeros(Int64, n)

        push!(operator.paulis, pauli_string_to_pauli(l))
        push!(operator.coeffs, -w/2.0)

        l[i] = 3
        l[j] = 3

        push!(operator.paulis, pauli_string_to_pauli(l))
        push!(operator.coeffs, +w/2.0)
    end

    #@warn "I'm still simplifying operators in the max cut Hamiltonian..."
    #op_simplify!(operator)
    return operator
end


function random_regular_max_cut_hamiltonian(n::Int, k::Int; seed=-1)
    g = random_regular_graph(n, k, Network)
    weights = EdgeMap(g, e -> rand())
    add_edge_property!(g, "weight", weights)
    return max_cut_hamiltonian(g)
end
