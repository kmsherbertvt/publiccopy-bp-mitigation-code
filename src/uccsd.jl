using IterTools


function _swap_term(i::Int, a::Int, n::Int)
    s_1 = repeat(["I"], n)
    s_2 = repeat(["I"], n)

    s_1[i] = "Y"
    s_1[a] = "X"

    s_2[i] = "X"
    s_2[a] = "Y"

    reverse!(s_1)
    reverse!(s_2)

    return Operator([
        pauli_string_to_pauli(join(s_1)),
        pauli_string_to_pauli(join(s_2)),
    ],[
        -1.0 + 0.0im, 1.0 + 0.0im
    ])
end

function _z_string_term(i::Int, a::Int, n::Int)
    s = repeat(["I"], n)
    for k=(i+1):(a-1)
        s[k] = "Z"
    end
    reverse!(s)
    # Debug to this step!
    return Operator([pauli_string_to_pauli(join(s))], [1.0 + 0.0im])
end

function cluster_sing_op(p::Int, q::Int, n::Int)
    """ Implements Eq A1
    https://arxiv.org/pdf/1701.02691.pdf

    The operator that this produces is anti-Hermitian!
    """
    if p<=q error("Must have p>q") end
    op = _z_string_term(q, p, n) * _swap_term(q, p, n)
    op.coeffs .*= (0.0 + 0.5im)
    op.coeffs .*= (-1.0 + 0.0im)
    return op
end

function _z_string_doub_term(a::Int, b::Int, i::Int, j::Int, n::Int)
    s = repeat(["I"], n)
    for k=(i+1):(j-1)
        s[k] = "Z"
    end
    for l=(a+1):(b-1)
        s[l] = "Z"
    end
    reverse!(s)
    return Operator([pauli_string_to_pauli(join(s))], [1.0 + 0.0im])
end

function pauli_inds_to_pauli(pairs, n::Int, coeff)
    s = repeat(["I"], n)
    for (a,i)=pairs
        s[i] = a
    end
    reverse!(s)
    return Operator([pauli_string_to_pauli(join(s))], [coeff])
end

function _doub_sum_term(a::Int, b::Int, i::Int, j::Int, n::Int)
    terms = [
        pauli_inds_to_pauli([("X", i), ("X", j), ("Y", a), ("X", b)], n, 1.0+0.0im),
        pauli_inds_to_pauli([("Y", i), ("X", j), ("Y", a), ("Y", b)], n, 1.0+0.0im),
        pauli_inds_to_pauli([("X", i), ("Y", j), ("Y", a), ("Y", b)], n, 1.0+0.0im),
        pauli_inds_to_pauli([("X", i), ("X", j), ("X", a), ("Y", b)], n, 1.0+0.0im),

        pauli_inds_to_pauli([("Y", i), ("X", j), ("X", a), ("X", b)], n, -1.0+0.0im),
        pauli_inds_to_pauli([("X", i), ("Y", j), ("X", a), ("X", b)], n, -1.0+0.0im),
        pauli_inds_to_pauli([("Y", i), ("Y", j), ("Y", a), ("X", b)], n, -1.0+0.0im),
        pauli_inds_to_pauli([("Y", i), ("Y", j), ("X", a), ("Y", b)], n, -1.0+0.0im)
    ]
    op = Operator([], [])
    for o in terms
        op = op+o
    end
    return op
end

function cluster_doub_op(a::Int, b::Int, i::Int, j::Int, n::Int)
    """ Implements Eq A2
    https://arxiv.org/pdf/1701.02691.pdf

    The operator that this produces is anti-Hermitian!
    """
    if !(b>a>j>i) error("Must have b>a>j>i") end
    op = _z_string_doub_term(a, b, i, j, n) * _doub_sum_term(a, b, i, j, n)
    op.coeffs ./= (8.0 + 0.0im)
    op.coeffs .*= (0.0 + 1.0im)
    op.coeffs .*= (1.0 + 0.0im)
    return op
end

function pool_singles(n::Int)
    """ Produces a set of Hermitian operators for the singles pool
    """
    res = []
    for (p,q)=product(1:n,1:n)
        if !(p>q) continue end
        op = cluster_sing_op(p, q, n)
        op.coeffs .*= 1im
        push!(res, op)
    end
    return res
end

function pool_doubles(n::Int)
    """ Produces a set of Hermitian operators for the doubles pool
    """
    res = []
    for (a,b,i,j)=product(1:n,1:n,1:n,1:n)
        if !(b>a>j>i) continue end
        op = cluster_doub_op(a, b, i, j, n)
        op.coeffs .*= 1im
        push!(res, op)
    end
    return res
end

function uccsd_pool(n::Int)
    """ Produces a set of Hermitian operators for the UCCSD pool
    """
    res = []
    append!(res, pool_singles(n))
    append!(res, pool_doubles(n))
    return res
end