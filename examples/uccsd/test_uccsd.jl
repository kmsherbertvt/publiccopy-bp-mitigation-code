using AdaptBarren
using Test

function terms_to_pauli(terms, n)
    output_string = repeat(['I'], n)
    if length(terms) == 1
        if length(terms[1]) == 0
            return pauli_string_to_pauli(join(output_string))
        end
    end
    for p in terms
        axis = p[1]
        qubit = parse(Int64, p[2:end]) + 1 # +1 because 0-indexing on given file

        output_string[qubit] = axis
    end

    return pauli_string_to_pauli(join(output_string))
end

function get_ham()
    operator = Operator([], [])
    n = 12
    open("h6_1A.ham") do f
        for line in readlines(f)
            line = replace(line, "["=>"")
            line = replace(line, "]"=>"")
            sp = split(line, " ")
            coeff = ComplexF64(parse(Float64, sp[1]))
            terms = sp[2:end]
            pauli = terms_to_pauli(terms, n)
            push!(operator.paulis, pauli)
            push!(operator.coeffs, coeff)
        end
    end
    return operator
end


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
    return Operator([pauli_string_to_pauli(join(s))], [1.0 + 0.0im])
end

function cluster_sing_op(p::Int, q::Int, n::Int)
    """ Implements Eq A1
    https://arxiv.org/pdf/1701.02691.pdf
    """
    if p<=q error("Must have p>q") end
    op = _z_string_term(q, p, n) * _swap_term(q, p, n)
    op.coeffs .*= (0.0 + 0.5im)
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
    """
    if !(b>a>j>i) error("Must have b>a>j>i") end
    op = _z_string_doub_term(a, b, i, j, n) * _doub_sum_term(a, b, i, j, n)
    op.coeffs ./= (8.0 + 0.0im)
    op.coeffs .*= (0.0 + 1.0im)
    return op
end

@testset "Test that terms in ops commute, doubles" begin
    using IterTools
    for n=4:6
        for (a,b,i,j)=product(1:n,1:n,1:n,1:n)
            if !(b>a>j>i) continue end
                @test subcommutes(cluster_doub_op(a, b, i, j, n))
        end
    end
end

@testset "Test that terms in ops commute, singles" begin
    using IterTools
    for n=4:6
        for (p,q)=product(1:n,1:n)
            if p<=q continue end
            op = cluster_sing_op(p, q, n)
            @test subcommutes(op)
        end
    end
end


# This was another attempt at an implementation that 
# I did not get working
    #function _z_bar(i::Int, n::Int)
    #    s = repeat("I", n-i) * repeat("Z", i)
    #    return pauli_string_to_pauli(s)
    #end
    #
    #function _z_bar_op(i::Int, n::Int)
    #    return Operator([_z_bar(i, n)], [1.0 + 0.0im])
    #end
    #
    #function _sigma_minus(i::Int, n::Int)
    #    s_x = repeat(["I"], n)
    #    s_y = repeat(["I"], n)
    #    s_x[i] = "X"
    #    s_y[i] = "Y"
    #    reverse!(s_x)
    #    reverse!(s_y)
    #
    #    return Operator([
    #        pauli_string_to_pauli(join(s_x)),
    #        pauli_string_to_pauli(join(s_y))
    #    ],[
    #        1.0 + 0.0im,
    #        0.0 - 1.0im
    #    ])
    #end
    #
    #function _sigma_plus(i::Int, n::Int)
    #    return dagger(_sigma_minus(i, n))
    #end
    #
    #function _a_lo_op(i::Int, n::Int)
    #    return _z_bar_op(i, n) * _sigma_minus(i, n)
    #end
    #
    #function _a_hi_op(i::Int, n::Int)
    #    return dagger(_a_lo_op(i, n))
    #end
    #
    ##function cluster_sing_op(p::Int, q::Int, n::Int)
    ##    x = _a_hi_op(p, n) * _a_lo_op(q, n)
    ##    return x - dagger(x)
    ##end
    ##
    ##function cluster_doub_op(p::Int, q::Int, r::Int, s::Int, n::Int)
    ##    x = _a_hi_op(p, n) * _a_hi_op(q, n) * _a_lo_op(r, n) * _a_lo_op(s, n)
    ##    return x - dagger(x)
    ##end
    #
    #
    #@testset "Test Z-bar term" begin
    #    i = 4
    #    n = 10
    #    ps_expected = "IIIIIIZZZZ"
    #    ps_actual = pauli_to_pauli_string(_z_bar(i, n), n)
    #    @test ps_actual == ps_expected
    #
    #    i = 1
    #    n = 5
    #    ps_expected = "IIIIZ"
    #    ps_actual = pauli_to_pauli_string(_z_bar(i, n), n)
    #    @test ps_actual == ps_expected
    #end