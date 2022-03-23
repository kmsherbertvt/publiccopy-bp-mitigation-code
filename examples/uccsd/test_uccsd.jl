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