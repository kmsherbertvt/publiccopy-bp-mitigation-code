using AdaptBarren

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

operator = get_ham()
@show operator

print_operator(operator)