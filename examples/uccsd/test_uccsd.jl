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

hamiltonian = get_ham()
opt = "LD_LBFGS"
(pool_labels, pool) = uccsd_pool(12, 6)
num_qubits = 12

hf_ind = Int(0b111111000000) + 1
initial_state = zeros(ComplexF64, 2^12)
initial_state[hf_ind] = 1.0 + 0.0im

en_fci = -3.236066279892345

println("Number of operators in the pool is: " * string(length(pool)))

callbacks = Function[
    MaxGradientStopper(1e-8),
    FloorStopper(en_fci, 1e-10),
    ParameterStopper(205),
    EnergyErrorPrinter(en_fci),
    MaxGradientPrinter(),
    ParameterPrinter(),
    OperatorIndexPrinter(pool_labels)
]

result = adapt_vqe_commuting(
    hamiltonian,
    pool,
    num_qubits,
    opt,
    callbacks;
    initial_state = initial_state
)