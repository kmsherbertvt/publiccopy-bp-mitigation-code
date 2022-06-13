using AdaptBarren
using BenchmarkTools

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

function get_ham(filename, num_qubits)
    operator = Operator([], [])
    open(filename) do f
        for line in readlines(f)
            line = replace(line, "["=>"")
            line = replace(line, "]"=>"")
            sp = split(line, " ")
            coeff = ComplexF64(parse(Float64, sp[1]))
            terms = sp[2:end]
            pauli = terms_to_pauli(terms, num_qubits)
            push!(operator.paulis, pauli)
            push!(operator.coeffs, coeff)
        end
    end
    return operator
end

opt = "LD_LBFGS"

# H_6 at 1A
#num_qubits = 12
#hamiltonian = get_ham("h6_1A.ham", num_qubits)
#(pool_labels, pool) = uccsd_pool(num_qubits, 6)
#hf_ind = Int(0b111111000000) + 1
#en_fci = -3.236066279892345

# N_2
num_qubits = 20
hamiltonian = get_ham("n2.ham", num_qubits)
(pool_labels, pool) = uccsd_pool(num_qubits, 10)
hf_ind = Int(0b11111111110000000000) + 1
en_fci = -107.657038063455


# H_8
num_qubits = 16
hamiltonian = get_ham("h8.ham", num_qubits)
(pool_labels, pool) = uccsd_pool(num_qubits, 8)
hf_ind = Int(0b1111111100000000) + 1
en_fci = -4.307571601998721

# H_4
num_qubits = 8
hamiltonian = get_ham("h4.ham", num_qubits)
(pool_labels, pool) = uccsd_pool(num_qubits, 4)
hf_ind = Int(0b11110000) + 1
en_fci = -2.1663874486347625

initial_state = zeros(ComplexF64, 2^num_qubits)
initial_state[hf_ind] = 1.0 + 0.0im
println("Number of operators in the pool is: " * string(length(pool)))

callbacks = Function[
    MaxGradientStopper(1e-8),
    FloorStopper(en_fci, 1e-10),
    ParameterStopper(300),
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
