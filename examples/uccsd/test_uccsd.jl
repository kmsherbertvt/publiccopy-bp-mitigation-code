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

function get_ham(filename)
    operator = Operator([], [])
    n = 12
    open(filename) do f
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

opt = "LD_LBFGS"

hamiltonian = get_ham("h6_1A.ham")
num_qubits = 12
(pool_labels, pool) = uccsd_pool(num_qubits, 6)
hf_ind = Int(0b111111000000) + 1
en_fci = -3.236066279892345

initial_state = zeros(ComplexF64, 2^12)
initial_state[hf_ind] = 1.0 + 0.0im
println("Number of operators in the pool is: " * string(length(pool)))

callbacks = Function[
    MaxGradientStopper(1e-8),
    FloorStopper(en_fci, 1e-10),
    ParameterStopper(200),
    EnergyErrorPrinter(en_fci),
    MaxGradientPrinter(),
    ParameterPrinter(),
    OperatorIndexPrinter(pool_labels)
]

result = @btime adapt_vqe_commuting(
    hamiltonian,
    pool,
    num_qubits,
    opt,
    callbacks;
    initial_state = initial_state
)
