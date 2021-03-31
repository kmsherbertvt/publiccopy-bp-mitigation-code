using NLopt

include("vqe.jl")
include("operator.jl")
include("pauli.jl")


ham = Operator(
    [
        Pauli(0, 0, 3, 0),
        Pauli(0, 0, 6, 0),
        Pauli(0, 0, 11, 0),
        Pauli(0, 0, 1, 2),
    ],
    [
        0.1,
        1.2,
        8.9,
        3.6
    ]
)

ansatz = [
    Pauli(15, 0, 0, 0),
    Pauli(0, 13, 0, 0),
    Pauli(0, 0, 5, 0),
    Pauli(0, 0, 0, 11)
]

opt = Opt(:LN_COBYLA, 4)

initial_point = [0.1, 0.2, 0.3, 0.4]

num_qubits = 4

result = VQE(ham, ansatz, opt, initial_point, num_qubits)

println(result)