from importlib import abc
from openfermion import (
    jordan_wigner_one_body,
    jordan_wigner_two_body,
    jordan_wigner,
    FermionOperator
)

def one_body(p, q):
    if not (p>q):
        raise ValueError("Must have p>q")
    qubit_op = jordan_wigner(FermionOperator(f'{p}^ {q}') - FermionOperator(f'{q}^ {p}'))
    return qubit_op

one_body(2, 1)

one_body(3, 1)

one_body(5, 2)

one_body(5, 3)

one_body(4, 2)

def two_body(b, a, j, i):
    if not (b>a>j>i):
        raise ValueError("Must have b>a>j>i")
    qubit_op = jordan_wigner(FermionOperator(f'{b}^ {a}^ {j} {i}') - FermionOperator(f'{b} {a} {j}^ {i}^'))
    return qubit_op

two_body(4,3,2,1)

two_body(4,3,2,1)

two_body(5,3,2,1)

two_body(5,4,2,1)

two_body(5,4,3,1)

two_body(5,4,3,2)


