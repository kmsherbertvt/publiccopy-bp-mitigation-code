using Test
using AdaptBarren
using LinearAlgebra
using IterTools

@testset "Test that terms in ops commute, doubles" begin
    for n=4:6
        for (a,b,i,j)=product(1:n,1:n,1:n,1:n)
            if !(b>a>j>i) continue end
                @test subcommutes(cluster_doub_op(a, b, i, j, n))
        end
    end
end

@testset "Test that terms in ops commute, singles" begin
    for n=4:6
        for (p,q)=product(1:n,1:n)
            if p<=q continue end
            op = cluster_sing_op(p, q, n)
            @test subcommutes(op)
        end
    end
end

function _op_mat_comp(a, b, n; eps=1e-5)
    m1 = operator_to_matrix(a, n)
    m2 = operator_to_matrix(b, n)
    #display(m1)
    #display(m2)
    return norm(m1 - m2) < eps
end

@testset "OpenFermion Consistency Check" begin
    p = 5
    q = 2
    n = 5
    op_expected = Operator([pauli_string_to_pauli("YZZXI"), pauli_string_to_pauli("XZZYI")], [0.0 - 0.5im, 0.0 + 0.5im])
    op_actual = cluster_sing_op(p, q, n)
    @test _op_mat_comp(op_expected, op_actual, n)

    p = 4
    q = 2
    n = 5
    op_expected = Operator([pauli_string_to_pauli("IYZXI"), pauli_string_to_pauli("IXZYI")], [0.0 - 0.5im, 0.0 + 0.5im])
    op_actual = cluster_sing_op(p, q, n)
    @test _op_mat_comp(op_expected, op_actual, n)

    p = 5
    q = 3
    n = 5
    op_expected = Operator([pauli_string_to_pauli("YZXII"), pauli_string_to_pauli("XZYII")], [0.0 - 0.5im, 0.0 + 0.5im])
    op_actual = cluster_sing_op(p, q, n)
    @test _op_mat_comp(op_expected, op_actual, n)

    p = 2
    q = 1
    n = 2
    op_expected = Operator([pauli_string_to_pauli("YX"), pauli_string_to_pauli("XY")], [0.0 - 0.5im, 0.0 + 0.5im])
    op_actual = cluster_sing_op(p, q, n)
    @test _op_mat_comp(op_expected, op_actual, n)

    p = 3
    q = 1
    n = 3
    op_expected = Operator([pauli_string_to_pauli("YZX"), pauli_string_to_pauli("XZY")], [0.0 - 0.5im, 0.0 + 0.5im])
    op_actual = cluster_sing_op(p, q, n)
    @test _op_mat_comp(op_expected, op_actual, n)
end