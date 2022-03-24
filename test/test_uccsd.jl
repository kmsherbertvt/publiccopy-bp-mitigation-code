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

@testset "OpenFermion Consistency Check, singles" begin
    p = 5
    q = 2
    n = 5
    op_expected = Operator(["YZZXI", "XZZYI"], [0.0 - 0.5im, 0.0 + 0.5im])
    op_actual = cluster_sing_op(p, q, n)
    @test _op_mat_comp(op_expected, op_actual, n)

    p = 4
    q = 2
    n = 5
    op_expected = Operator(["IYZXI", "IXZYI"], [0.0 - 0.5im, 0.0 + 0.5im])
    op_actual = cluster_sing_op(p, q, n)
    @test _op_mat_comp(op_expected, op_actual, n)

    p = 5
    q = 3
    n = 5
    op_expected = Operator(["YZXII", "XZYII"], [0.0 - 0.5im, 0.0 + 0.5im])
    op_actual = cluster_sing_op(p, q, n)
    @test _op_mat_comp(op_expected, op_actual, n)

    p = 2
    q = 1
    n = 2
    op_expected = Operator(["YX", "XY"], [0.0 - 0.5im, 0.0 + 0.5im])
    op_actual = cluster_sing_op(p, q, n)
    @test _op_mat_comp(op_expected, op_actual, n)

    p = 3
    q = 1
    n = 3
    op_expected = Operator(["YZX", "XZY"], [0.0 - 0.5im, 0.0 + 0.5im])
    op_actual = cluster_sing_op(p, q, n)
    @test _op_mat_comp(op_expected, op_actual, n)
end

@testset "OpenFermion Consistency Check, doubles" begin

end