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
    """ This is a consistency check with the following `openfermion`
    function:
        ```
        def one_body(p, q):
            if not (p>q):
                raise ValueError("Must have p>q")
            qubit_op = jordan_wigner(FermionOperator(f'{p}^ {q}') - FermionOperator(f'{q}^ {p}'))
            return qubit_op
        ```
    """
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
    """ This is a consistency check with the following `openfermion`
    function:
        ```
        def two_body(b, a, j, i):
            if not (b>a>j>i):
                raise ValueError("Must have b>a>j>i")
            qubit_op = jordan_wigner(FermionOperator(f'{b}^ {a}^ {j} {i}') - FermionOperator(f'{b} {a} {j}^ {i}^'))
            return qubit_op
        ```
    """

    n = 4
    (b,a,j,i) = (4,3,2,1)
    op_expected = Operator(["YXXX", "XYXX", "XXYX", "YYYX", "XXXY", "YYXY", "YXYY", "XYYY"], [+1, +1, -1, +1, -1, +1, -1, -1]*(1.0im/8.0))
    op_actual = cluster_doub_op(a, b, i, j, n)
    @test _op_mat_comp(op_expected, op_actual, n)

    n = 5
    (b,a,j,i) = (5,3,2,1)
    op_expected = Operator(["YZXXX", "XZYXX", "XZXYX", "YZYYX", "XZXXY", "YZYXY", "YZXYY", "XZYYY"], [+1, +1, -1, +1, -1, +1, -1, -1]*(1.0im/8.0))
    op_actual = cluster_doub_op(a, b, i, j, n)
    @test _op_mat_comp(op_expected, op_actual, n)

    n = 5
    (b,a,j,i) = (5,4,2,1)
    op_expected = Operator(["YXIXX", "XYIXX", "XXIYX", "YYIYX", "XXIXY", "YYIXY", "YXIYY", "XYIYY"], [+1, +1, -1, +1, -1, +1, -1, -1]*(1.0im/8.0))
    op_actual = cluster_doub_op(a, b, i, j, n)
    @test _op_mat_comp(op_expected, op_actual, n)

    n = 5
    (b,a,j,i) = (5,4,3,1)
    op_expected = Operator(["YXXZX", "XYXZX", "XXYZX", "YYYZX", "XXXZY", "YYXZY", "YXYZY", "XYYZY"], [+1, +1, -1, +1, -1, +1, -1, -1]*(1.0im/8.0))
    op_actual = cluster_doub_op(a, b, i, j, n)
    @test _op_mat_comp(op_expected, op_actual, n)

    n = 5
    (b,a,j,i) = (5,4,3,2)
    op_expected = Operator(["YXXXI", "XYXXI", "XXYXI", "YYYXI", "XXXYI", "YYXYI", "YXYYI", "XYYYI"], [+1, +1, -1, +1, -1, +1, -1, -1]*(1.0im/8.0))
    op_actual = cluster_doub_op(a, b, i, j, n)
    @test _op_mat_comp(op_expected, op_actual, n)
end