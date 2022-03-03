using Random
using Test
using LinearAlgebra
using AdaptBarren


@testset "Test unpacking" begin
    """ Unpacking a vector refers to taking the original vector (`x`) and
    repeating each element in a new vector some (potentially variable) number
    of times.

    If I have an ansatz where parameters are bound together,
    e.g. `U(t1) V(t2)` where
    `U(t) = e^(-i t A1) e^(-i t A2)`
    `V(t) = e^(-i t B1) e^(-i t B2)`
    then the parameter vector `[t1, t2]` unpacked according to the repetitions
    `[2, 2]` would be `[t1, t1, t2, t2]`.

    On the other hand, unpacking a vector undoes this process, where the aggregation
    function is assumed to be addition. Hence, packing the vector `[t1, t2, t3, t4]`
    according to the reptitions `[2, 2]` would yield the vector `[t1+t2, t3+t4]`.
    """
    k = 5
    x = rand(Float64, k)
    n = [1,3,2,1,3]
    test_xp = unpack_vector(x, n)
    actual_xp = [x[1], x[2], x[2], x[2], x[3], x[3], x[4], x[5], x[5], x[5]]
    @test norm(test_xp - actual_xp) == 0

    k = 1
    x = rand(Float64, k)
    n = [1]
    test_xp = unpack_vector(x, n)
    actual_xp = [x[1]]
    @test norm(test_xp - actual_xp) == 0

    k = 1
    x = rand(Float64, k)
    n = [3]
    test_xp = unpack_vector(x, n)
    actual_xp = [x[1], x[1], x[1]]
    @test norm(test_xp - actual_xp) == 0

    k = 5
    x = rand(Float64, k)
    n = [2,3,2,1,3]
    test_xp = unpack_vector(x, n)
    actual_xp = [x[1], x[1], x[2], x[2], x[2], x[3], x[3], x[4], x[5], x[5], x[5]]
    @test norm(test_xp - actual_xp) == 0

    k = 5
    x = rand(Float64, k)
    n = [2,3,2,1,1]
    test_xp = unpack_vector(x, n)
    actual_xp = [x[1], x[1], x[2], x[2], x[2], x[3], x[3], x[4], x[5]]
    @test norm(test_xp - actual_xp) == 0
end

@testset "Test packing" begin
    n = [1,3,2,1,3]
    N = sum(n)
    yp = rand(Float64, N)
    actual_y = [yp[1], yp[2]+yp[3]+yp[4], yp[5]+yp[6], yp[7], yp[8]+yp[9]+yp[10]]
    test_y = pack_vector(yp, n)
    @test norm(test_y - actual_y) <= 1e-10

    n = [2,3,1]
    N = sum(n)
    yp = rand(Float64, N)
    actual_y = [yp[1]+yp[2], yp[3]+yp[4]+yp[5], yp[6]]
    test_y = pack_vector(yp, n)
    @test norm(test_y - actual_y) <= 1e-10

    n = [1,3,2]
    N = sum(n)
    yp = rand(Float64, N)
    actual_y = [yp[1], yp[2]+yp[3]+yp[4], yp[5]+yp[6]]
    test_y = pack_vector(yp, n)
    @test norm(test_y - actual_y) <= 1e-10

    n = [1]
    N = sum(n)
    yp = rand(Float64, N)
    actual_y = [yp[1]]
    test_y = pack_vector(yp, n)
    @test norm(test_y - actual_y) <= 1e-10

    n = [3]
    N = sum(n)
    yp = rand(Float64, N)
    actual_y = [yp[1]+yp[2]+yp[3]]
    test_y = pack_vector(yp, n)
    @test norm(test_y - actual_y) <= 1e-10
end