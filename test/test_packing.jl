using Random
using Test
using LinearAlgebra
using AdaptBarren

function unpack_vector(x::Vector, n::Vector)
    if length(x) != length(n) error("Must be same length") end
    xp = []
    for (ni, xi) in zip(n,x)
        append!(xp, repeat([xi], ni))
    end
    return xp
end

function pack_vector(y_input::Vector, n::Vector)
    yp = copy(y_input)
    if length(yp) != sum(n) error("Cannot unpack") end
    y = []
    for ni in reverse(n)
        yi = sum([pop!(yp) for i=1:ni])
        push!(y, yi)
    end
    reverse!(y)
    return y
end

@testset "Test unpacking" begin
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