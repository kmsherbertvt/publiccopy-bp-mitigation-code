import LinearAlgebra: reshape


function moveaxis(a, source::Int64, destination::Int64)
    dims = [i for i in 1:ndims(a)]
    e1 = dims[source]
    e2 = dims[destination]
    dims[source] = e2
    dims[destination] = e1
    return permutedims(a, tuple(dims...))
end


function _reshape(a, dims)
    return reshape(a, dims)
end


function unfold(tens, mode::Int64, dims)
    new_shape = (dims[mode], prod(dims)÷dims[mode])
    if mode == 1
        return _reshape(tens, new_shape)
    else
        return _reshape(moveaxis(tens, mode, 1), new_shape)
    end
end


function refold(vec, mode::Int64, dims)
    if mode == 1
        return _reshape(vec, dims)
    else
        temp_dims = [dims[mode]]
        for (m,d) = enumerate(dims)
            if m != mode
                push!(temp_dims, d)
            end
        end
        temp_dims = tuple(temp_dims...)
        tens = _reshape(vec, temp_dims)
        return moveaxis(tens, 1, mode)
    end
end


function kron_vec_prod(mats, v)
    dims = tuple([size(m)[1] for m in mats]...)
    v = _reshape(v, dims)
    for (i, a)=enumerate(mats)
        v = refold(a * unfold(v, i, dims), i, dims)
    end
    v = [(v...)...]
end


function kron_vec_bf(mats, v)
    return reduce(kron, mats) * v
end


#=
Testing
=#

using Random
import LinearAlgebra: norm
using Test


function test_kron_prod(As, v)
    return kron_vec_prod(As, v) ≈ kron_vec_bf(As, v)
end


#@testset "End to end test" begin
#    # Setup
#    for num_qubits=2:5
#        # Test structured matrix
#        _dims = tuple([2 for i in 1:num_qubits]...)
#        As = []
#        for i=1:num_qubits
#            mat = [i i+1; i+2 i+3]
#            push!(As, mat)
#        end
#        v = Base.OneTo(prod(_dims))
#        @test test_kron_prod(As, v)
#
#        # Test Random
#        As = [rand(ComplexF64, (2, 2)) for i=1:num_qubits]
#        v = rand(ComplexF64, 2^num_qubits)
#        @test test_kron_prod(As, v)
#    end
#end


@testset "Test fold-unfold action" begin
    for num_qubits=2:5
        for mode=1:num_qubits
            dims = tuple([2 for i=1:num_qubits]...)
            vec = rand(ComplexF64, 2^num_qubits)
            a = unfold(refold(vec, mode, dims), mode, dims)
            b = vec
            @test size(a) === size(b)
            @test a ≈ b
        end
    end
end


# Reduce testing
#A = reduce(kron, As)
#println(A)
#
#_shape = tuple(_dims..., _dims...)
#println(_shape)
#
#A = _reshape(A, _shape)
#println(A)