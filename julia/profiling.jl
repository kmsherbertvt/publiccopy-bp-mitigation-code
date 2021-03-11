using Random
import LinearAlgebra: norm
using Test
using BenchmarkTools

include("fast_pauli_vec_mult.jl")
include("simulator.jl")

n = 10
d = 100

axes = [[rand(0:3) for j=1:n] for j=1:d]
theta = [pi*rand() for j=1:d]

# Compiling run
result = zeros(ComplexF64, 2^n)
tmp1 = zeros(ComplexF64, 2^n)
tmp2 = zeros(ComplexF64, 2^n)
pauli_ansatz_new!(axes, theta, result, tmp1, tmp2)

# Actual run
result = zeros(ComplexF64, 2^n)
tmp1 = zeros(ComplexF64, 2^n)
tmp2 = zeros(ComplexF64, 2^n)
@btime pauli_ansatz_new!(axes, theta, result, tmp1, tmp2)