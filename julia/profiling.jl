using Random
import LinearAlgebra: norm
using Test
using BenchmarkTools
using StatProfilerHTML

using AdaptBarren

n = 20
d = 100

axes = [[rand(0:3) for j=1:n] for j=1:d]
theta = [pi*rand() for j=1:d]

# Compiling run
result = zeros(ComplexF64, 2^n)
tmp1 = zeros(ComplexF64, 2^n)
pauli_ansatz!(axes, theta, result, tmp1)

# Actual run
result = zeros(ComplexF64, 2^n)
tmp1 = zeros(ComplexF64, 2^n)
@btime pauli_ansatz!(axes, theta, result, tmp1)