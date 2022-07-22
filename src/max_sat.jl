using IterTools
using Combinatorics
using StatsBase

struct SATProblem 
    A::Array{T, 2} where T <: Integer
    n
    m
end

function _get_z_term(i::Integer)
    return Operator([Pauli(0, 0, 2^(i-1), 0)], [1.0])
end

function _get_z_term(i::Integer, j::Integer)
    if i == j error("Can't be equal") end
    return Operator([Pauli(0, 0, 2^(i-1) + 2^(j-1), 0)], [1.0])
end

function distinct_columns(A::Array{T, 2}) where T <: Integer
    _, m = size(A)
    for (m1, m2)=product(1:m, 1:m)
        if m1 < m2
            if A[:,m1] == A[:,m2] return false end
        end
    end
    return true
end

function SATProblem(A::Array{T, 2}) where T <: Integer
    n, m = size(A)
    if !distinct_columns(A)
        error("Invalid problem, clauses must be distinct")
    end
    return SATProblem(A, n, m)
end

function max_1_2_sat_ham(F::SATProblem)
    H = Operator([Pauli(0, 0, 0, 0)], [F.m/2])

    for (i,j)=product(1:F.n, 1:F.n)
        if i<j
            Jij = dot(F.A[i,:], F.A[j,:])
            H += (Jij/2) * _get_z_term(i, j)
        else
            continue
        end
    end
    op_chop!(H)
    return H
end

function max_1_3_sat_ham(F::SATProblem)
    H = max_1_2_sat_ham(F)

    H += Operator([Pauli(0, 0, 0, 0)], [F.m/2])

    for i=1:F.n
        hi = -sum(F.A[i,:])
        H += (hi/2) * _get_z_term(i)
    end
    op_chop!(H)
    return H
end

function random_k_sat_instance(n::Integer, m::Integer, k::Integer; rng = _DEFAULT_RNG)
    a = zeros(Int, n)
    for i=1:k a[i] = 1 end
    
    sample_space = unique(collect(permutations(a)))
    vecs = sample(rng, sample_space, m; replace=false)

    A = zeros(Int, n, m)
    for (i,v)=enumerate(vecs)
        A[:,i] .= v
    end

    return A
end

function random_k_sat_hamiltonian(n::Integer, m::Integer, k::Integer; rng = _DEFAULT_RNG)
    A = random_k_sat_instance(n, m, k; rng)
    P = SATProblem(A)
    if k == 2
        return max_1_2_sat_ham(P)
    elseif k == 3
        return max_1_3_sat_ham(P)
    else
        error("Invalid k=$k")
    end
end