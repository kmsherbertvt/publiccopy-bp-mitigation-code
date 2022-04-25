using IterTools

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

function SATProblem(A::Array{T, 2}) where T <: Integer
    n, m = size(A)
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
    return H
end

function max_1_3_sat_ham(F::SATProblem)
    H = max_1_2_sat_ham(F)

    H += Operator([Pauli(0, 0, 0, 0)], [F.m/2])

    for i=1:F.n
        hi = -sum(F.A[i,:])
        H += (hi/2) * _get_z_term(i)
    end

    return H
end