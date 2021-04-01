include("pauli.jl")


mutable struct Operator
    paulis::Array{Pauli,1}
    coeffs::Array{ComplexF64,1}
end

function group_inds_by_eq(A::Array)
    l = length(A)
    grps = Dict{typeof(A[1]), Array{Int64,1}}()
    hit_elts = []
    for i=1:l
        el = A[i]
        if el in hit_elts
            append!(grps[el], i)
        else
            grps[el] = [i]
            append!(hit_elts, el)
        end
    end
    return grps
end

function op_simplify(A::Operator, tol::Float64 = 0.0)
    l = length(A.paulis)

    # Get rid of zero coeff ops
    for i=reverse(1:l)
        if abs(A.coeffs[i]) <= tol
            deleteat!(A.coeffs, i)
            deleteat!(A.paulis, i)
        end
    end

    # Bring all phases out of Paulis
    for i=1:l
        ph = A.paulis[i].phase
        A.paulis[i].phase = 0
        alpha = A.coeffs[i]
        A.coeffs[i] = phase_shift(alpha, ph)
    end

    # Sum duplicate paulis
    dict = group_inds_by_eq(A.paulis)
    new_coeffs = Array{ComplexF64,1}()
    new_paulis = Array{Pauli,1}()

    for (pauli, inds) in d
        c = 0.0 + 0.0im
        for i in inds
            c += A.coeffs[i]
        end
        append!(new_paulis, pauli)
        append!(new_coeffs, c)
    end
    
    A.coeffs = new_coeffs
    A.paulis = new_paulis
end


function op_product(A::Operator, B::Operator)
    new_paulis = Array{Pauli,1}()
    new_coeffs = Array{ComplexF64,1}()

    for i=1:length(A.paulis)
        for j=1:length(B.paulis)
            q = pauli_product(A.paulis[i], B.paulis[j])
            append!(new_paulis, q)
            append!(new_coeffs, A.coeffs[i]*B.coeffs[j])
        end
    end
    return Operator(new_paulis, new_coeffs)
end


function Base.:+(x::Operator, y::Operator)
    new_paulis = vcat(x.paulis, y.paulis)
    new_coeffs = vcat(x.coeffs, y.coeffs)
    return Operator(new_paulis, new_coeffs)
end


function Base.:-(x::Operator, y::Operator)
    new_paulis = vcat(x.paulis, y.paulis)
    new_coeffs = vcat(x.coeffs, -y.coeffs)
    return Operator(new_paulis, new_coeffs)
end


function Base.:*(x::Operator, y::Operator)
    return op_product(x, y)
end


function exp_val(A::Operator, state::Array{ComplexF64,1}, tmp::Array{ComplexF64})
    result = 0.0 + 0.0*im
    for (c,p)=zip(A.coeffs, A.paulis)
        # This convention is fine since P=dagger(P)
        # Equiv to (<psi|P)|psi> = <psi|(P|psi>)
        pauli_mult!(p, state, tmp) # tmp <- P.state
        result += phase_shift(c, p.phase)*dot(tmp, state)
    end
    return result
end