using IterTools

mutable struct Operator
    paulis::Array{Pauli{UInt64},1}
    coeffs::Array{ComplexF64,1}
end

function Operator(paulis::Vector{String}, coeffs::Vector)
    return Operator(map(s->pauli_string_to_pauli(s), paulis), complex(coeffs))
end

function dagger(op::Operator)
    return Operator(op.paulis, conj(op.coeffs))
end

function dagger!(op::Operator)
    conj!(op.coeffs)
end

function print_operator(o::Operator)
    n = num_qubits(o)
    for (p,c) in zip(o.paulis, o.coeffs)
        println("(" * string(c) * ") " * pauli_to_pauli_string(p, n))
    end
end

function group_inds_by_eq(A::Array)
    l = length(A)
    grps = Dict{typeof(A[1]), Array{Int64,1}}()
    hit_elts = []
    for i=1:l
        el = A[i]
        if el in hit_elts
            push!(grps[el], i)
        else
            grps[el] = [i]
            push!(hit_elts, el)
        end
    end
    return grps
end

function op_chop!(A::Operator, tol::Float64 = 0.0)
    l = length(A.paulis)
    if l==0
        return
    end
    # Get rid of zero coeff ops
    for i=reverse(1:l)
        if abs(A.coeffs[i]) <= tol
            deleteat!(A.coeffs, i)
            deleteat!(A.paulis, i)
        end
    end
end

function op_simplify!(A::Operator, tol::Float64 = 0.0)
    op_chop!(A, tol)

    l = length(A.paulis)
    # Bring all phases out of Paulis
    for i=1:l
        ph = A.paulis[i].phase
        A.paulis[i] = Pauli(A.paulis[i].x,A.paulis[i].y,A.paulis[i].z,0)
        alpha = A.coeffs[i]
        A.coeffs[i] = phase_shift(alpha, ph)
    end

    # Sum duplicate paulis
    d = group_inds_by_eq(A.paulis)
    new_coeffs = Array{ComplexF64,1}()
    new_paulis = Array{Pauli,1}()

    for (pauli, inds) in d
        c = 0.0 + 0.0im
        for i in inds
            c += A.coeffs[i]
        end
        push!(new_paulis, pauli)
        push!(new_coeffs, c)
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
            push!(new_paulis, q)
            push!(new_coeffs, A.coeffs[i]*B.coeffs[j])
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

function ham_state_mult!(O::Operator,
                         state::Array{ComplexF64,1},
                         tmp1::Array{ComplexF64},
                         tmp2::Array{ComplexF64}
                         )
    """Note: The resulting 'state' will not be normalized!

    The temporary vector and the state are mutated, but the assignment
    is  `|state> <- H|state>` so that the result is stored in `state`.

    Not yet tested!
    """
    tmp1 .= 0.0 + 0.0im
    for (c,p) in zip(O.coeffs, O.paulis)
        pauli_mult!(p, state, tmp2)
        @. tmp1 += phase_shift(c, p.phase)*tmp2
    end
    state .= tmp1
end


function exp_val(A::Operator, state::Array{ComplexF64,1}, tmp::Array{ComplexF64})
    """This can potentially use `ham_state_mult!`, but will require an extra temporary vector.
    """
    result = 0.0 + 0.0*im
    for (c,p)=zip(A.coeffs, A.paulis)
        # This convention is fine since P=dagger(P)
        # Equiv to (<psi|P)|psi> = <psi|(P|psi>)
        pauli_mult!(p, state, tmp) # tmp <- P.state
        result += phase_shift(c, p.phase)*dot(tmp, state)
    end
    return result
end


function subcommutes(op::Operator)
    for (p,q)=product(op.paulis, op.paulis)
        if !pauli_commute(p,q)
            return false
        end
    end
    return true
end


function commutator(A::Operator, B::Operator, simplify::Bool = true)
    new_paulis = Array{Pauli,1}()
    new_coeffs = Array{ComplexF64, 1}()
    for (p, pc)=zip(A.paulis, A.coeffs)
        for (q, qc)=zip(B.paulis, B.coeffs)
            if !pauli_commute(p, q)
                push!(new_paulis, pauli_product(p, q))
                push!(new_coeffs, +pc*qc)

                push!(new_paulis, pauli_product(q, p))
                push!(new_coeffs, -pc*qc)
            end
        end
    end
    res = Operator(new_paulis, new_coeffs)
    if simplify
        op_simplify!(res)
    end
    return res
end


function matrix_to_operator(A)
    if size(A)[1] != size(A)[2] || length(size(A)) > 2
        error("Invalid shape: $(size(A))")
    else
        N = size(A)[1]
        n = Int(log2(N))
    end

    operator = Operator([], [])

    for axes in Iterators.product(ntuple(i->[0, 1, 2, 3], n)...)
        pauli_mat = pauli_str(reverse([i for i in axes]))
        pauli = pauli_string_to_pauli(reverse([i for i in axes]))
        coeff = tr(pauli_mat*A) / N

        push!(operator.paulis, pauli)
        push!(operator.coeffs, coeff)
    end

    return operator
end


function num_qubits(O::Operator)
    n = 0
    for p in O.paulis
        if num_qubits(p) > n
            n = num_qubits(p)
        end
    end
    return n
end


function operator_to_matrix(O::Operator, n::Int)
    result = zeros(ComplexF64, 2^n, 2^n)
    for (c,p) in zip(O.coeffs, O.paulis)
        result += phase_shift(c, p.phase)*pauli_str(pauli_to_axes(p, n))
    end
    return result
end

function operator_to_matrix(O::Operator)
    return operator_to_matrix(O, num_qubits(O))
end


function diagonal_operator_to_vector(O::Operator)
    n = num_qubits(O)
    result = zeros(ComplexF64, 2^n)
    for (c,p) in zip(O.coeffs, O.paulis)
        if (p.x != 0) | (p.y != 0)
            throw("Found invalid Pauli in supposedly diagonal operator")
        end
        result += phase_shift(c, p.phase)*diag_pauli_str(pauli_to_axes(p, n))
    end
    return result
end