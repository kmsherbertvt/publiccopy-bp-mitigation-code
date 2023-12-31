using IterTools

"""
    Operator(paulis::Array{Pauli{UInt64},1}, coeffs::Array{ComplexF64,1})

Construct an operator which is a weighted sum of Pauli strings.
"""
mutable struct Operator
    paulis::Array{Pauli{UInt64},1}
    coeffs::Array{ComplexF64,1}
end

"""
    Operator(p::Pauli)

Create an operator that is a single Pauli string with weight `1.0`.
"""
function Operator(p::Pauli)
    return Operator([p], [1.0])
end

function Operator(o::Operator)
    return o
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

"""
    op_chop!(A::Operator, tol::Float64 = 0.0)

Remove terms inplace from an operator whose coefficients have coefficients
with magnitude less than `tol`.
"""
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


"""
    op_product(A::Operator, B::Operator)

Computes the product of operators `A` and `B`.
"""
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


"""
    +(x::Operator, y::Operator)

Computes `x+y` for operators `x` and `y`.
"""
function Base.:+(x::Operator, y::Operator)
    new_paulis = vcat(x.paulis, y.paulis)
    new_coeffs = vcat(x.coeffs, y.coeffs)
    return Operator(new_paulis, new_coeffs)
end


"""
    -(x::Operator, y::Operator)

Computes `x-y` for operators `x` and `y`.
"""
function Base.:-(x::Operator, y::Operator)
    new_paulis = vcat(x.paulis, y.paulis)
    new_coeffs = vcat(x.coeffs, -y.coeffs)
    return Operator(new_paulis, new_coeffs)
end


"""
    *(x::Operator, y::Operator)

Computes `x*y` for operators `x` and `y`.
"""
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


"""
    exp_val(A::Operator, state::Array{ComplexF64,1}, tmp::Array{ComplexF64})

Compute the expectation value `<state|A|state>`.

The `tmp` array is mutated, and `state` is left unchanged.
"""
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


"""
    commutator(A::Operator, B::Operator, simplify::Bool = true)

Compute the commutator `[A,B]` for operators `A` and `B`.
"""
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


"""
    matrix_to_operator(A::Array{ComplexF64,1})

Return the matrix `A` represented as an operator in the Pauli basis.
"""
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


"""
    num_qubits(O::Operator)

Compute the number of qubits on which `O` acts non-trivially, including
qubits with lower index than the maximum.

For example, a Pauli with Pauli string `XYI` would act on `3` qubits,
but `IYX` would act on only `2` qubits.
"""
function num_qubits(O::Operator)
    n = 0
    for p in O.paulis
        if num_qubits(p) > n
            n = num_qubits(p)
        end
    end
    return n
end


"""
    operator_to_matrix(O::Operator)

Compute the matrix representation of a given operator.
"""
function operator_to_matrix(O::Operator)
    n = num_qubits(O)
    result = zeros(ComplexF64, 2^n, 2^n)
    for (c,p) in zip(O.coeffs, O.paulis)
        result += phase_shift(c, p.phase)*pauli_str(pauli_to_axes(p, n))
    end
    return result
end


"""
    diagonal_operator_to_vector(O::Operator)

Return the diagonal of an operator which is diagonal in the computational
basis.
"""
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


"""
    get_ground_state(h::Operator)

Compute the smallest eigenvalue of an operator.

Currently only diagonal operators are supported.
"""
function get_ground_state(h::Operator)
    try
        h_vec = real(diagonal_operator_to_vector(h))
        ground_state_energy = minimum(h_vec)

        gs_ind = argmin(h_vec)
        gs_vec = zeros(ComplexF64, length(h_vec))
        gs_vec[gs_ind] = 1.0 + 0.0im

        sort!(h_vec)

        return ground_state_energy
    catch err
        error("off diagonal not implemented yet")
    end
end


"""
    get_energy_gap(h::Operator)

Compute the difference beteween the largest and smallest
eigenvalues of a given Hamiltonian.
"""
function get_energy_gap(h::Operator)
    try
        h_vec = real(diagonal_operator_to_vector(h))
        gap = maximum(h_vec) - minimum(h_vec)
        return gap
    catch err
        error("off diagonal not implemented yet")
    end
end


"""
    ground_state_overlap(hamiltonian::Vector{Float64}, state::Vector{ComplexF64}, degen_tol = 1e-10)

Compute the overlap between the given state and the ground state of a given operator.

Note that if the given operator has degenerate ground states, the overlaps are summed over.
Numerically, all states with energy `E` such that `|E - E_0| <= degen_tol` are said to be degenerate.
"""
function ground_state_overlap(hamiltonian::Vector{Float64}, state::Vector{ComplexF64}, degen_tol = 1e-10)
    gse = minimum(hamiltonian)
    inds = findall(e -> abs(e - gse) <= degen_tol, hamiltonian)
    return sum(abs(state[i])^2 for i in inds)
end


"""
    clump_degenerate_values(v::Vector{T}, degen_tol = 1e-10) where T<:Real

Given a sorted vector `v`, produce a vector of vectors where the resulting elements
are all within `degen_tol` of each other in absolute value.

# Examples
Given the vector `[0, 1, 1, 2, 2, 2, 3]`, the clumped output
should be `[[0], [1, 1], [2, 2, 2], [3]]`.
"""
function clump_degenerate_values(v::Vector{T}, degen_tol = 1e-10) where T<:Real
    # this assumes x is sorted in ascending order
    v_copy = copy(v)
    result = [[]]
    x_old = v_copy[1]
    while length(v_copy) > 0
        x_new = popfirst!(v_copy)
        if abs(x_new - x_old) <= degen_tol
            push!(result[end], x_new)
        else
            push!(result, [x_new])
            x_old = x_new
        end
    end
    return result
end