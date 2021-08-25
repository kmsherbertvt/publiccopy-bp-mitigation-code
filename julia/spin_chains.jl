# hamiltonians and associated functions for working with quantum spin chains (spin-1/2 for now)

# functions to construct Hamiltonians using the efficient operator class
function xyz_model(L,Jx,Jy,Jz,PBCs)
    ham = Operator([],[])

    for site=1:L-1
        xx_str = "I"^(site-1)*"XX"*"I"^(L-(site+1))
        yy_str = "I"^(site-1)*"YY"*"I"^(L-(site+1))
        zz_str = "I"^(site-1)*"ZZ"*"I"^(L-(site+1))
        push!(ham.paulis,pauli_string_to_pauli(xx_str))
        push!(ham.coeffs,Jx)
        push!(ham.paulis,pauli_string_to_pauli(yy_str))
        push!(ham.coeffs,Jy)
        push!(ham.paulis,pauli_string_to_pauli(zz_str))
        push!(ham.coeffs,Jz)
    end
    if PBCs
        xx_str = "X"*"I"^(L-2)*"X"
        yy_str = "Y"*"I"^(L-2)*"Y"
        zz_str = "Z"*"I"^(L-2)*"Z"
        push!(ham.paulis,pauli_string_to_pauli(xx_str))
        push!(ham.coeffs,Jx)
        push!(ham.paulis,pauli_string_to_pauli(yy_str))
        push!(ham.coeffs,Jy)
        push!(ham.paulis,pauli_string_to_pauli(zz_str))
        push!(ham.coeffs,Jz)
    end
    return ham
end

function xxz_model(L,Jxy,Jz,PBCs)
    return xyz_model(L,Jxy,Jxy,Jz,PBCs)
end

function heisenberg_model(L,J,PBCs)
    return xyz_model(L,J,J,J,PBCs)
end

function magnetic_field(L,Bx,By,Bz)
    if typeof(Bx) == Float64
        Bxvec = fill(Bx,L)
    else
        Bxvec = Bx
    end
    if typeof(By) == Float64
        Byvec = fill(By,L)
    else
        Byvec = By
    end
    if typeof(Bz) == Float64
        Bzvec = fill(Bz,L)
    else
        Bzvec = Bz
    end
    ham = Operator([],[])
    for site=1:L
        Bx_str = "I"^(site-1)*"X"*"I"^(L-site)
        By_str = "I"^(site-1)*"Y"*"I"^(L-site)
        Bz_str = "I"^(site-1)*"Z"*"I"^(L-site)
        if abs(Bxvec[site]) > 1.0e-15
            push!(ham.paulis,pauli_string_to_pauli(Bx_str))
            push!(ham.coeffs,Bxvec[site])
        end
        if abs(Byvec[site]) > 1.0e-15
            push!(ham.paulis,pauli_string_to_pauli(By_str))
            push!(ham.coeffs,Byvec[site])
        end
        if abs(Bzvec[site]) > 1.0e-15
            push!(ham.paulis,pauli_string_to_pauli(Bz_str))
            push!(ham.coeffs,Bzvec[site])
        end
    end
    return ham
end

# functions to construct matrix forms of models - for purposes of exact diagonalization
pdict = Dict('I' => [1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], 'X' => [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im],
             'Y' => [0.0+0.0im 0.0-1.0im; 0.0+1.0im 0.0+0.0im], 'Z' => [1.0+0.0im 0.0+0.0im; 0.0+0.0im -1.0+0.0im])

function paulis_matrix(pstr)
    res = copy(pdict[pstr[1]])
    for ch in pstr[2:length(pstr)]
        res = kron(res,pdict[ch])
    end
    return res
end

function xyz_matrix(L,Jx,Jy,Jz,PBCs)
    ham = zeros(ComplexF64,2^L,2^L)
    for site=1:L-1
        xx_str = "I"^(site-1)*"XX"*"I"^(L-(site+1))
        yy_str = "I"^(site-1)*"YY"*"I"^(L-(site+1))
        zz_str = "I"^(site-1)*"ZZ"*"I"^(L-(site+1))
        ham .= ham .+ Jx*paulis_matrix(xx_str) .+ Jy*paulis_matrix(yy_str) .+ Jz*paulis_matrix(zz_str)
    end
    if PBCs
        xx_str = "X"*"I"^(L-2)*"X"
        yy_str = "Y"*"I"^(L-2)*"Y"
        zz_str = "Z"*"I"^(L-2)*"Z"
        ham .= ham .+ Jx*paulis_matrix(xx_str) .+ Jy*paulis_matrix(yy_str) .+ Jz*paulis_matrix(zz_str)
    end
    return ham
end

function xxz_matrix(L,Jxy,Jz,PBCs)
    return xyz_matrix(L,Jxy,Jxy,Jz,PBCs)
end

function heisenberg_matrix(L,J,PBCs)
    return xyz_matrix(L,J,J,J,PBCs)
end

function magnetic_field_matrix(L,Bx,By,Bz)
    if typeof(Bx) == Float64
        Bxvec = fill(Bx,L)
    else
        Bxvec = Bx
    end
    if typeof(By) == Float64
        Byvec = fill(By,L)
    else
        Byvec = By
    end
    if typeof(Bz) == Float64
        Bzvec = fill(Bz,L)
    else
        Bzvec = Bz
    end
    ham = zeros(ComplexF64,2^L,2^L)
    for site=1:L
        Bx_str = "I"^(site-1)*"X"*"I"^(L-site)
        By_str = "I"^(site-1)*"Y"*"I"^(L-site)
        Bz_str = "I"^(site-1)*"Z"*"I"^(L-site)
        ham .= ham .+ Bxvec[site]*paulis_matrix(Bx_str) .+ Byvec[site]*paulis_matrix(By_str) .+ Bzvec[site]*paulis_matrix(Bz_str)
    end
    return ham
end
