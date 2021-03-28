struct ADAPTHistory
    energy::Array{Float64,1}
    opt_pars::Array{Array{Float64,1},1}
    gradients::Array{Array{Float64,1},1}
    max_grad::Array{Float64,1}
    op_indices::Array{Array{Int64,1},1}
    halt_reason::string
end


struct Operator
    coeffs::Array{Float64,1}
    paulis::Array{Array{Int64,1},1}
end