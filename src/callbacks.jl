#=
A callback is a function of the step history that
returns true if the outer loop should halt,
and false if it should continue.
=#

"""
    MaxGradientStopper(eps::Float64 = 1e-8)

Halts the algorithm if the maximum gradient falls below `eps`.
"""
function MaxGradientStopper(eps::Float64 = 1e-8)
    function stopper(hist::ADAPTHistory)
        if abs(hist.max_grad[end]) < eps
            return true
        else
            return false
        end
    end
    return stopper
end

"""
    MaxGradientPrinter()

Print the maximum gradient at each layer.
"""
function MaxGradientPrinter()
    function stopper(hist::ADAPTHistory)
        num_pars = length(hist.opt_pars[end])
        max_grad = string(hist.max_grad[end])
        println("Pars: $num_pars, MaxGrad: $max_grad")
        return false
    end
    return stopper
end

"""
    OperatorIndexPrinter(formatted_ops = nothing)

At each layer, indicate the operator chosen by the ADAPT criterion.

If `formatted_ops === nothing`, the index of each operator as it exists
in the pool will be printed. If a list of strings is given, the string
of that index will instead be printed.
"""
function OperatorIndexPrinter(formatted_ops = nothing)
    function stopper(hist::ADAPTHistory)
        num_pars = length(hist.opt_pars[end])
        max_grad_ind = hist.max_grad_ind[end]
        if formatted_ops === nothing
            printed_op = max_grad_ind
        else
            printed_op = formatted_ops[max_grad_ind]
        end
        println("Pars: $num_pars, MaxGradOp: $printed_op")
        return false
    end
    return stopper
end

"""
    ParameterPrinter()

Prints the optimal parameters at each iteration.
"""
function ParameterPrinter()
    function stopper(hist::ADAPTHistory)
        num_pars = length(hist.opt_pars[end])
        opt_pars = string(hist.opt_pars[end])
        println("Pars: $num_pars, OptPars: $opt_pars")
        return false
    end
    return stopper
end


"""
    EnergyErrorPrinter(gse::Float64)

Print the difference between the energy at the last iteration
and the ground state energy, given as the argument `gse`.
"""
function EnergyErrorPrinter(gse::Float64)
    function stopper(hist::ADAPTHistory)
        num_pars = length(hist.opt_pars[end])
        en_err = hist.energy[end] - gse
        println("Pars: $num_pars, EnErr: $en_err")
        return false
    end
    return stopper
end

"""
    EnergyPrinter()

Print the energy returned at the last iteration.
"""
function EnergyPrinter()
    function stopper(hist::ADAPTHistory)
        num_pars = length(hist.opt_pars[end])
        en = hist.energy[end]
        println("Pars: $num_pars, En: $en")
        return false
    end
    return stopper
end


"""
    ParameterStopper(max_pars::Int64)

Halt the algorithm if the maximum number of parameters is exceeded.
"""
function ParameterStopper(max_pars::Int64)
    function stopper(hist::ADAPTHistory)
        if length(hist.opt_pars) >= max_pars + 1
            return true
        else
            return false
        end
    end
    return stopper
end

"""
    DeltaYStopper(delta::Float64 = 1e-8, n_best::Int64 = 5)

Halt the algorithm if the `n_best` many energies are all within `delta` of each other.

This is supposed to indicate that the algorithm has stalled in performance.
"""
function DeltaYStopper(delta::Float64 = 1e-8, n_best::Int64 = 5)
    function stopper(hist::ADAPTHistory)
        en_sort = sort(hist.energy)
        if length(en_sort) <= n_best
            return false
        end
        best = en_sort[1]
        worst = en_sort[n_best]
        if worst - best < delta
            return true
        else
            return false
        end
    end
    return stopper
end


"""
    FloorStopper(floor::Float64, delta::Float64 = 1e-8)

Halt the algorithm if the current energy falls within `delta` of `floor`.
"""
function FloorStopper(floor::Float64, delta::Float64 = 1e-8)
    function stopper(hist::ADAPTHistory)
        if abs(hist.energy[end] - floor) <= delta
            return true
        else
            return false
        end
    end
    return stopper
end
