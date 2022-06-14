#=
A callback is a function of the step history that
returns true if the outer loop should halt,
and false if it should continue.
=#


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

function MaxGradientPrinter()
    function stopper(hist::ADAPTHistory)
        num_pars = length(hist.opt_pars[end])
        max_grad = string(hist.max_grad[end])
        println("Pars: $num_pars, MaxGrad: $max_grad")
        return false
    end
    return stopper
end

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

function ParameterPrinter()
    function stopper(hist::ADAPTHistory)
        num_pars = length(hist.opt_pars[end])
        opt_pars = string(hist.opt_pars[end])
        println("Pars: $num_pars, OptPars: $opt_pars")
        return false
    end
    return stopper
end


function EnergyErrorPrinter(gse::Float64)
    function stopper(hist::ADAPTHistory)
        num_pars = length(hist.opt_pars[end])
        en_err = hist.energy[end] - gse
        println("Pars: $num_pars, EnErr: $en_err")
        return false
    end
    return stopper
end

function EnergyPrinter()
    function stopper(hist::ADAPTHistory)
        num_pars = length(hist.opt_pars[end])
        en = hist.energy[end]
        println("Pars: $num_pars, En: $en")
        return false
    end
    return stopper
end


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


function DeadlineStopper(max_seconds::Int)
    t_0 = time()
    function stopper(hist::ADAPTHistory)
        if time() - t_0 > max_seconds
            return true
        else
            return false
        end
    end
end