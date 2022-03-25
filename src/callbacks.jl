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
        println("Maximum gradient is: " * string(hist.max_grad))
        return false
    end
    return stopper
end

function OperatorIndexPrinter()
    function stopper(hist::ADAPTHistory)
        println("Maximum gradient has index: " * string(hist.max_grad_ind))
        return false
    end
    return stopper
end

function ParameterPrinter()
    function stopper(hist::ADAPTHistory)
        println("Optimal parameters are: " * string(hist.opt_pars))
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
