include("pauli.jl")

#=
A callback is a function of the step history that
returns true if the outer loop should halt,
and false if it should continue.
=#


function MaxGradientStopper(eps::Float64 = 1e-8)
    function stopper(hist::ADAPTHistory)
        if abs(hist.max_grad[-1]) < eps
            return true
        else
            return false
        end
    end
    return stopper
end


function ParameterStopper(max_pars::Int64)
    function stopper(hist::ADAPTHistory)
        if length(hist.opt_pars[-1]) >= max_pars
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
        if len(en_sort) <= n_best
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
        if abs(hist.energy[-1] - floor) <= delta
            return true
        else
            return false
        end
    end
    return stopper
end