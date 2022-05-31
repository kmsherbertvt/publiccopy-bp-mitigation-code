function safe_floor(x::Float64, eps=1e-15, delta=1e-8)
    if x <= -delta error("Too negative...") end
    if x <= 0.0
        return eps
    else
        return x
    end
end

function log_mean(x)
	return 10^mean(log10.(safe_floor.(x)))
end

function get_git_id()
    return chop(read(`git rev-parse --short HEAD`, String), tail=2)
end