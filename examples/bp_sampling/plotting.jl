using AdaptBarren
using DataFrames
using Plots
using StatsBase
using IterTools: product
using LinearAlgebra: norm
using StatsPlots
using CSV
using Statistics
gr();

# Hyperparameters
FIGS_DIR = "./figs"
DATA_DIR = "./data"
DATA_SUFFIX = "csv"
max_num_qubits = 6
qubit_range = 4:2:max_num_qubits
gid = "CID:$(get_git_id())"

function diff_opt_pars(opt_pars:: Vector{Vector{Float64}})
    res = Vector{Float64}()
    for i=1:(length(opt_pars)-1)
        if length(opt_pars[i]) == 0 continue end
        v1 = opt_pars[i]
        v2 = opt_pars[i+1][1:length(v1)]
        push!(res, norm(v1 .- v2))
    end
    return res
end

# Data Import
t_0 = time()
println("Loading data..."); flush(stdout)

df_en = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_en_$(n).$(DATA_SUFFIX)")) for n=qubit_range]...)
df_grad = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_grad_$(n).$(DATA_SUFFIX)")) for n=qubit_range]...)
df_en_errs = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_en_errs_$(n).$(DATA_SUFFIX)")) for n=qubit_range]...)
df_rel_en_errs = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_rel_en_errs_$(n).$(DATA_SUFFIX)")) for n=qubit_range]...)

df_ball_en = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_ball_en_$(n).$(DATA_SUFFIX)")) for n=qubit_range]...)
df_ball_grad = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_ball_grad_$(n).$(DATA_SUFFIX)")) for n=qubit_range]...)
df_ball_en_errs = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_ball_en_errs_$(n).$(DATA_SUFFIX)")) for n=qubit_range]...)
df_ball_rel_en_errs = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_ball_rel_en_errs_$(n).$(DATA_SUFFIX)")) for n=qubit_range]...)

df_dict_ball = Dict(
    "en" => df_ball_en,
    "grad" => df_ball_grad,
    "en_errs" => df_ball_en_errs,
    "rel_en_errs" => df_ball_rel_en_errs,
)
df_dict = Dict(
    "en" => df_en,
    "grad" => df_grad,
    "en_errs" => df_en_errs,
    "rel_en_errs" => df_rel_en_errs,
)

df_res = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_res_$(n).$(DATA_SUFFIX)")) for n=qubit_range]...)
cols_to_eval = [ :energies, :max_grads, :opt_pars, :energy_errors, :approx_ratio, :relative_error, :ground_state_overlaps, ]
println("Took $(time()-t_0) seconds"); flush(stdout)
println("Transforming data..."); flush(stdout)
t_0 = time()
transform!(df_res, cols_to_eval .=> ByRow(x -> eval(Meta.parse(x))) .=> cols_to_eval)
transform!(df_res, :energies .=> ByRow(x -> length(x)) .=> :final_depth)
transform!(df_res, :opt_pars .=> ByRow(x -> diff_opt_pars(x)) .=> :opt_pars_diffs)
println("Took $(time()-t_0) seconds"); flush(stdout)

# Utility Functions
function mean_on(df, grp_cols, mean_col; aggr_fn = mean)
    gdf = groupby(df, grp_cols)
    return combine(gdf, mean_col => (x -> aggr_fn(x)) => mean_col)
end

function fig_out(filename)
    savefig("$(FIGS_DIR)/$(filename).pdf")
end

function flatten_and_count(df, var, counter)
    _df = copy(df)
    transform!(_df, var .=> ByRow(x -> collect(enumerate(x))) .=> var)
    _df = flatten(_df, var)
    transform!(_df, var .=> ByRow(x -> x[1]) .=> counter)
    transform!(_df, var .=> ByRow(x -> x[2]) .=> var)
    return _df
end

# Main Plots

function plot_1(aggr, sampling, fn, select_instance = missing)
    """ 4 plots, x axis is number of layers, y axis is var grad, hue num qubits """
    plot_names = unique(df_en[!, :method])

    if sampling === "ball"
        _df_dict = df_dict_ball
        x_axis = :rad
        x_axis_label = "Radius"
    elseif sampling === "layers"
        _df_dict = df_dict
        x_axis = :d
        x_axis_label = "Layers"
    else
        error("Invalid method: $sampling")
    end


    if fn === "grad"
        _df_fn = _df_dict["grad"]
        st = "Grad"
        st_long = "gradient"
        symb = :grad
        use_safe_floor = false
        use_abs = true
        yaxis_scale = :linear
    elseif fn === "en"
        _df_fn = _df_dict["en"]
        st = "En"
        st_long = "energy"
        symb = :en
        use_safe_floor = false
        use_abs = false
        yaxis_scale = :linear
    elseif fn === "en_errs"
        _df_fn = _df_dict["en_errs"]
        st = "En Errs"
        st_long = "energy_errors"
        symb = :en_err
        use_safe_floor = true
        use_abs = true
        yaxis_scale = :linear
    elseif fn === "rel_en_errs"
        _df_fn = _df_dict["rel_en_errs"]
        st = "Rel En Errs"
        st_long = "relative_energy_errors"
        symb = :rel_en_err
        use_safe_floor = true
        use_abs = true
        yaxis_scale = :linear
    else
        error("invalid fn")
    end

    if aggr === "var"
        filename = "convergence_$(sampling)_$(st_long)_variance"
        aggr_fn = var
        ylabel = "Var($st)"
    elseif aggr === "mean"
        filename = "convergence_$(sampling)_$(st_long)_mean"
        _if_abs = if use_abs abs else identity end
        _if_sfl = if use_safe_floor safe_floor else identity end
        #aggr_fn = x -> mean(x)
        aggr_fn = x -> mean(_if_abs.(_if_sfl.(x)))
        ylabel = "Mean($st)"
    else
        error("Invalid aggr=$aggr")
    end

    if select_instance !== missing
        filename = "inst_" * filename
    end

    if select_instance !== missing
        _n = select_instance["n"]
        _seed = select_instance["seed"]
        _df_fn = filter(:n => ==(_n), filter(:seed => ==(_seed), copy(_df_fn)))
        filename = filename * "_n$(_n)seed$(_seed)"
    end

    ymin = minimum(_df_fn[!, symb])
    ymax = maximum(_df_fn[!, symb])

    plots = Dict()
    for nm in plot_names
        _df = mean_on(filter(:method => ==(nm), _df_fn), [:n, x_axis], symb; aggr_fn = aggr_fn)
        plots[nm] = @df _df plot(cols(x_axis), cols(symb), group=:n, 
            yaxis=yaxis_scale, 
            title=replace(uppercase(nm), "_" => " "),
            xlabel=x_axis_label,
            ylabel=ylabel,
            #ylim=(ymin,ymax),
            #top_margin    = 10Plots.mm,
            #right_margin  = 5Plots.mm,
            #left_margin   = 20Plots.mm,
            #bottom_margin = 10Plots.mm,
            )
    end

    main_plot = plot(collect(values(plots))...,
        top_margin    = 5Plots.mm,
        right_margin  = 10Plots.mm,
        left_margin   = 15Plots.mm,
        bottom_margin = 10Plots.mm
    )
    fig_out(filename)
    return main_plot
end

function plot_2(figure_of_merit;
        leg_pos = :topright, 
        yaxis_scale = :log, 
        select_instance = missing,
        safe_floor_agg = false
        )
    """ 3 plots comparing layer-wise performance,
    x axis is number of layers, y axis is mean figure of merit, hue num qubits """
    plot_names = ["qaoa", "adapt_qaoa_2l", "adapt_vqe_2l"]

    filename = "convergence_$(string(figure_of_merit))"
    if select_instance !== missing
        filename = "inst_" * filename
    end
    if safe_floor_agg
        aggr_fn = x -> mean(safe_floor.(x))
    else
        aggr_fn = x -> mean(x)
    end
    ylabel = string(figure_of_merit)

    if select_instance !== missing
        _n = select_instance["n"]
        _seed = select_instance["seed"]
        _df = filter(:n => ==(_n), filter(:seed => ==(_seed), copy(df_res)))
        filename = filename * "_n$(_n)seed$(_seed)"
    else
        _df = copy(df_res)
    end

    _df = flatten_and_count(_df, figure_of_merit, :depth)

    plots = Dict()
    ymin = minimum(_df[!, figure_of_merit])
    ymax = maximum(_df[!, figure_of_merit])
    for nm in plot_names
        _df_tmp = mean_on(filter(:method => ==(nm), _df), [:n, :depth], figure_of_merit; aggr_fn = aggr_fn)
        plots[nm] = @df _df_tmp plot(:depth, cols(figure_of_merit), group=:n, 
            yaxis=yaxis_scale, 
            title=replace(uppercase(nm), "_" => " "),
            xlabel="Layers",
            #ylim=(ymin,ymax),
            legend=leg_pos
            )
    end

    main_plot = plot(
        collect(values(plots))...,
        layout=(1, length(plot_names)),
        top_margin    = 5Plots.mm,
        right_margin  = 10Plots.mm,
        left_margin   = 10Plots.mm,
        bottom_margin = 10Plots.mm,
        link=:y,
        plot_title=titlecase(replace(ylabel, "_" => " "))
        )
    fig_out(filename)
    return main_plot
end

# Main
function main()
    for fn in ["mean", "var"]
        for strat in ["layers", "ball"]
            for fom in ["grad", "en", "en_errs", "rel_en_errs"]
                plot_1(fn, strat, fom, missing)
            end
        end
    end
    
    plot_2(:opt_pars_diffs, leg_pos=:topright, yaxis_scale=:log, safe_floor_agg=true)
    plot_2(:energies; leg_pos=:topright, yaxis_scale=:linear)
    plot_2(:energy_errors; leg_pos=:bottomleft, safe_floor_agg=true)
    plot_2(:approx_ratio; leg_pos=:bottomright)
    #plot_2(:relative_error; leg_pos=:bottomright, yaxis_scale=:linear) # This is just the approximation ratio
    plot_2(:ground_state_overlaps; leg_pos=:bottomleft, safe_floor_agg=true)

    individual_instances = [Dict("n" => n, "seed" => seed) for (n,seed)=product(4:2:max_num_qubits,1:5)]
    for inst in individual_instances
        for fn in ["mean", "var"]
            for strat in ["layers", "ball"]
                for fom in ["grad", "en", "en_errs", "rel_en_errs"]
                    plot_1(fn, strat, fom, inst)
                end
            end
        end
        plot_2(:energies; leg_pos=:topright, select_instance=inst, yaxis_scale=:linear)
        plot_2(:opt_pars_diffs, leg_pos=:topright, select_instance=inst, yaxis_scale=:log, safe_floor_agg=true)
        plot_2(:energy_errors; leg_pos=:bottomleft, select_instance=inst, safe_floor_agg=true)
        plot_2(:approx_ratio; leg_pos=:bottomright, select_instance=inst)
        #plot_2(:relative_error; leg_pos=:bottomright, select_instance=inst, yaxis_scale=:linear) # This is just the approximation ratio
        plot_2(:ground_state_overlaps; leg_pos=:bottomleft, select_instance=inst, safe_floor_agg=true)
    end
end

t_0 = time()
println("Starting plots..."); flush(stdout)
main()
println("Took $(time()-t_0) seconds"); flush(stdout)