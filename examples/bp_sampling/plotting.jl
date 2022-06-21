using AdaptBarren
using DataFrames
using Plots
using StatsBase
using StatsPlots
using CSV
using Statistics
gr();

# Hyperparameters
FIGS_DIR = "./figs"
DATA_DIR = "./data"
DATA_SUFFIX = "csv"
qubit_range = 4:2:12
gid = "CID:$(get_git_id())"

# Data Import
df_en = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_en_$(n).$(DATA_SUFFIX)")) for n=qubit_range]...)
df_grad = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_grad_$(n).$(DATA_SUFFIX)")) for n=qubit_range]...)
df_res = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_res_$(n).$(DATA_SUFFIX)")) for n=qubit_range]...)
cols_to_eval = [ :energies, :max_grads, :opt_pars, :energy_errors, :approx_ratio, :relative_error, :ground_state_overlaps, ]
transform!(df_res, cols_to_eval .=> ByRow(x -> eval(Meta.parse(x))) .=> cols_to_eval)
transform!(df_res, :energies .=> ByRow(x -> length(x)) .=> :final_depth)

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

function plot_1(aggr)
    """ 4 plots, x axis is number of layers, y axis is var grad, hue num qubits """
    plot_names = unique(df_en[!, :method])

    if aggr === "var"
        filename = "convergence_gradient_variance"
        aggr_fn = var
        ylabel = "Var(Grad)"
    elseif aggr === "mean"
        filename = "convergence_gradient_mean"
        aggr_fn = x -> mean(abs.(x))
        ylabel = "Mean(Grad)"
    end

    plots = Dict()
    for nm in plot_names
        _df = mean_on(filter(:method => ==(nm), df_grad), [:n, :d], :grad; aggr_fn = aggr_fn)
        plots[nm] = @df _df plot(:d, :grad, group=:n, 
            yaxis=:log, 
            title=uppercase(nm),
            xlabel="Layers",
            ylabel=ylabel
            )
    end

    main_plot = plot(collect(values(plots))...)
    fig_out(filename)
    return main_plot
end

function plot_2(figure_of_merit; leg_pos = :topright)
    """ 3 plots comparing layer-wise performance,
    x axis is number of layers, y axis is mean figure of merit, hue num qubits """
    plot_names = ["qaoa", "adapt_qaoa_2l", "adapt_vqe_2l"]

    filename = "convergence_$(string(figure_of_merit))"
    aggr_fn = x -> mean(abs.(x))
    ylabel = string(figure_of_merit)

    _df = copy(df_res)
    _df = flatten_and_count(_df, figure_of_merit, :depth)

    plots = Dict()
    for nm in plot_names
        _df_tmp = mean_on(filter(:method => ==(nm), _df), [:n, :depth], figure_of_merit; aggr_fn = aggr_fn)
        plots[nm] = @df _df_tmp plot(:depth, cols(figure_of_merit), group=:n, 
            yaxis=:log, 
            title=uppercase(nm),
            xlabel="Layers",
            ylabel=ylabel,
            legend=leg_pos
            )
    end

    main_plot = plot(collect(values(plots))..., layout=(1, length(plot_names)))
    fig_out(filename)
    return main_plot
end

# Main
function main()
    plot_1("mean")
    plot_1("var")
    plot_2(:energies; leg_pos=:topright)
    plot_2(:energy_errors; leg_pos=:bottomleft)
    plot_2(:approx_ratio; leg_pos=:bottomright)
    plot_2(:relative_error; leg_pos=:bottomright)
    plot_2(:ground_state_overlaps; leg_pos=:bottomleft)
end

main()