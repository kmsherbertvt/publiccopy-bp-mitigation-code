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

# Utility Functions
function mean_on(df, grp_cols, mean_col; aggr_fn = mean)
    gdf = groupby(df, grp_cols)
    return combine(gdf, mean_col => (x -> aggr_fn(x)) => mean_col)
end

function fig_out(filename)
    savefig("$(FIGS_DIR)/$(filename).pdf")
end

# Main Plots

function plot_1()
    """ 4 plots, x axis is number of layers, y axis is var grad, hue num qubits """
    plot_names = unique(df_en[!, :method])

    plots = Dict()
    for nm in plot_names
        _df = mean_on(filter(:method => ==(nm), df_grad), [:n, :d], :grad; aggr_fn = var)
        plots[nm] = @df _df plot(:d, :grad, group=:n, 
            yaxis=:log, 
            title=uppercase(nm),
            xlabel="Layers",
            ylabel="var(grad)"
            )
    end

    main_plot = plot(collect(values(plots))...)
    fig_out("variance_convergence")
    return main_plot
end

# Main
function main()
    plot_1()
end

main()