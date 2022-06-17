using AdaptBarren
using DataFrames
using Plots
using StatsPlots
using CSV
using Statistics
gr();

FIGS_DIR = "./figs"
DATA_DIR = "./data"
gid = "CID:$(get_git_id())"

df_en = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_en_$(n).csv.test")) for n=[4,6]]...)
df_grad = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_grad_$(n).csv.test")) for n=[4,6]]...)

df_res = vcat([DataFrame(CSV.File("$(DATA_DIR)/data_res_$(n).csv.test")) for n=[4,6]]...)
#transform!(df_res, :energies => x -> eval(Meta.parse(x)) => :energies)

cols_to_eval = [
    :energies,
    :max_grads,
    :opt_pars,
    :energy_errors,
    :approx_ratio,
    :relative_error,
    :ground_state_overlaps,
]
transform!(df_res, cols_to_eval .=> ByRow(x -> eval(Meta.parse(x))) .=> cols_to_eval)

