using AdaptBarren
using DataFrames
using Plots
using StatsPlots
using CSV
using Statistics
gr();

FIGS_DIR = "./figs"
BASE_NAME = "qaoa_max_cut"
DATA_FILE = "./data.csv"
gid = "CID:$(get_git_id())"

df = DataFrame(CSV.File(DATA_FILE));
gdf = groupby(df, [:n, :layer, :alg]);
cdf = combine(gdf, [:err => log_mean => :err_mean, :overlap => mean => :overlap_mean, :rel_err => log_mean => :rel_err_mean, :apprx => mean => :apprx_mean]);

plot_adapt = @df filter(:alg => ==("2l"), cdf)   plot(:layer, :overlap_mean, group=:n, xlabel="Parameters", legend=:bottomright, ylim=(0,1), title="ADAPT-QAOA, $gid", ylabel="Overlap (summed)");
plot_qaoa  = @df filter(:alg => ==("qaoa"), cdf) plot(:layer, :overlap_mean, group=:n, xlabel="Parameters", legend=:bottomright, ylim=(0,1), title="QAOA, $gid");
plot(plot_adapt, plot_qaoa, layout=(1,2), top_margin = 5Plots.mm, left_margin = 12Plots.mm, bottom_margin=5Plots.mm); savefig("$FIGS_DIR/$(BASE_NAME)_overlap.pdf")

plot_adapt = @df filter(:alg => ==("2l"), cdf)   plot(:layer, :apprx_mean, group=:n, xlabel="Parameters", legend=:bottomright, ylim=(0.75,1), title="ADAPT-QAOA, $gid", ylabel="Approximation Ratio");
plot_qaoa  = @df filter(:alg => ==("qaoa"), cdf) plot(:layer, :apprx_mean, group=:n, xlabel="Parameters", legend=:bottomright, ylim=(0.75,1), title="QAOA, $gid");
plot(plot_adapt, plot_qaoa, layout=(1,2), top_margin = 5Plots.mm, left_margin = 12Plots.mm, bottom_margin=5Plots.mm); savefig("$FIGS_DIR/$(BASE_NAME)_approx_ratio.pdf")

plot_adapt = @df filter(:alg => ==("2l"), cdf)   plot(:layer, :err_mean, group=:n, xlabel="Parameters", legend=:bottomleft, ylim=(1e-15, 10), yaxis=:log, title="ADAPT-QAOA, $gid", ylabel="Energy Error");
plot_qaoa  = @df filter(:alg => ==("qaoa"), cdf) plot(:layer, :err_mean, group=:n, xlabel="Parameters", legend=:bottomleft, ylim=(1e-15, 10), yaxis=:log, title="QAOA, $gid");
plot(plot_adapt, plot_qaoa, layout=(1,2), top_margin = 5Plots.mm, left_margin = 12Plots.mm, bottom_margin=5Plots.mm); savefig("$FIGS_DIR/$(BASE_NAME)_energy_error.pdf")

plot_adapt = @df filter(:alg => ==("2l"), cdf)   plot(:layer, :rel_err_mean, group=:n, xlabel="Parameters", legend=:bottomleft, ylim=(1e-15, 1), yaxis=:log, title="ADAPT-QAOA, $gid", ylabel="Relative Error");
plot_qaoa  = @df filter(:alg => ==("qaoa"), cdf) plot(:layer, :rel_err_mean, group=:n, xlabel="Parameters", legend=:bottomleft, ylim=(1e-15, 1), yaxis=:log, title="QAOA, $gid");
plot(plot_adapt, plot_qaoa, layout=(1,2), top_margin = 5Plots.mm, left_margin = 12Plots.mm, bottom_margin=5Plots.mm); savefig("$FIGS_DIR/$(BASE_NAME)_relative_energy_error.pdf")