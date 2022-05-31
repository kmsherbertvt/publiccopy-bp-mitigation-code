using AdaptBarren
using DataFrames
using Plots
using StatsPlots
using CSV
using Statistics
gr();

FIGS_DIR = "./figs"
BASE_NAME = "qaoa_max_cut_"
DATA_FILE = "./data.csv"

df = DataFrame(CSV.File(DATA_FILE));
gdf = groupby(df, [:n, :layer, :alg]);
cdf = combine(gdf, [:err => mean => :err_mean, :overlap => mean => :overlap_mean, :rel_err => mean => :rel_err_mean, :apprx => mean => :apprx_mean]);

plot_adapt = @df filter(:alg => ==("2l"), cdf)   plot(:layer, :overlap_mean, group=:n, legend=:topleft, ylim=(0,1), title="ADAPT-QAOA", ylabel="Overlap (summed)");
plot_qaoa  = @df filter(:alg => ==("qaoa"), cdf) plot(:layer, :overlap_mean, group=:n, legend=:topleft, ylim=(0,1), title="QAOA");
plot(plot_adapt, plot_qaoa, layout=(2,1)); savefig("$FIGS_DIR/$(BASE_NAME)_overlap.pdf")

plot_adapt = @df filter(:alg => ==("2l"), cdf)   plot(:layer, :apprx_mean, group=:n, legend=:topleft, ylim=(0.5,1), title="ADAPT-QAOA", ylabel="Approximation Ratio");
plot_qaoa  = @df filter(:alg => ==("qaoa"), cdf) plot(:layer, :apprx_mean, group=:n, legend=:topleft, ylim=(0.5,1), title="QAOA");
plot(plot_adapt, plot_qaoa, layout=(2,1)); savefig("$FIGS_DIR/$(BASE_NAME)_approx_ratio.pdf")

plot_adapt = @df filter(:alg => ==("2l"), cdf)   plot(:layer, :err_mean, group=:n, legend=:topleft, ylim=(1e-10, 1), yaxis=:log, title="ADAPT-QAOA", ylabel="Energy Error");
plot_qaoa  = @df filter(:alg => ==("qaoa"), cdf) plot(:layer, :err_mean, group=:n, legend=:topleft, ylim=(1e-10, 1), yaxis=:log, title="QAOA");
plot(plot_adapt, plot_qaoa, layout=(2,1)); savefig("$FIGS_DIR/$(BASE_NAME)_energy_error.pdf")

plot_adapt = @df filter(:alg => ==("2l"), cdf)   plot(:layer, :rel_err_mean, group=:n, legend=:topleft, ylim=(1e-10, 1), yaxis=:log, title="ADAPT-QAOA", ylabel="Relative Error");
plot_qaoa  = @df filter(:alg => ==("qaoa"), cdf) plot(:layer, :rel_err_mean, group=:n, legend=:topleft, ylim=(1e-10, 1), yaxis=:log, title="QAOA");
plot(plot_adapt, plot_qaoa, layout=(2,1)); savefig("$FIGS_DIR/$(BASE_NAME)_relative_energy_error.pdf")