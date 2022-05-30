using AdaptBarren
using DataFrames
using Plots
using StatsPlots
using CSV
using Statistics
gr();

FIGS_DIR = "./figs"
DATA_FILE = "./data.csv"

df = DataFrame(CSV.File(DATA_FILE));
gdf = groupby(df, [:n, :layer, :alg]);
cdf = combine(gdf, [:err => mean => :err_mean, :overlap => mean => :overlap_mean]);

# yaxis=:log, 
plot_overlap_adapt = @df filter(:alg => ==("2l"), cdf) plot(:layer, :overlap_mean, group=:n, title="ADAPT-QAOA", legend=:topleft, ylabel="Overlap (summed)");
plot_overlap_qaoa = @df filter(:alg => ==("qaoa"), cdf) plot(:layer, :overlap_mean, group=:n, title="QAOA", legend=:topleft);

plot_energy_adapt = @df filter(:alg => ==("2l"), cdf) plot(:layer, :err_mean, group=:n, legend=:bottomleft, yaxis=:log, ylims=(1e-10, 10), xlabel="p", ylabel="Energy Error");
plot_energy_qaoa = @df filter(:alg => ==("qaoa"), cdf) plot(:layer, :err_mean, group=:n, legend=:bottomleft, yaxis=:log, ylims=(1e-10, 10), xlabel="p");
plot(plot_overlap_adapt, plot_overlap_qaoa, plot_energy_adapt, plot_energy_qaoa, layout=(2,2), left_margin=15Plots.mm, bottom_margin=6Plots.mm)
savefig("$FIGS_DIR/fig.pdf")

#println("Plotting..."); flush(stdout)
### Plotting
#using Plots; gr()
#
#for n_=n_min:2:n_max
#    df_n = filter(:n => n -> n==n_, df)
#    n = n_
#    plot()
#    @df filter(:alg => ==("QAOA"), df_n) plot!(:layer, :err, group=:seed, color=:blue, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=false)
#    @df filter(:alg => ==("ADAPT"), df_n) plot!(:layer, :err, group=:seed, color=:red, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=false)
#    savefig("test_qaoa_comp_$n.pdf")
#
#    plot()
#    gdf_n = groupby(df_n, [:layer, :alg])
#    @df combine(gdf_n, :err => (x -> 10^mean(log10.(safe_floor.(x)))) => :err_mean) plot(:layer, :err_mean, group=:alg, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=true)
#    savefig("test_qaoa_comp_mean_$n.pdf")
#end
#