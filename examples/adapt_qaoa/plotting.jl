using AdaptBarren
using DataFrames
using Plots
using StatsPlots
using CSV
using Statistics
gr();

df = DataFrame(CSV.File("data.csv"));
gdf = groupby(df, [:n, :layer, :alg]);
cdf = combine(gdf, :err => (x -> 10^mean(log10.(safe_floor.(x)))) => :err_mean);

_layers = 20;
_df = filter(:layer => layer -> layer==_layers, cdf);
@df _df plot(:n, :err_mean,
	group=:alg,
	yaxis=:log,
	xlabel="Num Qubits",
	ylabel="Mean Energy Error",
	title="Energy Error at $_layers Layers",
	legend=:bottomright
)
savefig(".pdf")

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