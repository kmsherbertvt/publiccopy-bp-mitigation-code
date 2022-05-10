using CSV
using StatsPlots
using DataFrames

n_min = 4
n_max = 14
max_pars = 40
function mean(x) return sum(x)/length(x) end
function safe_floor(x::Float64, eps=1e-15, delta=1e-8)
    if x <= -delta error("Too negative...") end
    if x <= 0.0
        return eps
    else
        return x
    end
end

df = CSV.read("data_shallow.csv", DataFrame)
println("Plotting..."); flush(stdout)

function mean(x) return sum(x)/length(x) end
function safe_floor(x::Float64, eps=1e-15, delta=1e-8)
    if x <= -delta error("Too negative...") end
    if x <= 0.0
        return eps
    else
        return x
    end
end

### Plotting
using Plots; gr()

plot()
gdf = groupby(filter(:alg => ==("QAOA"), df), [:layer, :n])
_df = combine(gdf, :err => (x -> 10^mean(log10.(safe_floor.(x)))) => :err_mean)
plot_qaoa = @df _df plot(:layer, :err_mean, group=:n, xlim=(2, max_pars), yaxis=:log, left_margin=10Plots.mm, legend=:bottomleft, title="QAOA")

gdf = groupby(filter(:alg => ==("ADAPT"), df), [:layer, :n])
_df = combine(gdf, :err => (x -> 10^mean(log10.(safe_floor.(x)))) => :err_mean)
plot_adapt = @df _df plot(:layer, :err_mean, group=:n, xlim=(2, max_pars), yaxis=:log, left_margin=10Plots.mm, legend=:bottomleft, title="ADAPT")

plot(plot_adapt, plot_qaoa, layout=(1, 2))
savefig("comparison_plot.pdf")

exit()

for n_=n_min:2:n_max
    df_n = filter(:n => n -> n==n_, df)
    n = n_
    plot()
    @df filter(:alg => ==("QAOA"), df_n) plot!(:layer, :err, group=:seed, color=:blue, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=false)
    @df filter(:alg => ==("ADAPT"), df_n) plot!(:layer, :err, group=:seed, color=:red, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=false)
    savefig("test_shallow_qaoa_comp_$n.pdf")

    plot()
    gdf_n = groupby(df_n, [:layer, :alg])
    @df combine(gdf_n, :err => (x -> 10^mean(log10.(safe_floor.(x)))) => :err_mean) plot(:layer, :err_mean, group=:alg, yaxis=:log, xlim=(2,max_pars), left_margin=10Plots.mm, legend=true)
    savefig("test_shallow_qaoa_comp_mean_$n.pdf")
end