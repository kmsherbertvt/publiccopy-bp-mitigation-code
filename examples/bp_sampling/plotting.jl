using AdaptBarren
using DataFrames
using Plots
using StatsPlots
using CSV
using Statistics
gr();

FIGS_DIR = "./figs"
DATA_FILE = "./data/data_*.csv.test"
gid = "CID:$(get_git_id())"

df = DataFrame(CSV.File(DATA_FILE));