source ${ZDOTDIR-$HOME}/.zsh/*.zsh

julia --project=@. --procs=28 bp_sampling.jl > output_sims.txt
finished "Simulations"
julia --project=@. plotting.jl > output_plots.txt
finished "Plots"
