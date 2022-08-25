source ${ZDOTDIR-$HOME}/.zsh/*.zsh

julia --project=@. plotting.jl > output_plots.txt
finished "Plots"
