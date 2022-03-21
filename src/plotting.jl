


function plot_procs(procs :: Process{T}; opts...) where T
    p = plot(procs.y, procs.E; legend = false, color = :gray, opts...)
    return plot!(p, procs.y, procs.E[:, 1]; color = :black)
end


