
struct Simulations{T}
    p_vals_KS :: Matrix{T}
    p_vals_SS :: Matrix{T}
end



function repeat_simulations(reps, n_p, λ, sim_fun, sim_args)
    lmm = sim_fun(sim_args...)
    lmm_gof = gof(lmm, n_p, λ)
    L = size(lmm_gof.p_vals, 2)

    p_vals_KS = zeros(reps, L)
    p_vals_SS = zeros(reps, L)
    p_vals_KS[1, :] = lmm_gof.p_vals[1, :]
    p_vals_SS[1, :] = lmm_gof.p_vals[2, :]

    #@showprogress 
    for i in ProgressBar(2:reps)
        lmm = fit_REML(sim_fun(sim_args...))
        lmm_gof = gof(lmm, n_p, λ)
        p_vals_KS[i, :] = lmm_gof.p_vals[1, :]
        p_vals_SS[i, :] = lmm_gof.p_vals[2, :]
    end

    return Simulations(p_vals_KS, p_vals_SS)
end




function repeat_simulations_sl(reps, n_p, λ, sim_fun, sim_args)
    lmm = sim_fun(sim_args...)
    lmm_gof = gof_sl(lmm, n_p, λ)
    L = size(lmm_gof.p_vals, 2)

    p_vals_KS = zeros(reps, L)
    p_vals_SS = zeros(reps, L)
    p_vals_KS[1, :] = lmm_gof.p_vals[1, :]
    p_vals_SS[1, :] = lmm_gof.p_vals[2, :]

    #@showprogress 
    for i in ProgressBar(2:reps)
        lmm = fit_REML(sim_fun(sim_args...))
        lmm_gof = gof_sl(lmm, n_p, λ)
        p_vals_KS[i, :] = lmm_gof.p_vals[1, :]
        p_vals_SS[i, :] = lmm_gof.p_vals[2, :]
    end

    return Simulations(p_vals_KS, p_vals_SS)
end



function hist_KS(sims :: Simulations{T}) where T
    histogram(sims.p_vals_KS[:, 1])
end