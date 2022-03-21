




struct TempData{T}
    v1 :: Vector{T}
    vs :: Vector{Vector{T}}
    perm :: Vector{Int}
    temp :: Matrix{T}
    temp2 :: Matrix{T}
    temp3 :: Matrix{T}
end


struct GOF{T}
    lmm :: LMM{T}
    n_p :: Int

    procsF :: ProcessF{T}
    procsO :: ProcessO{T}
    procsO_ls :: Vector{ProcessO_l{T}}

    sigf_mat :: SigfMat{T}
    temp :: TempData{T}
    p_vals :: Matrix{T}
end





## GOF


function gof!(gof :: GOF{T}) where T
    modify_y!(gof)
    calculate_modified_residuals!(gof)
    standardize!(gof)
    reorder!(gof)
    construct_processes!(gof)
    scale!(gof)
    calculate_p_vals!(gof)

    return gof
end




function gof(lmm_data :: LMM_Data{T}, n_p, λ) where T
    #return gof(fit_REML(lmm_data), n_p, λ)
    return gof(fit(lmm_data), n_p, λ)
end




function gof(lmm :: MixedModels.LinearMixedModel, n_p, λ)
    return gof(LMM(lmm), n_p, λ)
end


function gof(lmm :: LMM{T}, n_p, λ) where T
    gof = prepare_gof(lmm, n_p, λ)
    return gof!(gof)
end




##
# Single level modification

function gof_sl!(gof :: GOF{T}) where T
    modify_y_single!(gof)
    calculate_modified_residuals!(gof)
    standardize!(gof)
    reorder!(gof)
    construct_processes!(gof)
    scale!(gof)
    calculate_p_vals!(gof)

    return gof
end




function gof_sl(lmm_data :: LMM_Data{T}, n_p, λ) where T
    #return gof(fit_REML(lmm_data), n_p, λ)
    return gof_sl(fit(lmm_data), n_p, λ)
end




function gof_sl(lmm :: MixedModels.LinearMixedModel, n_p, λ)
    return gof_sl(LMM(lmm), n_p, λ)
end


function gof_sl(lmm :: LMM{T}, n_p, λ) where T
    gof = prepare_gof(lmm, n_p, λ)
    return gof_sl!(gof)
end













##




function prepare_gof(lmm :: LMM{T}, n_p, λ) where T
    procF = ProcessF(lmm, n_p)
    procO = ProcessO(lmm, n_p)
    procO_ls = lmm.L > 1 ? [ProcessO_l(lmm, n_p, l) for l = Int32.(1:lmm.L)] : ProcessO_l{T}[]
    temp = prepare_temp(lmm, n_p)

    n_sigfM = calculate_n_sigfM!(temp.v1, λ, procF, procO, procO_ls)
    N = lmm.N
    sigfM = SigfMat(zeros(n_sigfM, N), zeros(Int32, N - 1), λ, N)
    p_vals = zeros(T, 2, lmm.L + 2)
    
    return GOF(lmm, n_p, procF, procO, procO_ls, sigfM, temp, p_vals)
end



function prepare_temp(lmm :: LMM{T}, n_p) where T
    N = lmm.N
    n_A = size(lmm.A, 1)
    n_H = size(lmm.H, 1)
    v1 = zeros(T, N)
    vs = [zeros(T, N) for i in 1:Threads.nthreads()]
    perm = zeros(Int, N)
    temp = zeros(n_A, n_p)
    temp2 = zeros(n_A, n_p)
    temp3 = zeros(n_H, n_p)

    return TempData(v1, vs, perm, temp, temp2, temp3)
end






function calculate_n_sigfM!(v, λ, procF, procO, procO_ls)
    n_y = length(v)
    n = calculate_n(sort!(copyto!(v, procF.y)), λ, n_y)
    n = max(n, calculate_n(sort!(copyto!(v, procO.y)), λ, n_y))
    
    for procs in procO_ls
        n = max(n, calculate_n(sort!(copyto!(v, procs.y)), λ, n_y))
    end

    return n
end



## Modification


function modify_y!(gof :: GOF{T}) where T
    lmm = gof.lmm
    temp = gof.temp
    E = gof.procsF.E
    E2 = gof.procsF.E2
    E3 = gof.procsO.E

    yh = lmm.X * lmm.β
    eh = lmm.y - yh

    fill_mat!(E, eh)
    mult_Vi05!(E, E2, temp.temp, temp.temp2, lmm)

    Threads.@threads for j = 1:size(E, 2)
        v = temp.vs[Threads.threadid()]
        rand!(v, [-1.0, 1.0])

        for i = 1:size(E, 1)
            E[i, j] *= v[i]
        end
    end

    mult_V05!(E, E2, temp.temp, temp.temp2, lmm)
    fill_mat!(E2, yh)
    add!(E, E2)
    copyto!(E2, E)
    get_fitted!(E, E3, temp.temp, temp.temp2, temp.temp3, lmm)    
    sub!(E2, E)
    E2[:, 1] = eh
    copyto!(gof.procsO.E2, E2)

    for procs in gof.procsO_ls
        copyto!(procs.E, E2)
    end
end













function modify_y_single!(gof :: GOF{T}) where T
    lmm = gof.lmm
    temp = gof.temp
    E = gof.procsF.E
    E2 = gof.procsF.E2
    E3 = gof.procsO.E

    yh = lmm.X * lmm.β
    eh = lmm.y - yh

    fill_mat!(E, eh)
    mult_Vi05!(E, E2, temp.temp, temp.temp2, lmm)

    Threads.@threads for j = 1:size(E, 2)

        for i = 1:lmm.ran_efs[1].groups.n
            indexes_i = lmm.ran_efs[1].groups.indexes[i]
            E[indexes_i, j] .*= rand([-1.0, 1.0])
        end
    end

    mult_V05!(E, E2, temp.temp, temp.temp2, lmm)
    fill_mat!(E2, yh)
    add!(E, E2)
    E[:, 1] = lmm.y
    copyto!(E2, E)
    get_fitted!(E, E3, temp.temp, temp.temp2, temp.temp3, lmm)    
    sub!(E2, E)
    #E2[:, 1] = eh
    copyto!(gof.procsO.E2, E2)

    for procs in gof.procsO_ls
        copyto!(procs.E, E2)
    end
end









## Modified residuals

function calculate_modified_residuals!(gof :: GOF{T}) where T
    lmm = gof.lmm
    temp = gof.temp 
    
    calculate_modified_residuals!(gof.procsO, temp, lmm)

    for procs in gof.procsO_ls
        #copyto!(procs.E2, gof.procsO.E2)
        calculate_modified_residuals!(procs, temp, lmm)
    end


end


function calculate_modified_residuals!(procs :: ProcessO{T}, temp, lmm) where T
    mult_Vi!(procs.E2, procs.E, temp.temp, temp.temp2, lmm)
end




function calculate_modified_residuals!(procs :: ProcessO_l{T}, temp, lmm) where T
    mult_Vi_l!(procs.E2, lmm.ran_efs[procs.l], procs.E)

end







    








## Standardizing



function standardize!(gof :: GOF{T}) where T
    lmm = gof.lmm
    temp = gof.temp

    standardize!(gof.procsF, temp, lmm)
    standardize!(gof.procsO, temp, lmm)

    for procs in gof.procsO_ls
        standardize!(procs, temp, lmm)
    end
end



function standardize!(procs :: ProcessF{T}, temp, lmm) where T
    mult_Vi05!(procs.E2, procs.E, temp.temp, temp.temp2, lmm)
    #mult_Vi05_slow!(procs.E2, lmm.ran_efs[1])
end



function standardize!(procs :: ProcessO{T}, temp, lmm) where T
    mult_P!(procs.E2, procs.E, temp.temp, temp.temp2, lmm)
    #mult_P_slow!(procs.E2, lmm.ran_efs[1])
end



function standardize!(procs :: ProcessO_l{T}, temp, lmm) where T
    mult_P_l!(procs.E2, procs.E, temp.temp, lmm, procs.l)
end




## Reordering

function reorder!(gof :: GOF{T}) where T
    temp = gof.temp
    reorder!(gof.procsF, temp)
    reorder!(gof.procsO, temp)

    for procs in gof.procsO_ls
        reorder!(procs, temp)
    end
end



function reorder!(procs :: Process{T}, temp) where T
    v = temp.v1
    copyto!(v, procs.y)
    perm = temp.perm
    sortperm!(perm, procs.y)


    Threads.@threads for i = 1:size(procs.E, 1)
        p_i = perm[i]
        procs.y[i] = v[p_i]

        for j = 1:size(procs.E, 2)
            procs.E[i, j] = procs.E2[p_i, j]
        end
    end
end




## Scaling


function scale!(gof :: GOF{T}) where T
    C_F = 1 / (gof.lmm.σ * sqrt(length(gof.lmm.y)))
    C_O = gof.lmm.σ / sqrt(length(gof.lmm.y))
    scale!(gof.procsF, C_F)
    scale!(gof.procsO, C_O)

    for procs in gof.procsO_ls
        scale!(procs, C_O)
    end
end


function scale!(procs :: Process{T}, C :: T) where T
    E = procs.E
    
    Threads.@threads for j = 1:size(E, 2)
        @turbo for i = 1:size(E, 1)
            E[i, j] *= C
        end
    end
end





## Construction of processes


function construct_processes!(gof :: GOF{T}) where T
    sigfM = gof.sigf_mat
    construct_processes!(gof.procsF, sigfM)
    construct_processes!(gof.procsO, sigfM)

    for procs in gof.procsO_ls
        construct_processes!(procs, sigfM)
    end
end



function construct_processes!(procs :: Process{T}, sigfM) where T
    copyto!(procs.E2, procs.E)
    update!(sigfM, procs.y)
    mult!(procs.E, sigfM)
end



## Test statistics


function calculate_p_vals!(gof :: GOF{T}) where T
    temp = gof.temp.temp
    p_vals = gof.p_vals

    calculate_p_vals!(gof.procsF, temp, p_vals, 1)
    calculate_p_vals!(gof.procsO, temp, p_vals, 2)

    for (l, procs) in enumerate(gof.procsO_ls)
        calculate_p_vals!(procs, temp, p_vals, l + 2)
    end
end



function calculate_p_vals!(procs :: Process{T}, temp, p_vals, p_id) where T
   
    Threads.@threads for j in 1:size(procs.E, 2)
        proc = view(procs.E, :, j)
        temp[1, j] = maximum(abs, proc)
        temp[2, j] = norm(proc, 2)
    end

    t_KS = temp[1, 1]
    t_SS = temp[2, 1]
    n_p = size(procs.E, 2)

    p_vals[1, p_id] = count(x -> x > t_KS, view(temp, 1, :)) / (n_p - 1)
    p_vals[2, p_id] = count(x -> x > t_SS, view(temp, 2, :)) / (n_p - 1)
end