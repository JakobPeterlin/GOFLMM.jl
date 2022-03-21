
struct ProcessF{T}
    y :: Vector{T}
    e :: Vector{T}
    E :: Matrix{T}
    E2 :: Matrix{T}
end




struct ProcessO{T}
    y :: Vector{T}
    e :: Vector{T}
    E :: Matrix{T}
    E2 :: Matrix{T}
end




struct ProcessO_l{T}
    y :: Vector{T}
    e :: Vector{T}
    E :: Matrix{T}
    E2 :: Matrix{T}
    l :: Int32
end





Process{T} = Union{ProcessF{T}, ProcessO{T}, ProcessO_l{T}} where T





function fill_mat!(M, v)
    Threads.@threads for j = 1:size(M, 2)
        for i = 1:size(M, 1)
            M[i, j] = v[i]
        end
    end

    return M
end
        


function Process(lmm :: LMM{T}, n_p, y_fun, l) where T
    y = y_fun(lmm, l)
    e = lmm.y - y
    N = length(y)
    E = fill_mat!(zeros(T, N, n_p), e)
    E2 = copy(E)

    return (y, e, E, E2)
end


ProcessF(lmm, n_p) = ProcessF(Process(lmm, n_p, calculate_yF, 0)...)



ProcessO(lmm, n_p) = ProcessO(Process(lmm, n_p, calculate_yO, 0)...)


ProcessO_l(lmm, n_p, l :: Int32) = ProcessO_l(Process(lmm, n_p, calculate_yO_l, l)..., l)




function calculate_yF(lmm, _)
    return lmm.X * lmm.β
end



function calculate_yO(lmm, _)
    y = lmm.X * lmm.β

    for ran_ef in lmm.ran_efs
        y += ran_ef.Zb
    end

    return y
end


function calculate_yO_l(lmm, l)
    y = lmm.X * lmm.β
    y += lmm.ran_efs[l].Zb
    return y
end