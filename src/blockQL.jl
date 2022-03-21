struct QMat{T}
    mat :: Matrix{T}
    L :: Int
    ks :: Vector{Int}
end




struct LMat{T}
    mat :: LowerTriangular{T, Matrix{T}}
    L :: Int
    ks :: Vector{Int}
end




function col_range(l, mat :: Union{QMat{T}, LMat{T}}) where T
    
end





function blockQL(ran_efs, N)
    L = length(ran_efs)
    ks = cumsum( reverse!([ran_ef.k for ran_ef in ran_efs]) )
    K = ks[end]
    Q = QMat(zeros(N, K), L, ks)
    L = LMat(LowerTriangular(zeros(K, K)), L, ks)
    
    return blockQL!(Q, L, ran_efs, L)
end




function blockQL!(Q, L, lmm)
    prepareQ!(Q, ran_efs)
    prepareL!(L, ran_efs)

    for (i, ran_ef) in enumerate(ran_efs)
        QL!(Q, L, i, ran_ef.temp)

        if i < L
            for j in (i + 1):L
                remove_proj!(Q, L, i, j, ran_efs)
            end
        end
    end

    return Q, L
end




function prepareQ!(Q :: QMat{T}, ran_efs) where T
    for (i, ran_ef) in enumerate(ran_efs)

    end
end




function prepareL!(L :: LMat{T}) where T
    fill!(L.mat, zero(T))
end







































## Householder QL for the blocks



function QL!(X :: Matrix{T}) where T
    n, m = size(X)
    Q = similar(X)
    L = zeros(T, m, m)
    W = zeros(T, n, m)
    μs = zeros(T, m)

    return QL!(Q, L, W, μs, X)
end




function QL!(Q, L, W, μs, X)
    n, m = size(X)

    for k = 1:m
        j = m - k + 1
        calculate_w!(W, μs, X, j, k, n)
        multPj!(X, W, μs, k, n, j)
    end

    copy_L!(L, X, m)
    computeQ!(Q, W, μs, n, m)
    return (Q = Q, L = L, W = W, μs = μs)
end












function calculate_w!(W, μs, X, j, k, n)
    norm_x2 = X_j_norm(X, j, k, n)
    μ = norm_x2 * (norm_x2 + abs(X[k, j]))
    #μs[k] = norm_x2
    μs[k] = μ

    @turbo for i = k:n
        W[i, k] = X[i, j]
    end

    W[k, k] += sign(W[k, k]) * norm_x2
    return W
end




function X_j_norm(X :: Matrix{T}, j, k, n) where T
    res = zero(T)
    @turbo for i = k:n
        res += X[i, j]^2
    end

    return sqrt(res)
end            




function dprod!(X , W, μ, j, k, n) 
    dprod = zero(eltype(W))

    @turbo for i = k:n
        dprod += W[i, k] * X[i, j]
    end

    dprod /= μ

    @turbo for  i = k:n
        X[i, j] -= dprod * W[i, k]
    end
end




function computeQ!(Q, W, μs, n, m)
    prepareQ!(Q, m)
    
    for k = m:-1:1
        multPj!(Q, W, μs, k, n, m)
    end

    return Q
end




function prepareQ!(Q :: Matrix{T}, m) where T
    fill!(Q, zero(T))

    for j = 1:m
        k = m - j + 1
        Q[k, j] = one(T)
    end

    return Q
end




function multPj!(M :: Matrix{T}, W, μs, k, n, m) where T
    μ = μs[k]
    
    @batch  for j = 1:m
        dprod!(M, W, μ, j, k, n)
    end

    return M
end




## Extraction of L


function copy_L!(L, X, m)
    for j = 1:m
        for i = 1:(m - j + 1)
            L[m - i + 1, j] = X[i, j]
        end
    end

    return L
end
