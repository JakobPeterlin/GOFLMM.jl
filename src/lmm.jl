

struct LMM{T}
    y :: Vector{T}
    X :: Matrix{T}

    β :: Vector{T}
    σ :: T

    ran_efs :: Vector{RandomTerm{T}}
    N :: Int32
    L :: Int32
    col_starts :: Vector{Int32}

    A :: Symmetric{T, Matrix{T}}
    H :: Cholesky{T, Matrix{T}}
    
    #TODO: No need if the process O is based on e^O.
    Ai :: Matrix{T}
    Ai05 :: Matrix{T}
    D :: Matrix{T}
    Bi :: Matrix{T}
    C :: Matrix{T}
    Ci :: Matrix{T}
end




function LMM(lmm :: LinearMixedModel)
    y = Vector(lmm.y)
    rank_range = lmm.feterm.piv[1 : lmm.feterm.rank]
    X = lmm.feterm.x[:, rank_range]

    β = lmm.β[rank_range]
    σ = lmm.σ
    L = length(lmm.reterms)
    ran_efs = [RandomTerm(lmm, i) for i in 1:L]

    i0 = zero(Int32)
    col_starts = zeros(Int32, length(ran_efs) + 1)

    for l = 1:L
        i0 += ran_efs[l].groups.ncols
        col_starts[l + 1] = i0
    end


    A, A05, Ai, Ai05 = calculate_As(ran_efs, col_starts)
    D = calculate_D(ran_efs, col_starts)
    
    #TODO: No need if the process O is based on e^O.

    Bi = inv(I + D * A)
    C, Ci = calculate_Cs(A05 * D * A05)
    H = calculate_H(copy(X), ran_efs, col_starts, D, Bi)
    return LMM(y, X, β, σ, ran_efs, Int32(length(y)), Int32(L), col_starts, A, H, Ai, Ai05, D, Bi, C, Ci)
end



function calculate_H(X, ran_efs :: Vector{RandomTerm{T}}, col_starts, D, Bi) where T
    X1 = copy(X)
    X2 = copy(X)
    temp = zeros(size(D, 2), size(X, 2))
    temp2 = copy(temp)
    mult_Vi!(X1, X2, temp, temp2, ran_efs, col_starts, D, Bi)
    H  = Symmetric(X' * X1)
    return cholesky(H)
end




function calculate_As(ran_efs, col_starts)
    A = calculateA(ran_efs, col_starts)
    eig = eigen(A)
    V = Matrix(eig.vectors)
    inv_values = invert_values(eig.values)

    A05 = V * Diagonal(sqrt_values(eig.values)) * V'
    Ai = V * Diagonal(inv_values) * V'
    Ai05 = V * Diagonal(sqrt.(inv_values)) * V'

    return (A, A05, Ai, Ai05)
end




function calculate_D(ran_efs :: Vector{RandomTerm{T}}, col_starts) where T
    n_D = 0

    for ran_ef in ran_efs
        n_D += ran_ef.groups.ncols
    end

    D = zeros(T, n_D, n_D)
    return calculate_D!(D, ran_efs, col_starts)
end



function calculate_D!(D, ran_efs, col_starts)
    for (l, ran_ef) in enumerate(ran_efs)
        for i in 1:ran_ef.groups.n
            mult_range = get_mult_range(ran_ef, col_starts[l], i)
            D[mult_range, mult_range] = ran_ef.λ * ran_ef.λ'
        end
    end

    return D
end




function invert_values(values, tol = 10^(-12))
    invs = zeros(length(values))

    for (i, val) in enumerate(values)
        if abs(val) > tol
            invs[i] = 1 / val
        end
    end

    return invs
end



function sqrt_values(values, tol = 10^(-12))
    sqrts = zeros(length(values))

    for (i, val) in enumerate(values)
        if val > 0
            sqrts[i] = sqrt(val)
        end
    end

    return sqrts
end






function get_mult_range(lmm :: LMM{T}, l, i) where T
    return get_mult_range(lmm.ran_efs[l], lmm.col_starts[l], i)
end



function get_mult_range(ran_ef :: RandomTerm{T}, start, i) where T
     return ran_ef.groups.col_ranges[i] .+ start
end



function multZ!(Y, lmm :: LMM{T}, X) where T
    multZ!(Y, lmm.ran_efs, lmm.col_starts, X)
end



function multZt!(Y, lmm :: LMM{T}, X) where T
    multZt!(Y, lmm.ran_efs, lmm.col_starts, X)
end


function multZ!(Y, ran_efs, col_starts, X) where T
    fill!(Y, zero(eltype(Y)))

    for (l, ran_ef) in enumerate(ran_efs)
        for i in 1:ran_ef.groups.n
            Z_i = get_Z_i(ran_ef, i)
            indexes_li = ran_ef.groups.indexes[i]
            mult_range = get_mult_range(ran_efs[l], col_starts[l], i)
            mul!(view(Y, indexes_li, :), Z_i', view(X, mult_range, :), 1.0, 1.0)
        end
    end
end



function multZt!(Y, ran_efs, col_starts, X) where T
    fill!(Y, zero(eltype(Y)))

    for (l, ran_ef) in enumerate(ran_efs)
        for i in 1:ran_ef.groups.n
            Z_i = get_Z_i(ran_ef, i)
            indexes_li = ran_ef.groups.indexes[i]
            mult_range = get_mult_range(ran_efs[l], col_starts[l], i)
            mul!(view(Y, mult_range, :), Z_i, view(X, indexes_li, :), 1.0, 1.0)
        end
    end
end








function mult_Vi!(X, X2, temp, temp2, lmm :: LMM{T}) where T
    return  mult_Vi!(X, X2, temp, temp2, lmm.ran_efs, lmm.col_starts, lmm.D, lmm.Bi)
end



function mult_Vi!(X, X2, temp, temp2, ran_efs, col_starts, D, Bi)
    multZt!(temp, ran_efs, col_starts, X)
    mul!(temp2, D, temp)
    mul!(temp, Bi, temp2)
    multZ!(X2, ran_efs, col_starts, temp)
    sub!(X, X2)
end












function get_fitted!(Y, Y2, temp, temp2, temp3, lmm)
    mult_Vi!(Y, Y2, temp, temp2, lmm)
    mul!(temp3, lmm.X', Y)
    ldiv!(lmm.H, temp3)
    mul!(Y, lmm.X, temp3)
    return Y
end
















function multQt!(Y, temp, lmm :: LMM{T}, X) where T
    multZt!(temp, lmm, X)
    mul!(Y, lmm.Ai05, temp)
end



function multQ!(Y, temp, lmm :: LMM{T}, X) where T
    mul!(temp, lmm.Ai05, X)
    multZ!(Y, lmm, temp)
end




function mult_V05!(X, X2, temp, temp2, lmm)
    multQt!(temp, temp2, lmm, X)
    mul!(temp2, lmm.C, temp)
    multQ!(X2, temp, lmm, temp2)
    add!(X, X2)
end



function mult_Vi05!(X, X2, temp, temp2, lmm)
    multQt!(temp, temp2, lmm, X)
    mul!(temp2, lmm.Ci, temp)
    multQ!(X2, temp, lmm, temp2)
    add!(X, X2)
end




function calculate_Cs(M)
    eig = eigen(Symmetric(M))
    V = eig.vectors
    μs = sqrt_values(eig.values .+ 1) .- 1
    μis = 1 ./ sqrt_values(eig.values .+ 1) .- 1
    C = V * Diagonal(μs) * V'
    Ci = V * Diagonal(μis) * V'
    return C, Ci
end



function mult_P!(X, X2, temp, temp2, lmm)
    multZt!(temp, lmm, X)
    mul!(temp2, lmm.Ai, temp)
    multZ!(X2, lmm, temp2)
    sub!(X, X2)
    return X
end




function mult_P_l!(X, X2, temp, lmm, l)
    ran_ef = lmm.ran_efs[l]
    X3 = view(temp, 1:ran_ef.groups.ncols, :)
    multQt!(X3, ran_ef, X)
    multQ!(X2, ran_ef, X3)
    sub!(X, X2)
end