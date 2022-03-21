





struct GroupIndexes
    n :: Int32
    k :: Int32
    ns :: Vector{Int32}
    refs :: CategoricalArray{Int32, 1, UInt32}
    indexes :: Vector{Vector{Int32}}
    col_ranges :: Vector{UnitRange{Int32}}
    ncols :: Int32
end




 struct RandomTerm{T}
    groups :: GroupIndexes
    Z :: Matrix{T}
    λ :: LowerTriangular{T, Matrix{T}}
    b :: Matrix{T}
    Zb :: Vector{T}

    QRs :: Vector{LinearAlgebra.QRCompactWY{T, Matrix{T}}}
end




function indexes_i(ran_ef, i)
    return ran_ef.groups.indexes[i]
end








function GroupIndexes(refs, n, k)
    indexes = [Int32[] for i = 1:n]
    
    for (i, r_i) in enumerate(refs)
        push!(indexes[unwrap(r_i)], i)
    end

    ns = length.(indexes)
    col_ranges = [(1 + (i - 1) * k):(i * k) for i = 1:n]
    return GroupIndexes(n, k, ns, refs, indexes, col_ranges, n * k)
end




function update_groups!(groups :: GroupIndexes, refs, k)
    new_groups = GroupIndexes(refs, length(groups.ns), k)
    
    copyto!(groups.ns, new_groups.ns)
    copyto!(groups.refs, new_groups.refs)
    copyto!(groups.indexes, new_groups.indexes)
    copyto!(groups.col_ranges, new_groups.col_ranges)
end



function RandomTerm(n, refs, Z, λ, b)
    k = size(Z, 1)
    groups = GroupIndexes(refs, n, k)
    Zb = calculate_Zb(b, Z , groups)
    
    QRs = [qr(get_Z_i(Z, i, groups)') for i = 1:n]

    return RandomTerm(groups, Z, λ, b, Zb, QRs)
end





function calculate_Zb(b, Z, groups :: GroupIndexes)
    Zb = zeros(size(Z, 2))

    for i = 1:groups.n
        indexes_i = groups.indexes[i]
        mul!(view(Zb, indexes_i), view(Z, :, indexes_i)', view(b, :, i))
    end

    return Zb
end
    




function get_Z_i(Z, i, groups :: GroupIndexes)
    return view(Z, :, groups.indexes[i])
end




function get_Z_i(ran_ef :: RandomTerm, i)
    return get_Z_i(ran_ef.Z, i, ran_ef.groups)
end




function RandomTerm(lmm :: LinearMixedModel, i)
    retrm = lmm.reterms[i]
    n = length(retrm.levels)
    return RandomTerm(n, retrm.refs, retrm.z, retrm.λ, lmm.b[i])
end












function multQ!(Y, ran_ef :: RandomTerm{T}, X) where T

    for i in ran_ef.groups.n
        indexes_i = ran_ef.groups.indexes[i]
        cols_i = ran_ef.groups.col_ranges[i]
        #TODO - zamenjaj
        Q_i = Matrix(ran_ef.QRs[i].Q)
        
        mul!(view(Y, indexes_i, :), Q_i, view(X, cols_i, :))
    end
end




function multQt!(Y, ran_ef :: RandomTerm{T}, X) where T

    for i in ran_ef.groups.n
        indexes_i = ran_ef.groups.indexes[i]
        cols_i = ran_ef.groups.col_ranges[i]
        #TODO - zamenjaj
        Q_i = Matrix(ran_ef.QRs[i].Q)

        mul!(view(Y, cols_i, :), Q_i', view(X, indexes_i, :))
    end
end




function mult_Vi_l!(Y, ran_ef :: RandomTerm{T}, X) where T
    for i in 1:ran_ef.groups.n
        indexes_i = ran_ef.groups.indexes[i]
        Z_i = get_Z_i(ran_ef, i)'
        D = ran_ef.λ * ran_ef.λ'
        temp =  (I + D * Z_i' * Z_i) \ (D *  (Z_i' * X[indexes_i, :]))
        mul!(view(Y, indexes_i, :), Z_i, temp, - 1.0, 1.0)
    end

    return Y
end



















## For testing

function get_ZT(ran_ef :: RandomTerm{T}) where T
    ZT = zeros(size(ran_ef.Z, 2), ran_ef.groups.ncols)

    for i in 1:ran_ef.groups.n
        indexes_i = ran_ef.groups.indexes[i]
        cols_i = ran_ef.groups.col_ranges[i]
        ZT[indexes_i, cols_i] = ran_ef.Z[:, indexes_i]'
    end

    return ZT
end



function get_DT(ran_ef :: RandomTerm{T}) where T
    DT = zeros(ran_ef.groups.ncols, ran_ef.groups.ncols)
    D = ran_ef.λ * ran_ef.λ'

    for i in 1:ran_ef.groups.n
        cols_i = ran_ef.groups.col_ranges[i]
        DT[cols_i, cols_i] = D
    end

    return DT
end


function get_VTi(ran_ef)
    ZT = get_ZT(ran_ef)
    DT = get_DT(ran_ef)
    return inv(I + ZT * DT * ZT')
end



function get_Vi05_i(ran_ef :: RandomTerm{T}, i) where T
    D = ran_ef.λ * ran_ef.λ'
    Z_i = get_Z_i(ran_ef, i)'
    return inv(cholesky(Symmetric(I + Z_i * D * Z_i')).L )
end




function get_P_i(ran_ef :: RandomTerm{T}, i) where T
    D = ran_ef.λ * ran_ef.λ'
    Z_i = get_Z_i(ran_ef, i)'
    return I - Z_i * inv(Z_i' * Z_i) * Z_i'
end



function mult_Vi05_slow!(Y, ran_ef)
    for i = 1:ran_ef.groups.n
        indexes_i = ran_ef.groups.indexes[i]
        Vi05_i = get_Vi05_i(ran_ef, i)
        Y[indexes_i, :] =  Vi05_i * Y[indexes_i, :]
    end

    return Y
end



function mult_P_slow!(Y, ran_ef)
    for i = 1:ran_ef.groups.n
        indexes_i = ran_ef.groups.indexes[i]
        P_i = get_P_i(ran_ef, i)
        Y[indexes_i, :] =  P_i * Y[indexes_i, :]
    end

    return Y
end



