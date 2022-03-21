
function calculateA(ran_efs :: Vector{RandomTerm{T}}, col_starts) where T
    n_A = col_starts[end]
    A = zeros(T, n_A, n_A)

    return calculateA!(A, ran_efs, col_starts)
end




function calculateA!(A, ran_efs, col_starts)
    ran_efs = ran_efs
    for (l, ran_ef) in enumerate(ran_efs)
        calculate_diagonal_blocksA!(A, ran_ef, col_starts[l])
    end

    for l1 = 1:length(ran_efs)
        for l2 = 1:(l1 - 1)
            caolculate_blocksA!(A, ran_efs[l1], ran_efs[l2], col_starts[l1], col_starts[l2])
        end
    end

    copy_lower_to_upper!(A)

    return Symmetric(A)
end




function calculate_diagonal_blocksA!(A, ran_ef, col_start)

    for i = 1:ran_ef.groups.n
        Z_i = get_Z_i(ran_ef, i)
        mult_range = get_mult_range(ran_ef, col_start, i)
        mul!(view(A, mult_range, mult_range), Z_i, Z_i')
    end
end









function caolculate_blocksA!(A, ran_ef1, ran_ef2, start1, start2)
    refs1 = ran_ef1.groups.refs
    refs2 = ran_ef2.groups.refs
    N = length(refs1)

    for i in 1:N
        col_range = get_mult_range(ran_ef2, start2, levelcode(refs2[i]))

        row_range = get_mult_range(ran_ef1, start1, levelcode(refs1[i]))
        mult_cols_A!(A, ran_ef1.Z, ran_ef2.Z, i, row_range, col_range)
    end
end





function mult_cols_A!(A, Z1, Z2, i, row_range, col_range)
    for j_b in 1:length(col_range)
        for i_b in 1:length(row_range)
            A[row_range[i_b], col_range[j_b]] += Z1[i_b, i] * Z2[j_b, i]
        end
    end
end





function copy_lower_to_upper!(A)
    n = size(A, 1)
    for j = 1:n
        for i = 1:(j - 1)
            A[i, j] = A[j, i]
        end
    end
end