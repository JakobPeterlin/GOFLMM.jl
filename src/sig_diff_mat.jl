


struct SigfMat{T}
    M :: Matrix{T}
    prod_lengths :: Vector{Int32}
    λ :: T
    n_y :: Int32
end







function SigfMat(y :: Vector{T}, λ :: T) where T
    n_y = length(y)
    n_min = calculate_n(y, λ, n_y)
    return SigfMat(y, n_min, λ, Int32(n_y))
end








function SigfMat(y :: Vector{T}, n_min, n_y) where T
    λ = calculate_λ(y, n_min, n_y)
    
    return SigfMat(y, n_min, λ, n_y)
end




function SigfMat(y :: Vector{T}, n_min, λ :: T, n_y :: Int32) where T
    M = zeros(T, n_min, n_y - 1)
    prod_lengths = zeros(Int32, n_y - 1)
    return update!(SigfMat(M, prod_lengths, λ, n_y), y)
end
    
    
   
function update!(sigfM :: SigfMat{T}, y :: Vector{T}) where T
    M = sigfM.M
    prod_lengths = sigfM.prod_lengths
    λ = sigfM.λ
    n_y = sigfM.n_y
    δ = 1 / λ
    for i = 1:(n_y - 1)
        y_i = y[i]

        for j = (i + 1):n_y
            if (y[j] - y_i) < δ
                prod_lengths[i] = j - i
                M[j - i, i] = sigf(λ * (y[j] - y[i]))
            else
                break
            end
        end
    end

    return sigfM
end










p(x) =  - x^3 * (10 - 15 * x + 6 * x^2) + 1

sigf(x) = max(min(p(x), 1.0), 0.0)

# Assume that y is sorted

function calculate_λ(y, n, n_y)
    diff_max = y[min(n, n_y)] - y[1]

    if n_y > n
        for i = 1:(n_y - n)
            diff_max = max(diff_max, y[i + n] - y[i])
        end
    end

    return 1 / diff_max
end



function calculate_n(y, λ, n_y)
    n_min = zero(Int32)
    δ = 1 / λ

    for i = 1:(n_y - 1)
        j = i + 1
        y_i = y[i]

        for j = i:n_y
            if (y[j] - y_i) < δ
                n_min = max(n_min, j - i)
            else
                break
            end
        end
    end

    return n_min
end





function mult!(E, sigf_mat :: SigfMat{T}) where T
    Threads.@threads for j in 1:size(E, 2)
        mult!(E, sigf_mat, j)
    end

    return E
end



function mult!(E, sigf_mat :: SigfMat{T}, j) where T
    cumsum_part = zero(T)
    M = sigf_mat.M

    for i = 1:(size(E, 1) - 1)
        cumsum_part += E[i, j]

        val = cumsum_part
        prod_length_i = sigf_mat.prod_lengths[i]

        @turbo for k = 1:prod_length_i
            val += M[k, i] * E[i + k, j]
        end

        E[i, j] = val
    end

    return E
end
        








