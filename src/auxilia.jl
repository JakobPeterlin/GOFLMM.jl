


function add!(v, u)
    @assert size(v) == size(u) "Sizes do not match!"
    @turbo for i = 1:length(v)
        v[i] += u[i]
    end

    return v
end








function calculate_Mi!(Mi, M_temp, eig, l_temp)
    @turbo for i = 1:length(eig.values)
        l_temp[i, i] = - eig.values[i] / (1 + eig.values[i])
    end

    mul!(M_temp, l_temp, eig.vectors')
    mul!(Mi, eig.vectors, M_temp)
end




function calculate_Mi05!(Mi05, M_temp, eig, l_temp)
    @turbo for i = 1:length(eig.values)
        l_temp[i, i] =  sqrt(1 / (1 + eig.values[i])) - 1
    end

    mul!(M_temp, l_temp, eig.vectors')
    mul!(Mi05, eig.vectors, M_temp)
end








"Solve the sqrt system for calclulating V_i^{-1/2}"
function solve_sqrt_Vi!(Mi, Mi05, M_temp, l_temp :: Diagonal{T, Vector{T}}, R, D) where T
    mul!(Mi05, D, R')
    mul!(Mi, R, Mi05)

    eig = eigen(Symmetric(Mi))

    calculate_Mi!(Mi, M_temp, eig, l_temp)
    calculate_Mi05!(Mi05, M_temp, eig, l_temp)
end
    
    

