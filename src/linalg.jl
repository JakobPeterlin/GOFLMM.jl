


function sub!(Y, X) 
    for j = 1:size(Y, 2)
        @turbo for i = 1:size(Y, 1)
            Y[i, j] -= X[i, j]
        end
    end

    return Y
end




function add!(Y, X) 
    for j = 1:size(Y, 2)
        @turbo for i = 1:size(Y, 1)
            Y[i, j] += X[i, j]
        end
    end

    return Y
end






function sc_mult!(v, C)
    LoopVectorization.@turbo for i = 1:length(v)
        v[i] *= C
    end

    return v
end
