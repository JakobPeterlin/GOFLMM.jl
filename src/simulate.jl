



struct RandomEffect_Data{T}
    k :: Int
    Z :: Adjoint{T, Matrix{T}}
    D :: Union{T, Matrix{T}}
    bs :: Matrix{T}
    bs2 :: Matrix{T}
    n :: Int32
    groups :: GroupIndexes
end





struct LMM_Data{T}
    m :: Int
    y :: Vector{T}
    y2 :: Vector{T}
    X :: Matrix{T}
    ϵ :: Vector{T}
    β :: Vector{T}
    σ :: T
    N :: Int32
    ran_efs :: Vector{RandomEffect_Data{T}}
end







function group_indexes(N, n, k) 
    refs = (ones(Int32, N))
    refs[1:n] = collect(1:n)
    GroupIndexes(categorical(refs), n, k)
end



function prealloc_ranef(N :: Int32, n :: Int32, D, k)
    T = eltype(D)
    Z = Matrix{T}(undef, N, size(D, 1))'
    bs = Matrix{T}(undef, size(D, 1), n)
    bs2 = Matrix{T}(undef, size(D, 1), n)
    indexes = group_indexes(N, n, size(D, 1))
    return RandomEffect_Data(k, Z, D, bs, bs2, n, indexes)
end



function prealloc_lmm(N, ns, β, σ, Ds, 
    mm = length(β), ks = [size(D, 2) for D in Ds])
    T = eltype(β)
    y = Vector{T}(undef, N)
    y2 = copy(y)
    X = Matrix{T}(undef, N, length(β))
    ϵ = Vector{T}(undef, N)
    ran_efs = RandomEffect_Data{T}[]
    
    for i = 1:length(ns)
        push!(ran_efs, prealloc_ranef(Int32(N), Int32(ns[i]), Ds[i], ks[i]))
    end

    return LMM_Data(mm, y, y2, X, ϵ, β, σ, Int32(N), ran_efs)
end







function update_lmm!(lmm, X_fun!, ran_ef_funs!)
    X_fun!(lmm)
    
    for (i, ran_ef) in enumerate(lmm.ran_efs)
        update_refs(ran_ef, lmm, ran_ef_funs![i][1])
        ran_ef_funs![i][2](lmm, ran_ef)        
    end

    update_y!(lmm)
    
    return lmm
end
        



function update_y!(lmm)
    update_random!(lmm)
    mul!(lmm.y, lmm.X, lmm.β)
    
    @turbo for i in 1:length(lmm.y)
        lmm.y[i] += lmm.ϵ[i]
    end

    for ran_ef in lmm.ran_efs 
        for i = 1:ran_ef.n
            indxs_i = indexes_i(ran_ef, i)
            mul!(view(lmm.y2, indxs_i), ran_ef.Z[:, indxs_i]', ran_ef.bs[:, i])
            
            @turbo for j in 1:length(indxs_i)
                k = indxs_i[j]
                lmm.y[k] += lmm.y2[k]
            end
        end
    end
end




function update_random!(lmm)  
    randn!(lmm.ϵ)
    sc_mult!(lmm.ϵ, lmm.σ)

    for ran_ef in lmm.ran_efs
        randn!(ran_ef.bs2)
        mul!(ran_ef.bs, ran_ef.D, ran_ef.bs2)
    end

    return lmm
end



function update_refs(ran_ef, lmm, ref_fun!)
    ref_fun!(ran_ef, lmm)
    update_groups!(ran_ef.groups, ran_ef.groups.refs, ran_ef.groups.k)
end






## For Fitting




function fit(lmm :: LMM_Data; args ...)
    df = lmm_to_df( lmm )
    formula = lmm_to_formula(lmm)

    return MixedModels.fit(MixedModels.MixedModel, formula, df; args...)
end




fit_REML(lmm :: LMM_Data) = fit(lmm, REML = true)





function fit_pwr(lmm :: LMM_Data, m, ks, args ...)
    df = lmm_to_df( lmm, m, ks)
    formula = lmm_to_formula(lmm)

    return MixedModels.fit(MixedModels.MixedModel, formula, df, args...)
end




col_y() = "y"
col_X(i) = string("X", i)
col_I(i) = string("I", i)
col_Z(i, j) = string("Z", i, "_", j)



function lmm_to_df(lmm)
    return  lmm_to_df!(DataFrame(), lmm)
end



function lmm_to_df!(df, lmm)
    insertcols!(df, (col_y() => lmm.y); copycols = false)

    for i = 1:lmm.m
        insertcols!(df, (col_X(i) => view(lmm.X, :, i)); copycols = false)
    end

    for (i, ran_ef) in enumerate(lmm.ran_efs)
        insert_ran_ef!(df, ran_ef, i)
    end

    return df
end



function insert_ran_ef!(df, ran_ef, i)
    insertcols!(df, (col_I(i) => ran_ef.groups.refs); copycols = false)
    Z = ran_ef.Z

    for j = 1:ran_ef.k
        insertcols!(df, (col_Z(i, j) => view(Z, j, :)); copycols = false)
    end

    return df
end



term(str) = MixedModels.term(str)



function lmm_to_formula(lmm)
    response = term(col_y())
    fixed = term(0) + sum(term(col_X(i)) for i = 1:lmm.m)
    ran_efs = [ran_ef_to_formula(lmm.ran_efs[i], i) for i = 1:length(lmm.ran_efs)] 
    
    return response ~ fixed + sum(ran_efs)
end

 

function ran_ef_to_formula(ran_ef, i)
    formula = term(0)

    for j = 1:ran_ef.k
        formula += term(col_Z(i, j))
    end

    return formula | term(col_I(i))
end
        









## Simulating lmms

function classic_lmm(N, n, β, σ, D, intercept = false)
    X_fun! = intercept ? X_unif_1! : X_unif!
    lmm = prealloc_lmm(N, [n], β, σ, [D])
    update_lmm!(lmm, X_fun!, [(refs_ascending!, Z_X!)])
end




function three_ranefs_lmm(N, n, β, σ, Ds, intercept = false)
    X_fun! = intercept ? X_unif_1! : X_unif!
    lmm = prealloc_lmm(N, [n, n, n], β, σ, Ds)
    ran_ef_funs = [(refs_ascending!, Z_unif!), 
                    (refs_random!, Z_unif!), 
                    (refs_random!, Z_unif!)]
    update_lmm!(lmm, X_fun!, ran_ef_funs)
end






## Simulating power


function missing_square(N, n, β, σ, D, m, k, intercept = false)
    X_fun! = intercept ? Xsq_unif_1! : Xsq_unif!
    lmm = prealloc_lmm(N, [n], β, σ, [D], m, [k])
    update_lmm!(lmm, X_fun!, [(refs_ascending!, Z_X!)])
end




function rok(n, ni, β3, σ, σb, fixed = true, random = true)
    m = fixed ? 4 : 3
    k = random ? 2 : 1
    β = [-1.0, 0.25, 0.5, β3]
    D = [0.5 0.0; 0.0 σb]
    lmm = prealloc_lmm(n * ni, [n], β, σ, [D], m, [k])
    return update_lmm!(lmm, Xsq_unif_1!, [(refs_ascending!, Z_X!)])
end







function rok2(n, ni, β, σ1, σ2, σ3, random = true)
    k = random ? 3 : 2
    D = diagm([σ1, σ2, σ3])
    lmm = prealloc_lmm(n * ni, [n], β, 0.5, [D], 12, [k])
    return update_lmm!(lmm, X_r2!, [(refs_ascending!, Z_r2!)])
end





function rok3(n, ni, β, σ1, σ2, σ3, random = true)
    k = random ? 3 : 2
    D = diagm([σ1, σ2, σ3])
    lmm = prealloc_lmm(n * ni, [n], β, 0.5, [D], 12, [k])
    return update_lmm!(lmm, X_r2!, [(refs_ascending!, Z_r3!)])
end










function jakob(n, β3, σ, σb, fixed = true, random = true, interceptX = true, interceptZ = true)
    m = fixed ? 4 : 3
    k = random ? 3 : 2
    β = [-1.0, 0.25, 0.5, β3]
    D = [
        0.5 0.0 0.0; 
    0.0 0.5 0.0;
    0.0 0.0 σb]
    X_fun = interceptX ? Xsq_unif_1! : Xsq_unif!
    Z_fun = interceptZ ? Zsq_unif_1! : Zsq_unif!
    lmm = prealloc_lmm(n * 10, [n], β, σ, [D], m, [k])
    return update_lmm!(lmm, Xsq_unif_1!, [(refs_ascending!, Zsq_unif_1!)])
end





function jakob2(n, β3, σ, σb, fixed = true, random = true, interceptX = true, interceptZ = true)
    m = fixed ? 4 : 3
    k = random ? 3 : 2
    β = [-1.0, 0.25, 0.5, β3]
    D = [
        0.5 0.0 0.0; 
    0.0 0.5 0.0;
    0.0 0.0 σb]
    X_fun = interceptX ? Xsq_unif_1! : Xsq_unif!
    Z_fun = interceptZ ? Zsin_unif_1! : Zsin_unif!
    lmm = prealloc_lmm(n * 10, [n], β, σ, [D], m, [k])
    return update_lmm!(lmm, X_fun, [(refs_ascending!, Z_fun)])
end




function jakob3(n, β3, σ, σb, fixed = true, random = true, interceptX = true, interceptZ = true)
    m = fixed ? 4 : 3
    k = random ? 3 : 2
    β = [-1.0, 0.25, 0.5, β3]
    D = [
        0.5 0.0 0.0; 
    0.0 1.0 0.0;
    0.0 0.0 σb]
    X_fun = interceptX ? Xsq_unif_1! : Xsq_unif!
    Z_fun = interceptZ ? Z_unif_1! : Z_unif!
    lmm = prealloc_lmm(n * 10, [n], β, σ, [D], m, [k])
    return update_lmm!(lmm, X_fun, [(refs_ascending!, Z_fun)])
end







function jakob4(n, β3, σ, σb, fixed = true, random = true)
    m = fixed ? 3 : 2
    k = random ? 3 : 2
    β = [-1.0, 0.5, β3]
    D = [
        0.5 0.0 0.0; 
    0.0 1.0 0.0;
    0.0 0.0 σb]
    lmm = prealloc_lmm(n * 10, [n], β, σ, [D], m, [k])
    return update_lmm!(lmm, Xsq_unif_1!, [(refs_ascending!, Z_X!)])
end






function jakob5(n, σ, σb1, σb2, random = true, missing_Zz = true)
    k = random ? 4 : 3
    β = [-1.0, 0.5, 1.0]
    D = [
        0.5 0.0 0.0 0.0; 
    0.0 1.0 0.0 0.0;
    0.0 0.0 σb1 0.0;
    0.0 0.0 0.0 σb2]
    lmm = prealloc_lmm(n * 10, [n], β, σ, [D], 3, [k])
    Z_fun = missing_Zz ? Zzx1! : Zzx2!
    return update_lmm!(lmm, Xsq_unif_1!, [(refs_ascending!, Z_fun)])
end







function jakob6(n, β, σb, fixed = true, random = true)
    m = fixed ? 2 : 1
    k = random ? 2 : 1
    β = [1.0; β]
    D = Matrix(Diagonal([0.5, σb]))
    lmm = prealloc_lmm(n * 10, [n], β, 0.5, [D], m, [k])
    return update_lmm!(lmm, X_ort_1!, [(refs_ascending!, Z_X!)])
end






function jakob7(n, β, σb, fixed = true, random = true)
    m = fixed ? 2 : 1
    k = random ? 2 : 1
    β = [1.0; β]
    D = Matrix(Diagonal([0.5, σb]))
    lmm = prealloc_lmm(n * 10, [n], β, 0.5, [D], m, [k])
    return update_lmm!(lmm, X_cent_1!, [(refs_ascending!, Z_X!)])
end














## Simulating matrices and refs

function X_unif!(lmm)
    rand!(lmm.X)
end

function X_unif_1!(lmm)
    rand!(lmm.X)
    lmm.X[:, 1] .= 1
end




function Xsq_unif!(lmm)
    rand!(lmm.X)
    lmm.X[:, end] = lmm.X[:, 2].^2
end



function Xsq_unif_1!(lmm)
    rand!(lmm.X)
    lmm.X[:, 1] .= 1
    lmm.X[:, end] = lmm.X[:, 2].^2
end




function X_ort_1!(lmm)
    rand!(lmm.X)
    lmm.X[:, 1] .= 1.0
    xm = mean(lmm.X[:, 2])
    lmm.X[:, 2] .-= xm
end




function X_cent_1!(lmm)
    rand!(lmm.X)
    lmm.X[:, 1] .= 1.0
    xm = 0.5
    lmm.X[:, 2] .-= xm
end




function Z_X!(lmm, ran_ef)
    k = size(ran_ef.Z, 1)
    copyto!(ran_ef.Z, view(lmm.X, :, 1:k)')
end




function Z_unif!(lmm, ran_ef)
    rand!(ran_ef.Z)
end



function Z_unif_1!(lmm, ran_ef)
    rand!(ran_ef.Z)
    ran_ef.Z[1, :] .= 1.0
end



function Zsq_unif_1!(lmm, ran_ef)
    rand!(ran_ef.Z)
    ran_ef.Z[1, :] .= 1
    ran_ef.Z[end, :] .= ran_ef.Z[2, :] .^2
end




function Zsin_unif_1!(lmm, ran_ef)
    rand!(ran_ef.Z)
    ran_ef.Z[1, :] .= 1
    ran_ef.Z[end, :] .= sin.(5 * ran_ef.Z[2, :])
end





function refs_ascending!(ran_ef, lmm)
    N = lmm.N
    n = ran_ef.n
    D = ceil(N / n)

    for j = 1:N
        ran_ef.groups.refs[j] = floor((j - 1) / D) + 1
    end
end
    



function refs_random!(ran_ef, lmm)
    n = ran_ef.n
    rand!(ran_ef.groups.refs, 1:n)
end
    



function Zzx1!(lmm, ran_ef)
    rand!(ran_ef.Z)
    ran_ef.Z[1, :] .= 1.0
    ran_ef.Z[2, :] = lmm.X[:, 1]
    ran_ef.Z[4, :] = ran_ef.Z[3, :] .^2
end




function Zzx2!(lmm, ran_ef)
    rand!(ran_ef.Z)
    ran_ef.Z[1, :] .= 1.0
    ran_ef.Z[4, :] = lmm.X[:, 1]
    ran_ef.Z[2, :] = ran_ef.Z[3, :] .^2
end








function X_r2!(lmm)
    lmm.X[:, 1] .= 1.0
    rand!(view(lmm.X, :, 2))
    rand!(view(lmm.X, :, 3:12), [.0, 1.0])
end





function Z_r2!(lmm, ran_ef)
    rand!(view(ran_ef.Z, 1:2, :), [0.0, 1.0])
    ran_ef.Z[3, :] = lmm.X[:, end]
end




function Z_r3!(lmm, ran_ef)
    ran_ef.Z[1, :] = lmm.X[:, end]
    rand!(view(ran_ef.Z, 2:3, :), [0.0, 1.0])
end
