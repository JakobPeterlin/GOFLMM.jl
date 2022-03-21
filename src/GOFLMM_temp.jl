



module GOFLMM_temp

    using LinearAlgebra, LoopVectorization, Random, MixedModels, DataFrames, CategoricalArrays, Polyester, Statistics, ProgressBars, Plots
    
    include("linalg.jl")
    include("random_effects.jl")
    include("matrixA.jl")
    include("lmm.jl")
    include("simulate.jl")
    include("auxilia.jl")
    include("sig_diff_mat.jl")

    include("processes.jl")
    include("gof.jl")
    include("blockQL.jl")
    include("repeated_simulations.jl")
    include("plotting.jl")
end
