using GOFLMM_temp
using Documenter

DocMeta.setdocmeta!(GOFLMM_temp, :DocTestSetup, :(using GOFLMM_temp); recursive = true)

makedocs(;
    modules = [GOFLMM_temp],
    authors = "JakobPeterlin",
    repo = "https://github.com/JakobPeterlin/GOFLMM_temp.jl/blob/{commit}{path}#{line}",
    sitename = "GOFLMM_temp.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JakobPeterlin.github.io/GOFLMM_temp.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/JakobPeterlin/GOFLMM_temp.jl")
