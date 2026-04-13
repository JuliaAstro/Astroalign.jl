using Documenter, DocumenterInterLinks
using Documenter.Remotes: GitHub
using Astroalign
using AstroImages
using CairoMakie

CairoMakie.activate!(type = "png", px_per_unit = 2)

links = InterLinks(
    "Photometry" => "https://juliaastro.org/Photometry/stable/",
    "PSFModels" => "https://juliaastro.org/PSFModels/stable/",
)

makedocs(
    modules = [Astroalign],
    authors = "Ian Weaver <weaveric@gmail.com>",
    repo = GitHub("JuliaAstro/Astroalign.jl"),
    sitename = "Astroalign.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://juliaastro.org/Astroalign/stable/",
    ),
    plugins = [links],
    pages = [
        "Home" => "index.md",
        "Walkthrough" => "walkthrough.md",
    ],
    warnonly = [:missing_docs]
)

deploydocs(;
    repo = "github.com/JuliaAstro/Astroalign.jl",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#"], # Restrict to minor releases
)
