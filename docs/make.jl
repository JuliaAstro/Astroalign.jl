using Revise
Revise.revise()

using Documenter, DocumenterInterLinks
using Documenter.Remotes: GitHub
using Astroalign
using AstroImages

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
        prettyurls = true,
        canonical = "https://juliaastro.org/Astroalign/stable/",
    ),
    plugins = [links],
)
