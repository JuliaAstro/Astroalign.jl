# Home

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliaastro.org/Astroalign/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaastro.org/Astroalign.jl/dev)

[![CI](https://github.com/JuliaAstro/Astroalign.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaAstro/Astroalign.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JuliaAstro/Astroalign.jl/graph/badge.svg)](https://codecov.io/gh/JuliaAstro/Astroalign.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Align astronomical images of point sources. Based on the [`astroalign`](https://github.com/quatrope/astroalign) Python package.

Credit: [_Beroiz, M., Cabral, J. B., & Sanchez, B. (2020)_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract)

!!! warning
    This package is still in the experimental stage. If you notice an issue, please feel free to [let us know](@ref Contributing)!

## Installation

```julia-repl
pkg> add Astroalign
```

## Getting Started

The following will align `img_from` onto `img_to`:

```julia
using Astroalign

img_aligned, params = align_frame(img_to, img_from)
```

!!! info
    See the accompanying [Pluto.jl notebook](https://juliaastro.org/Astroalign.jl/notebook.html) for more on supported keywords and additional analysis.

## Contributing

[Issues](https://github.com/JuliaAstro/Astroalign.jl/issues) and [pull requests](https://github.com/JuliaAstro/Astroalign.jl/pulls) welcome.

# API / Reference

```@docs
Astroalign.align_frame
Astroalign.get_sources
Astroalign.find_nearest
Astroalign.triangle_invariants
```
