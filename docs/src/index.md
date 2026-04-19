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

## Quickstart

The following will align `img_from` onto `img_to`:

```julia
using Astroalign

img_aligned, params_aligned = align_frame(img_from, img_to)
```

!!! info
    See the [Walkthrough](@ref "Aligning Astronomical Images") page for more on supported keywords and a detailed step-by-step walkthrough of the algorithm.

## Contributing

[Issues](https://github.com/JuliaAstro/Astroalign.jl/issues) and [pull requests](https://github.com/JuliaAstro/Astroalign.jl/pulls) welcome.
