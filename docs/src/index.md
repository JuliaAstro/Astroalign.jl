# Astroalign.jl

[badges]

Align astronomical images of point sources. Based on the [`astroalign`](https://github.com/quatrope/astroalign) Python package.

Credit: [_Beroiz, M., Cabral, J. B., & Sanchez, B. (2020)_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract)

!!! warning
    This package is still in the experimental stage. If you notice an issue, please feel free to [let us know](@ref Contributing)!

## Installation

```julia-repl
pkg> add Astroalign
```

## Getting Started

```julia
img_aligned, params = align(img_to, img_from;
    box_size,
    ap_radius,
    min_fwhm = box_size .÷ 5,
    nsigma = 1,
    f = Astroalign.PSF(),
)
```

See the accompanying [Pluto.jl notebook](https://juliaastro.org/Astroalign.jl/notebook.html) for more.

## Contributing

[Issues](https://github.com/JuliaAstro/Astroalign.jl/issues) and [pull requests](https://github.com/JuliaAstro/Astroalign.jl/pulls) welcome.

# API / Reference

```@docs
Astroalign.align
Astroalign.get_sources
Astroalign.triangle_invariants
Astroalign.find_nearest
```
