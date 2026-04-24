# Aligning Astronomical Images

**Credit**: [_Beroiz, M., Cabral, J. B., & Sanchez, B. (2020)_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract)

## Motivation

Aligning images comes up a lot in astronomy, like for co-adding exposures or timeseries photometry. The problem is that it can be computationally expensive to accomplish this via the traditional plate solving approach where we first need to calculate the WCS coordinates in each frame via a routine like [astrometry.net](https://astrometry.net), and then perform the relevant coordinate transformations from there.

Enter [`astroalign.py`](https://github.com/quatrope/astroalign). This neat Python package sidesteps all of this by directly matching common star patterns between images to build a point-to-point correspondence. This page outlines the Julia reimplementation packaged as [JuliaAstro/Astroalign.jl](https://github.com/JuliaAstro/Astroalign.jl).

## How It Works

[_Beroiz, Cabral, & Sanchez_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract) use the fact that triangles can be uniquely characterised to match sets of three stars (asterisms) between images. This point-to-point correspondence then gives everything needed to compute the affine transformation.

For this implementation they use the invariant ``\mathscr M`` (the pair of two independent ratios of a triangle's side lengths, ``L_i``) to define this unique characterization:

```math
\begin{align}
&\mathscr M = (L_3/L_2,\ L_2/L_1), \\
&\text{where}\ L_3 > L_2 > L_1\;.
\end{align}
```

Astroalign.jl accomplishes this in the following steps:

1. Identify the `N_max` brightest point-like sources in `img_from` and `img_to`.
2. Calculate all triangular asterisms formed from these sources.
3. Build a `2 × 3 × 2 × N` array of candidate triangle-level correspondences
   by matching each from-triangle to its nearest to-triangle in
   the invariant ``\mathscr M`` space. Vertices are assigned via a canonical
   ordering that is invariant under rotation, so the positional correspondence
   between matched triangles is geometrically consistent.
4. Run RANSAC ([Fischler & Bolles, 1981](https://dl.acm.org/doi/10.1145/358669.358692))
   on the triangle matches to robustly identify the largest set of mutually
   consistent correspondences ("inliers").
5. Refine the transformation via the Kabsch/Umeyama least-squares algorithm
   applied to all vertex pairs from all inlier triangle matches.
6. Warp `img_from` to the coordinates of `img_to`.

The rest of this page will show how to align simulated images using this technique.

## Packages

Here is a summary of the packages we will use in this walkthrough:

```@example walkthrough
# Main package and visualization packages
using Astroalign, CairoMakie
using AstroImages: AstroImage, Percent, Zscale, set_cmap!
set_cmap!(:cividis)

# Simulated star fields
using Random: Xoshiro
using PSFModels: gaussian
using LinearAlgebra: I, norm
using Rotations: RotMatrix2
using CoordinateTransformations: LinearMap, Translation
using ImageTransformations: Periodic, warp
```

## Star field generator ✨

We'll start by creating a simulated star field to align on. For simplicity, we'll just create 12 Gaussian point sources placed randomly in a 300 x 300 grid with some noise over the whole image:

```@example walkthrough
# Modified from
# https://github.com/JuliaAstro/PSFModels.jl/blob/main/test/fitting.jl
function generate_model(rng, model, params, inds)
    cartinds = CartesianIndices(inds)
    psf = model.(cartinds; params..., amp = 30_000)
    noise = rand(rng, 1000:3000, size(psf))
    return psf .+ noise
end
```

```@example walkthrough
img_to = let
    # Initial setup
    rng = Xoshiro(1)
    img_size = (1:300, 1:300)
    N_sources = 12
    FWHMs = [rand(rng, 1:10) for _ in 1:N_sources]
    pad = 10 # Minimum space between the stars (in pixels)
    positions_to = rand(rng, 1+pad:pad:300-pad, N_sources, 2)

    # Build stars
    map(zip(eachrow(positions_to), FWHMs)) do ((x, y), fwhm)
        generate_model(rng, gaussian, (; x, y, fwhm), img_size)
    end
# We wrap in an `AstroImage` to enable nice plotting recipes
end |> sum |> AstroImage
```

We'll next create a transformed image that we would like to align back to the original:

```@example walkthrough
# "Truth" values for transformation
const SCALE_0, ROT_0, TRANS_0 = 0.8, π/8, [10, 7]

img_from = let
    tfm = Translation(TRANS_0...) ∘
        LinearMap(RotMatrix2(ROT_0)) ∘
        LinearMap(SCALE_0 * I)

    warp(img_to, tfm, axes(img_to); fillvalue = Periodic())
end |> AstroImage
```

In this particular case, `img_from` is i) **scaled by a factor of 0.8**, ii) **rotated counter-clockwise by 22.5°**, and iii) **translated by [10, 7] pixels** to arrive at `img_to`. We will now show how to align this image and recover these initial transformation parameters with Astroalign.jl.

First, here are some convenience functions that we will use to visualize our results:

```@example walkthrough
# Global colorbar lims
const ZMIN, ZMAX = let
    lims = Percent(99.5).((img_to, img_from))
    minimum(first, lims), maximum(last, lims)
end

set_theme!(;
    Axis = (; aspect = DataAspect(), xticks = LinearTicks(4), yticks = LinearTicks(4)),
    Image = (; colorrange = (ZMIN, ZMAX), colormap = :cividis),
    # For default aperure plots
    Scatter = (;
        cycle = [], # Disable so that `color` is not overriden
        marker = Circle,
        markersize = 36, # Roughly the aperture size used
        markerspace = :data,
        color = :transparent,
        strokewidth = 2,
        strokecolor = :lightgreen,
    ),
)

function plot_pair(img_left, img_right;
    titles = ["img_left", "img_right"],
    colorrange = (ZMIN, ZMAX),
)
    fig = Figure(; size = (600, 300), figure_padding = (0, 0, 0, 0))

    ax_from, p_from = image(fig[1, 1], AstroImage(img_left); colorrange)
    ax_from.title = first(titles)

    ax_to, p_to = image(fig[1, 2], AstroImage(img_right); colorrange)
    ax_to.title = last(titles)
    hideydecorations!(ax_to)

    colsize!(fig.layout, 1, Aspect(1, 1.0))
    colsize!(fig.layout, 2, Aspect(1, 1.0))

    fig
end
```

## Usage

We now use the exported [`align_frame`](@ref) function to align our image:

```@example walkthrough
# Available options
opts_phot = (;
    box_size = (31, 31),
    ap_radius = 18.6,
    min_fwhm = (3, 3),
    nsigma = 1,
    f = Astroalign.PSF(params = (fwhm = (6, 6),)),
    N_max = 10,
    use_fitpos = true,
);
opts_ransac = (; scale = true, ransac_threshold = 3.0);
opts_refinement = (; final_iters = 3, opts_ransac...)

# Align
arr_from_aligned, params_aligned = align_frame(img_from, img_to; opts_phot..., opts_refinement...);

# Visualize
plot_pair(arr_from_aligned, img_to; titles = ["img_from (aligned)", "img_to"])
```

That's it! See the next section for a brief analysis on how well we did.

## Recovered transformation

The transformation object `tfm` returned by [`align_frame`](@ref) defines the mapping `img_from => img_to`:

```@example walkthrough
tfm_aligned = params_aligned.tfm
```

Decomposing it into scale (`S`), rotation (`R`), and translation (`T`) components then gives:

```@example walkthrough
function decompose_tfm(tfm)
    M = tfm.linear
    S = sqrt(M'M)
    R = M * inv(S)
    T = tfm.translation
    return (; S, R, T)
end

# To compare with "truth" values
p_diff(x, x0) = round(100 * (x - x0) / x0; digits = 3)
p_diff(x::AbstractVector, x0::AbstractVector) = round(100 * norm(x - x0) / norm(x0); digits = 3)
print_diff(name, x, x0) = println(
    "$(name) : $(round.(x; digits = 4)) (truth: $(x0), error: $(p_diff(x, x0))%)"
)
```

```@example walkthrough
S, R, T = decompose_tfm(tfm_aligned)

params_tfm = (scale = S[1], rot = atan(R[2, 1], R[1, 1]), trans = T)

print_diff("Scale", params_tfm.scale, SCALE_0)
print_diff("Rotation", rad2deg(params_tfm.rot), rad2deg(ROT_0))
print_diff("Translation", params_tfm.trans, TRANS_0)
```

Taking a look at our RANSAC pass, these final transformation values were determined from 10 out of 35 detected correspondences (28.6 %).

```@example walkthrough
n_inliers = length(params_aligned.inlier_idxs)
n_total   = size(params_aligned.correspondences, 4)
println("RANSAC inliers: $n_inliers / $n_total ($(round(100*n_inliers/n_total; digits = 1))%)")
```

The rest of this document will walk through how this is accomplished behind the scenes, and the different options that we can pass to [`align_frame`](@ref).

## Step 1: Identify control points

This step is done solely on the Photometry.jl side for both our `img_from` and `img_to` images, which Astroalign.jl calls with some reasonable defaults via [`Astroalign._photometry`](@ref).

```@example walkthrough

phot_to, phot_to_params = Astroalign._photometry(img_to; opts_phot...)
phot_from, phot_from_params = Astroalign._photometry(img_from; opts_phot...)
```

This performs source extraction and source characterization of our images, storing the results in the `phot_from_params` and `phot_to_params` named tuples above. Here is a preview of the detected soures in each image:

```@example walkthrough
fig = plot_pair(img_from, img_to; titles = ["img_from", "img_to"])

# Show apertures
scatter!(fig.content[1], phot_from_params.sources.y, phot_from_params.sources.x)
scatter!(fig.content[2], phot_to_params.sources.y, phot_to_params.sources.x)

fig
```

### Source extraction

`astroalign.py` uses [`sep`](https://github.com/quatrope/astroalign/blob/d7463b4ca48fc35f3d86a72343015491cdf20d6a/astroalign.py#L537) under the hood for its source extraction. We use [`Photometry.Detection.extract_sources`](@extref) to pull out the regions around the brightest pixels and [`PSFModels.fit`](@extref) to fit PSF models to each detected source, allowing us to pick out the ones that look like stars (vs. hot pixels, artifacts, etc.). This process is executed by [`Astroalign._get_sources`](@ref).

In the future, additional photometry options may be added.

### Source characterization

[`Photometry.Aperture.photometry`](@extref) automatically computes aperture sums and returns them in a nice table for us. We also pass a function, `Astroalign.PSF`, to compute some PSF statistics for each source and stores them in the table as well.

!!! note
    Some PSF model fits may not converge for especially noisy data. Data cleaning / pre-procesing is outside the scope of this package.

In addition to the usual photometry fields returned, the `aperture_f` field contains a named tuple of PSF information computed by default with the [`Astroalign.PSF()`](@ref) callable:

- `psf_params`: Named tuple of `x` and `y` center, and `fwhm` of fitted PSF relative to its aperture.
- `psf_model`: The best fit PSF model. Uses a [`PSFModels.gaussian`](@extref) by default.
- `psf_data`: The underlying intersection array of the data within the aperture being fit.

The same parameters passed to [`PSFModels.fit`](@extref) can also be passed to [`Astroalign.PSF()`](@ref).

Below is a quick visual check comparing an observed point source with its fitted PSF model:

```@example walkthrough
function inspect_psf(phot; kwargs...)
    psf_data, psf_model = phot.aperture_f.psf_data, phot.aperture_f.psf_model

    println((; phot.xcenter, phot.ycenter))
    println(phot.aperture_f.psf_params)

    obs = AstroImage(psf_data)
    psf = AstroImage(psf_model.(CartesianIndices(psf_data)))

    plot_pair(obs, psf; kwargs...)
end

inspect_psf(first(phot_to); colorrange = (0, 1), titles = ["Data", "Model"])
```

!!! note
    PSF centers are relative to the aperture, while `xcenter` and `ycenter` are relative to the whole image. Astroalign.jl performs the necessary conversions from the former to the latter in [`Astroalign.to_subpixel`](@ref) before reporting the final fitted values.

With out sources identified, we now turn to the next step in the alignment algorithm.

## Step 2: Calculate invariants

This is done internally in [`align_frame`](@ref), but the invariants ``\mathscr M_i`` can also be exposed with [`Astroalign._triangle_invariants`](@ref).

```@example walkthrough
C_to, ℳ_to = Astroalign._triangle_invariants(phot_to)

# This can also be accessed through the named tuple
# returned by `Astroalign.align_frame`.
(; C_from, ℳ_from) = params_aligned
```

Below is a plot comparing the compents of the computed invariants for all control points in our `from` and `to` images. Overlapping regions between the `from` and `to` clouds indicate similar triangles found by Astroalign.jl. Compare to Fig. 1 in [Beroiz et al. (2020)](https://arxiv.org/pdf/1909.02946).

```@example walkthrough
# Use default theme
with_theme() do
    fig, ax, p = scatter(ℳ_to[1, :], ℳ_to[2, :];
        label = "img_to",
    )

    ax.xlabel = "L3/L2"
    ax.ylabel = "L2/L1"

    scatter!(ax, ℳ_from[1, :], ℳ_from[2, :];
        markersize = 20,
        color = :transparent,
        strokewidth = 2,
        strokecolor = :cornflowerblue,
        label = "img_from",
    )

    axislegend()

    fig
end
```

!!! note
    The number of triangle combinations may differ between frames if sources drift towards or off the edge of the frame between images. All that is needed is one matching triangle.

## Step 3: Build candidate correspondences

We next build our list of candidate correspondences in this invariant space via a nearest neighbors search:

```@example walkthrough
correspondences = Astroalign._build_correspondences(C_from, ℳ_from, C_to, ℳ_to)

println("Candidate triangle matches: $(size(correspondences, 4))")
```

## Step 4: RANSAC pass

The largest mutually consistent set of correspondences ("inliers") is found via a RANSAC pass using [JuliaAstro/ConsensusFitting.jl](https://github.com/JuliaAstro/ConsensusFitting.jl) with [`Astroalign._ransac`](@ref):

```@example walkthrough
fwd_tfm_initial, inlier_idxs_initial = Astroalign._ransac(correspondences; opts_ransac...)

println("Initial RANSAC inliers: $(length(inlier_idxs_initial)) / $(size(correspondences, 4))")
```

## Step 5: Refine transformation

The transformation and inlier set from the previous step are successively refined via [`Astroalign._refine_transform`](@ref) using all detected control points, capturing previously missed inliers while dropping incorrect assignments:

```@example walkthrough
tfm, inlier_idxs, point_map = Astroalign._refine_transform(fwd_tfm_initial, inlier_idxs_initial, correspondences; opts_refinement...)
```

For this example, all 10 initial inliers remain after refinement:

```
println("Final RANSAC inliers: $(length(inlier_idxs)) / $(size(correspondences, 4))")

inlier_idxs == inlier_idxs_initial
```

The matched control points in both images are shown below:

```@example walkthrough
fig = plot_pair(img_from, img_to; titles = ["img_from", "img_to"])

# Solution apertures
x_from, y_from, x_to, y_to = (getindex.(f.(point_map), i) for f in (first, last) for i in (1, 2))
strokecolor = Makie.categorical_colors(:tab10, length(x_from))
scatter!(fig.content[1], x_from, y_from; strokecolor)
scatter!(fig.content[2], x_to, y_to; strokecolor)

fig
```

## Step 6: Apply transformation

Once the linear transformation parameters have been finalized in [Step 5](@ref "Step 5: Refine transformation"), we hand it off to [`ImageTransformations.warp`](https://juliaimages.org/ImageTransformations.jl/stable/reference/#ImageTransformations.warp) to perform resampling to align `img_from` with `img_to`:

```@example walkthrough
img_aligned_from = AstroImage(warp(img_from, inv(tfm), axes(img_to)))

plot_pair(img_aligned_from, img_to; titles = ["img_from (aligned)", "img_to"])
```
