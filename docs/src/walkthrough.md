```@meta
CurrentModule = Astroalign
```

# Aligning Astronomical Images

**Credit**: [_Beroiz, M., Cabral, J. B., & Sanchez, B. (2020)_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract)

## Motivation

Aligning images comes up a lot in astronomy, like for co-adding exposures or timeseries photometry. The problem is that it can be computationally expensive to accomplish this via the traditional plate solving approach where we first need to calculate the WCS coordinates in each frame via a routine like [astrometry.net](https://astrometry.net), and then perform the relevant coordinate transformations from there.

Enter [`astroalign.py`](https://github.com/quatrope/astroalign). This neat Python package sidesteps all of this by directly matching common star patterns between images to build a point-to-point correspondence. This page outlines the Julia reimplementation packaged as [JuliaAstro/Astroalign.jl](https://github.com/JuliaAstro/Astroalign.jl).

```@setup align_example
using Astroalign, AstroImages, PSFModels, Rotations, Photometry,
      ImageTransformations, CoordinateTransformations, LinearAlgebra, Random
using CairoMakie
using ConsensusFitting: ransac

CairoMakie.activate!(type = "png", px_per_unit = 2)

# ── "Truth" transformation parameters ────────────────────────────────────────
const SCALE_0, ROT_0, TRANS_0 = 0.8, π/8, [10, 7]
const N_sources = 12

# Fixed seed — star field 1 from the original notebook
const RNG = Xoshiro(1)
const FWHMS = [rand(RNG, 1:10) for _ in 1:N_sources]
const img_size = (1:300, 1:300)

# Modified from PSFModels.jl/test/fitting.jl
function generate_model(rng, model, params, inds)
    psf   = model.(CartesianIndices(inds); params..., amp = 30_000)
    noise = rand(rng, 1000:3000, size(psf))
    return psf .+ noise
end

# ── Generate the reference image ─────────────────────────────────────────────
img_to = let
    pad = 10
    positions = rand(RNG, 1+pad:pad:300-pad, N_sources, 2)
    imgs = map(zip(eachrow(positions), FWHMS)) do ((x, y), fwhm)
        generate_model(RNG, gaussian, (; x, y, fwhm), img_size)
    end
    sum(imgs)
end |> AstroImage;

# ── Apply the "truth" transformation to produce img_from ─────────────────────
const tfm_fwd_0 = Translation(TRANS_0...) ∘
    LinearMap(RotMatrix2(ROT_0)) ∘
    LinearMap(SCALE_0 * I)

img_from = warp(img_to, tfm_fwd_0, axes(img_to);
               fillvalue = ImageTransformations.Periodic()) |> AstroImage;

# ── Shared colour-scale limits ────────────────────────────────────────────────
const ZMIN, ZMAX = let
    lims = Percent(99.5).((img_to, img_from))
    minimum(first, lims), maximum(last, lims)
end

# ── Utility helpers ───────────────────────────────────────────────────────────
function decompose_tfm(tfm)
    M = tfm.linear
    S = sqrt(M'M)
    R = M * inv(S)
    T = tfm.translation
    return (; S, R, T)
end

p_diff(x, x0) = round(100 * (x - x0) / x0; digits = 3)
p_diff(x::AbstractVector, x0::AbstractVector) =
    round(100 * norm(x - x0) / norm(x0); digits = 3)

# ── CairoMakie plot helpers ───────────────────────────────────────────────────
function show_image!(ax, img; clims = (ZMIN, ZMAX))
    d = parent(img)
    n1, n2 = size(d)
    heatmap!(ax, 1:n1, 1:n2, d'; colormap = :cividis, colorrange = clims)
end

function plot_pair(img_l, img_r; titles = ["Left", "Right"],
                   srcpts_l = nothing, srcpts_r = nothing,
                   src_color_l = :lime, src_color_r = :lime)
    fig = Figure(size = (780, 360))
    ax_l = Axis(fig[1, 1]; title = titles[1],
                xlabel = "X (pixels)", ylabel = "Y (pixels)", aspect = DataAspect())
    ax_r = Axis(fig[1, 2]; title = titles[2],
                xlabel = "X (pixels)", aspect = DataAspect())
    show_image!(ax_l, img_l')
    show_image!(ax_r, img_r')
    if !isnothing(srcpts_l)
        scatter!(ax_l, srcpts_l[:, 1], srcpts_l[:, 2];
                 color = src_color_l, markersize = 8, strokecolor = :black, strokewidth = 0.5)
    end
    if !isnothing(srcpts_r)
        scatter!(ax_r, srcpts_r[:, 1], srcpts_r[:, 2];
                 color = src_color_r, markersize = 8, strokecolor = :black, strokewidth = 0.5)
    end
    return fig
end

# ── RANSAC step helpers (expose internal pipeline) ────────────────────────────
function step4(correspondences; scale = false, ransac_threshold = 3.0)
    fittingfn = scale ? Astroalign._fit_minimal_similarity_triangle :
                        Astroalign._fit_minimal_rigid_triangle
    ransac(correspondences, fittingfn, Astroalign._triangle_distfn, 1, ransac_threshold)
end

function step_5(correspondences, fwd_tfm_0, inlier_idxs_0;
                scale = false, ransac_threshold = 3.0)
    fwd_tfm     = fwd_tfm_0
    inlier_idxs = inlier_idxs_0
    for _ in 1:3
        isempty(inlier_idxs) && break
        pts_from = reshape(correspondences[:, :, 1, inlier_idxs], 2, :)
        pts_to   = reshape(correspondences[:, :, 2, inlier_idxs], 2, :)
        new_fwd  = kabsch(pts_from => pts_to; scale)
        new_idxs, _ = Astroalign._triangle_distfn([new_fwd], correspondences, ransac_threshold)
        isempty(new_idxs) && break
        fwd_tfm, inlier_idxs = new_fwd, new_idxs
    end
    point_map = mapreduce(vcat, inlier_idxs) do i
        [correspondences[:, v, 1, i] => correspondences[:, v, 2, i] for v in 1:3]
    end
    return unique(point_map), inv(fwd_tfm)
end
```

## Usage

Here is a brief usage example aligning `img_from` onto `img_to`.
In this example, `img_from` is `img_to` scaled by a factor of 0.8, rotated counter-clockwise by 22.5°, and translated by [10, 7] pixels.

```@example align_example
arr_from_aligned, params_aligned = align_frame(img_from, img_to; scale = true)
nothing # hide
```

Plotting the input pair and the aligned result:

```@example align_example
plot_pair(img_from, img_to; titles = ["img_from", "img_to"])
```

```@example align_example
plot_pair(AstroImage(arr_from_aligned), img_to; titles = ["img_from (aligned)", "img_to"])
```

That's it! The rest of this page walks through how this works behind the scenes and illustrates some knobs you can turn.

### Recovered transformation

The transformation object `tfm` returned by [`align_frame`](@ref) defines the mapping `img_from => img_to`:

```@example align_example
tfm_aligned = params_aligned.tfm
```

Decomposing it into scale (`S`), rotation (`R`), and translation (`T`) components:

```@example align_example
S, R, T = decompose_tfm(tfm_aligned)
params_tfm = (scale = S[1], rot = atan(R[2, 1], R[1, 1]), trans = T)
println("Scale       : $(round(params_tfm.scale; digits=4))  (truth: $SCALE_0, error: $(p_diff(params_tfm.scale, SCALE_0))%)")
println("Rotation    : $(round(rad2deg(params_tfm.rot); digits=3))°  (truth: $(round(rad2deg(ROT_0); digits=1))°)")
println("Translation : $(round.(params_tfm.trans; digits=2))  (truth: $TRANS_0, error: $(p_diff(params_tfm.trans, Float64.(TRANS_0)))%)")
```

Taking a look at our RANSAC pass, these final transformation values were determined from 10 out of 35 detected correspondences (28.6 %).

```@example align_example
n_inliers = length(params_aligned.inlier_idxs)
n_total   = size(params_aligned.correspondences, 4)
println("RANSAC inliers: $n_inliers / $n_total ($(round(100n_inliers/n_total; digits=1))%)")
```

## How It Works

[_Beroiz, Cabral, & Sanchez_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract) use the fact that triangles can be uniquely characterised to match sets of three stars (asterisms) between images. This point-to-point correspondence then gives everything needed to compute the affine transformation.

For this implementation they use the invariant ``\mathscr M`` (the pair of two independent ratios of a triangle's side lengths, ``L_i``) to define this unique characterisation:

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

## Details

The following sections walk through each step in detail using the same star fields shown in the [Usage](@ref) example above.

### Step 1: Identify control points

#### Source extraction

`astroalign.py` uses [`sep`](https://github.com/quatrope/astroalign/blob/d7463b4ca48fc35f3d86a72343015491cdf20d6a/astroalign.py#L537) under the hood for its source extraction. We use [`Photometry.extract_sources`](https://juliaastro.org/Photometry/stable/detection/#Photometry.Detection.extract_sources) to pull out the regions around the brightest pixels and [`PSFModels.fit`](https://juliaastro.org/PSFModels/stable/api/#PSFModels.fit) to fit PSF models to each detected source, allowing us to pick out the ones that look like stars (vs. hot pixels, artifacts, etc.). In the future additional photometry options may be added. This process is executed by the `Astroalign.get_sources` function.

```@example align_example
sources_to, subt_to, _ = get_sources(img_to)
```

#### Source characterisation

[`Photometry.photometry`](https://juliaastro.org/Photometry/stable/apertures/#Photometry.Aperture.photometry) automatically computes aperture sums and returns them in a nice table for us. We also pass a function, `Astroalign.PSF`, to compute some PSF statistics for each source and stores them in the table as well.

!!! note
	Some PSF model fits may not converge for really noisy sources.

```@example align_example
box_size  = Astroalign._compute_box_size(img_to)
ap_radius = 0.6 * first(box_size)

aps_to  = CircularAperture.(sources_to.y, sources_to.x, ap_radius)
phot_to = let
    phot = photometry(aps_to, subt_to; f = Astroalign.PSF())
    Astroalign.to_subpixel(phot, aps_to)
end
sort!(phot_to; by = x -> hypot(x.aperture_f.psf_params.fwhm...), rev = true)
phot_to
```

In addition to the usual [`Photometry.photometry`](https://juliaastro.org/Photometry/stable/apertures/#Photometry.Aperture.photometry) fields returned, the `aperture_f` field contains a named tuple of PSF information computed by default with the `Astroalign.PSF()` callable:

* `psf_params`: Named tuple of `x` and `y` center, and `fwhm` of fitted PSF relative to its aperture.
* `psf_model`: The best fit PSF model. Uses a `gaussian` by default.
* `psf_data`: The underlying intersection array of the data within the aperture being fit.

The same parameters passed to `Photometry.fit` can also be passed to `Astroalign.PSF()`.

Below is a quick visual check comparing an observed point source with its fitted PSF model (source 1):

```@example align_example
let
    psf_data  = phot_to[1].aperture_f.psf_data
    psf_model = phot_to[1].aperture_f.psf_model
    model_arr = psf_model.(CartesianIndices(psf_data))

    fig = Figure(size = (550, 240))
    ax1 = Axis(fig[1, 1]; title = "Observed (normalised)", aspect = DataAspect())
    ax2 = Axis(fig[1, 2]; title = "Fitted PSF model",      aspect = DataAspect())
    n1, n2 = size(psf_data)
    heatmap!(ax1, 1:n1, 1:n2, psf_data';  colormap = :magma)
    heatmap!(ax2, 1:n1, 1:n2, model_arr'; colormap = :magma)
    fig
end
```

!!! note
	PSF centers are relative to the aperture, while `xcenter` and `ycenter` are relative to the whole image. Astroalign.jl performs the necessary conversions from the former to the latter in `Astroalign.to_subpixel` before reporting the final fitted values.

### Step 2: Calculate invariants

This is done internally in [`align_frame`](@ref), but the invariants ``\mathscr M_i`` can also be exposed with [`triangle_invariants`](@ref). Below is a plot comparing the compents of the computed invariants for all control points in our `from` and `to` images. Overlapping regions between the `from` and `to` clouds indicate similar triangles found by Astroalign.jl. Compare to Fig. 1 in [Beroiz et al. (2020)](https://arxiv.org/pdf/1909.02946).

```@example align_example
C_to,   ℳ_to   = triangle_invariants(phot_to)

# Obtain from-image invariants from the cached params returned by align_frame
(; C_from, ℳ_from) = params_aligned

fig = Figure(size = (500, 440))
ax  = Axis(fig[1, 1]; xlabel = "L₃/L₂", ylabel = "L₂/L₁",
           title = "Triangle invariants")
scatter!(ax, ℳ_to[1, :],   ℳ_to[2, :];   label = "img_to",   markersize = 6)
scatter!(ax, ℳ_from[1, :], ℳ_from[2, :]; label = "img_from",
         marker = :circle, markersize = 10, strokecolor = :dodgerblue,
         strokewidth = 1.5, color = (:dodgerblue, 0))
axislegend(ax; position = :rb)
fig
```

!!! note
    The number of triangle combinations may differ between frames if sources drift towards or off the edge of the frame between images. All that is needed is one matching triangle.

### Step 3: Build candidate correspondences

We next build our list of candidate correspondences in this invariant space via a nearest neighbors search.

```@example align_example
correspondences = Astroalign._build_correspondences(C_from, ℳ_from, C_to, ℳ_to)
println("Candidate triangle matches: $(size(correspondences, 4))")
```

### Step 4: RANSAC pass

The largest mutually consistent set of correspondences ("inliers") is found via a RANSAC pass using [JuliaAstro/ConsensusFitting.jl](https://github.com/JuliaAstro/ConsensusFitting.jl):

```@example align_example
fwd_tfm_initial, inlier_idxs_initial = step4(correspondences;
    scale = true, ransac_threshold = 3.0)
println("Initial RANSAC inliers: $(length(inlier_idxs_initial)) / $(size(correspondences, 4))")
```

### Step 5: Refine transformation

The transformation and inlier set from the previous step are successively refined using all detected control points, capturing previously missed inliers while dropping incorrect assignments:

```@example align_example
point_map, tfm = step_5(correspondences, fwd_tfm_initial, inlier_idxs_initial;
    scale = true, ransac_threshold = 3.0)
println("Final matched control-point pairs: $(length(point_map))")
```

The matched control points in both images are shown below:

```@example align_example
let
    pts_from = reduce(hcat, [p.first  for p in point_map])'
    pts_to   = reduce(hcat, [p.second for p in point_map])'
    _colors = [cgrad(:magma)[z] for z in range(0, 1, length = size(pts_from, 1))]

    fig = Figure(size = (780, 360))
    ax_l = Axis(fig[1, 1]; title = "img_from — matched sources",
                xlabel = "X (pixels)", ylabel = "Y (pixels)", aspect = DataAspect())
    ax_r = Axis(fig[1, 2]; title = "img_to — matched sources",
                xlabel = "X (pixels)", aspect = DataAspect())
    show_image!(ax_l, img_from')
    show_image!(ax_r, img_to')
    scatter!(ax_l, pts_from[:, 1], pts_from[:, 2];
    # scatter!(ax_l, pts_from[1, :], pts_from[2, :];
             strokecolor = _colors,
             color = :transparent, markersize = 40, strokewidth = 4)
    scatter!(ax_r, pts_to[:, 1],   pts_to[:, 2];
    # scatter!(ax_r, pts_to[1, :],   pts_to[2, :];
             strokecolor = _colors,
             color = :transparent, markersize = 40, strokewidth = 4)
    fig
end
```

### Step 6: Apply transformation

Once the linear transformation parameters have been finalized in step 5, we hand it off to [ImageTransformations.jl](https://github.com/JuliaImages/ImageTransformations.jl) to perform resampling to align `img_from` with `img_to`:

```@example align_example
img_aligned_from = AstroImage(warp(img_from, tfm, axes(img_to)))
plot_pair(img_aligned_from, img_to; titles = ["img_from (aligned)", "img_to"])
```

This should match the result returned by [`align_frame`](@ref) in the [Usage](@ref Usage) section above.
