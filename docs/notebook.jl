### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ d8d4c414-64a0-11f0-15a3-0d566872a687
begin
	import Pkg
	Pkg.resolve()
	Pkg.activate(Base.current_project())
	Pkg.instantiate()
	
	using Revise
	
	using Astroalign, AstroImages, PlutoPlotly, PlutoUI, PSFModels, Rotations, Photometry, ImageTransformations, CoordinateTransformations
	
	AstroImages.set_cmap!(:cividis)

	Pkg.status()
end

# ╔═╡ d97c367c-4db1-4dd0-8066-3f12e08d2f01
using Random

# ╔═╡ 9e130a37-1073-4d0f-860a-0ec8d164dde1
md"""
# 📐 Aligning astronomical images

Like in the days of overhead transparencies. Companion notebook to [Astroalign.jl](https://github.com/JuliaAstro/Astroalign.jl).

Credit: [_Beroiz, M., Cabral, J. B., & Sanchez, B. (2020)_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract)
"""

# ╔═╡ fa1180d4-c1ea-4a1b-8476-0e8d185d5622
md"""
## Motivation 🤸

Aligning images comes up a lot in astronomy, like for co-adding exposures or trying to do some timeseries photometry. The problem is that it can be computationally expensive to accomplish this via the traditional plate solving approach where we first need to calculate the WCS coordinates in each frame via a routine like [astrometry.net](https://astrometry.net/), and then perform the relevant coordinate transformations from there.

Enter [`astroalign.py`](https://github.com/quatrope/astroalign). This really neat Python package sidesteps all of this by directly matching common star patterns between images to build this point-to-point correspondence. This notebook is an experiment to re-implement its core functionality in Julia, which we package as [JuliaAstro/Astroalign.jl](https://github.com/JuliaAstro/Astroalign.jl), (final name and package location tbd).
"""

# ╔═╡ 40c14093-3806-401f-aedf-f6435f785eb4
md"""
### Usage

Here is a brief usage example aligning `img_from` onto `img_to` with the exported `align` function from Astroalign.jl. Click the button below to generate a new star / galaxy field.

"""

# ╔═╡ 5c4155fa-92bd-4260-a9d4-cea9dc5f3d93
@bind seed confirm(Slider(1:30; show_value=true); label = "Confirm")

# ╔═╡ b51e47f6-af8e-478a-a716-af74e33c9e99
md"""
In this particular case, `img_from` is rotated clockwise, and shifted vertically upwards and horizontally to the left relative to `img_to` in the above plot. Let's fix it.
"""

# ╔═╡ a1cb22fc-e956-4cf7-aafc-0168da23e556
md"""
For even more control, each step of the alignment process has an associated API that can be used from Astroalign.jl, along with additional parameters returned by `Astroalign.align_frame`, which we show in the rest of this notebook.
"""

# ╔═╡ b8323ad8-c26b-4cc7-9891-caa05c128fb1
md"""
## Load images 📷

These are the images shown in the [Usage](#Usage) section above. Throughout this notebook, we will use the convention that we transform `from` the second image `to` the first image.
"""

# ╔═╡ dc01eaaa-f1d0-4bc6-884f-778d848918c6
const N_sources = 10

# ╔═╡ 78c0bf28-bb96-4aea-8bf5-5929ef45adc1
img_size = (1:300, 1:300)

# ╔═╡ eff56f6e-ab01-4371-a75f-f44bdde7cfd6
md"""
### Generate some fake data

For simplicity, we'll just create $(N_sources) Gaussian point sources in a $(length(first(img_size))) x $(length(last(img_size))) grid with some noise over the whole image. We can then check our fitted values against these "truth" values at the end.
"""

# ╔═╡ 0ae46a86-dd86-4092-9d34-05f643ec08af
begin
	rng = Xoshiro(seed)
	# fwhms = [(rand(rng, 1:20), rand(rng, 1:20))
	# 		 for _ in 1:N_sources]
	fwhms = [(rand(rng, 1:20), rand(rng, 1:20)) for _ in 1:N_sources]
	positions_to = rand(rng, 30:12:240, N_sources, 2)
end;

# ╔═╡ f7639401-1fc9-4cb1-824c-4335a4bb8b25
# Modified from
# https://github.com/JuliaAstro/PSFModels.jl/blob/main/test/fitting.jl
function generate_model(rng, model, params, inds)
	cartinds = CartesianIndices(inds)
	psf = model.(cartinds; params..., amp = 30_000)
    return psf .+ rand(rng, 1000:3000, size(psf))
end

# ╔═╡ 95531bde-8386-4d51-8c83-ffb796a41e90
img_to = map(zip(eachrow(positions_to), fwhms)) do ((x, y), fwhm)
	generate_model(rng, gaussian, (; x, y, fwhm), img_size)
end |> sum |> AstroImage;

# ╔═╡ 5882adec-7591-4d93-98e2-efb81496c54d
img_from = let
	tfm = Translation(80, -120) ∘ LinearMap(RotMatrix2(π/8))
	warp(img_to, tfm, axes(img_to);
		 fillvalue = ImageTransformations.Periodic(),
	)
end |> AstroImage;

# ╔═╡ 445a0d35-2b49-42cc-8529-176778b0e090
arr_aligned_from, align_params = align_frame(img_to, img_from;
	# box_size,
	# ap_radius,
	# min_fwhm = box_size .÷ 5,
	# nsigma = 1,
	# f = Astroalign.PSF(),
);

# ╔═╡ a2ed7b77-1277-41a3-8c29-a9814b124d09
md"""
## Align ✨
"""

# ╔═╡ 2bc269e1-dbe3-4c68-9a30-8c6054bc3a82
md"""
### Detect

`astroalign.py` uses [`sep`](https://github.com/quatrope/astroalign/blob/d7463b4ca48fc35f3d86a72343015491cdf20d6a/astroalign.py#L537) under the hood for its source extraction, so we'll use a combination of [`Photometry.extract_sources`](https://juliaastro.org/Photometry/stable/detection/#Photometry.Detection.extract_sources) to pull out the regions around the brightest pixels, and [`PSFModels.fit`](https://juliaastro.org/PSFModels/stable/api/#PSFModels.fit) to just pick out the ones that look like stars (vs. hot pixels, artifacts, etc.).

!!! todo
	Explore other point source detection algorithms in Julia.
"""

# ╔═╡ fe518d92-fbfd-4d6f-ba71-0b7b23a73fd7
md"""
#### Source extraction

Starting with `Astroalign.get_sources`, we get the following candidate `sources` in our first image:
"""

# ╔═╡ 00116e4a-8d9d-46dd-b09f-005a19ddf4ee
img = img_to;

# ╔═╡ fb0efcf4-26d4-4554-a5cf-b1136f5a6c17
# Sources, background subtracted image, background
# Guard against extraneous matches by only taking first 10
sources, subt, errs = get_sources(img);

# ╔═╡ 8afa31f0-ee57-4628-bedf-dd2b79faef72
box_size = Astroalign._compute_box_size(img)

# ╔═╡ b3c689bd-1778-4568-96b9-869f2e6a83c0
ap_radius = 0.6 * first(box_size)

# ╔═╡ 07abbeb9-15a4-4086-86ca-093e5475c0db
aps = CircularAperture.(sources.y, sources.x, ap_radius)

# ╔═╡ c6ef3b26-ccc1-401b-ba9b-88276d4c5067
md"""
Since these are just locations of the brightest points in our image, some could be hot pixels or other artifacts. To address this, we can next filter them out by fitting a PSF to each source and only taking ones that meet a minimum FWHM.
"""

# ╔═╡ b0ad71b1-3a3c-481b-a08e-2ee558e8e1c5
md"""
#### Source characterization

[`Photometry.photometry`](https://juliaastro.org/Photometry/stable/apertures/#Photometry.Aperture.photometry) automatically computes aperture sums and returns them in a nice table for us. We use a [slightly modified version](https://github.com/JuliaAstro/Photometry.jl/pull/74) in Astroalign.jl to also compute some PSF statistics for each source and stores them in `phot`.

!!! note
	Some PSF models will not converge, but I think that is to be expected for really noisy sources and is probably fine since they won't be included in the final filtered list anyway.
"""

# ╔═╡ 4e1c0615-d26d-4147-a096-d20940b8046a
phot_to = let
	phot = photometry(aps, subt; f = Astroalign.PSF())
	phot = Astroalign.to_subpixel(phot, aps)
	sort!(phot; by = x -> hypot(x.aperture_f.psf_params.fwhm...), rev = true)
end

# ╔═╡ fcb02cf0-4fb5-4e31-bab9-d19a0755def9
md"""
In addition to the usual [`Photometry.photometry`](https://juliaastro.org/Photometry/stable/apertures/#Photometry.Aperture.photometry) fields returned, the `aperture_f` field contains a named tuple of computed PSF information by default with the `Astroalign.PSF()` callable:

* `psf_P`: Named tuple of `x` and `y` center, and `fwhm` of fitted PSF relative to its aperture.
* `psf_model`: The best fit PSF model. Uses a `gaussian` by default.
* `psf_data`: The underlying intersection array of the data within the aperture being  fit.

The same parameters passed to `Photometry.fit` can also be passed to `Astroalign.PSF()`.
"""

# ╔═╡ 9109a7a0-4a37-4dca-a923-16a9302556ee
md"""
Below is a quick visual check that compares our observed point source with its returned fitted PSF model:
"""

# ╔═╡ 3da14f39-9fad-412e-824b-c3db190700aa
@bind i Slider(eachindex(phot_to); show_value=x -> "Source $(x)")

# ╔═╡ 0083d7bb-07f2-45e6-b4f8-44099ff1a0bf
# And "truth" fwhms for comparison
sort(fwhms; by = x -> hypot(x...), rev = true)

# ╔═╡ 35befaff-e36c-4741-b28f-3589afe596cd
function inspect_psf(phot)
	psf_data, psf_model = phot.aperture_f.psf_data, phot.aperture_f.psf_model
	@debug (; phot.xcenter, phot.ycenter)
	@debug phot.aperture_f.psf_params
	return AstroImage(psf_data), imview(psf_model.(CartesianIndices(psf_data)))
end

# ╔═╡ 68139ad3-cf00-4286-b9eb-a435dd20aca2
inspect_psf(phot_to[i])

# ╔═╡ c73692f2-178d-4b35-badc-e9e682551989
md"""
Looks to be fitting alright! From here, results can be sorted and filtered as needed. By default, Astrolign.jl sorts from largest to smallest FWHM. We will just use all $(N_sources) sources here to help fill out our search space in the final alignment step next.
"""

# ╔═╡ 0603752b-5fcc-4e14-ae41-292cc49c6711
md"""
## Match 👥

This is the secret sauce: _Beroiz, Cabral, & Sanchez_ use the fact that triangles can be uniquely characterized to match sets of three stars (asterisms) between images. This point-to-point correspondence then gives us everything we need to compute the affine transformation between them.

For this implementation, they use the invariant ``\mathscr M`` (the pair of two independent ratios of a triangle's side lengths, ``L_i``) to define this unique characterization:

```math
\begin{align}
&\mathscr M = (L_3/L_2, L_2/L_1), \\

&\text{where}\ L_3 > L_2 > L_1\quad.
\end{align}
```

Astroalign.jl accomplishes this in the following steps:

1. Identify up to ``N`` control points in each image (could be less depending on viewing conditions).
1. Calculate the invariant ``\mathscr M`` for all ``N\choose{3}`` possible combinations of triangles made up of these control points.
1. Calculate the pairwise distances between them and choose the closest pair. This will be our point-to-point correspondence.
1. Return the corresponding affine transformation matrix resulting from this matched pair.
"""

# ╔═╡ 0d4ce3b5-665a-4cc8-8884-90600e99f6ba
md"""
!!! todo
	Add graceful handling of duplicate matches from, e.g., hot pixels that managed to sneak through.
"""

# ╔═╡ f2720f8e-8df1-4cfe-a21c-59855e646106
md"""
### Step 1: Identify control points

This is performed internally on a per-image basis with the `Astroalign._get_photometry` method, which is summarized in the above steps we used to produce `phot`.
"""

# ╔═╡ 255cb3ee-2ac4-4b20-8d4f-785ca9400668
md"""
### Step 2: Calculate invariants

This is done internally in `Astroalign.align_frame`, but the computed invariants `ℳᵢ` can be exposed with `Astroalign.triangle_invariants` for plotting and debugging. Below is a plot comparing the compents of the computed invariants for all control points in our `from` and `to` images. Overlapping sections indicates similar triangle between images found by Astroalign.jl. Compare to Fig 1. in [Beroiz, M., Cabral, J. B., & Sanchez, B. (2020)](https://arxiv.org/pdf/1909.02946).
"""

# ╔═╡ c46335bc-ae9a-4257-8a85-b4ccb94d1744
C_to, ℳ_to = triangle_invariants(phot_to)

# ╔═╡ d1d3f995-b901-4aab-86cd-e2d6f2393190
md"""
This can also be accessed through the `align_params` named tuple returned by `Astroalign.align_frame` in the [Usage](#Usage) example. We will use this to get the corresponding invariants for our `from` image;
"""

# ╔═╡ ad82de06-50f8-4e30-80b9-e4821e845162
(; C_from, ℳ_from) = align_params; C_from, ℳ_from

# ╔═╡ cdba7937-eea8-409a-b9e3-714e4516486c
let
	l = Layout(;
		xaxis = attr(title="L3/L2"),
		yaxis = attr(title="L2/L1"),
	)
	p1 = scatter(x=ℳ_to[1, :], y=ℳ_to[2, :];
		mode = :markers,
		name = "img_to",
	)
	p2 = scatter(x=ℳ_from[1, :], y=ℳ_from[2, :];
		mode = :markers,
		marker = attr(symbol="circle-open", size=10),
		name = "img_from",
	)

	plot([p1, p2], l)
end

# ╔═╡ 23a1364a-4ba0-42af-93bf-b6f900b9a13d
md"""
Note that the number of combinations of triangles in each frame can differ if the number of control points detected in each image is not the same. This can happen when sources drift towards the edge or off of the frame between images. All we need is one match though, which is helped by the additional combinations available.
"""

# ╔═╡ dffa0f3c-100f-4916-96c7-90274c0df5f2
md"""
### Step 3: Select nearest

We will find this closest match next to define our point-to-point correspondence using `Astroalign.find_nearest` with our computed invariants.
"""

# ╔═╡ dc2a101a-36d7-4402-b543-c576aba987ea
sol_to, sol_from = find_nearest(C_to, ℳ_to, C_from, ℳ_from)

# ╔═╡ bd2d9faf-7e0c-4a46-91e9-b3984dd3090e
aps_sol_to = map(sol_to) do sol
	CircularAperture(sol.xcenter, sol.ycenter, ap_radius)
end

# ╔═╡ 7f0b20db-e369-4e6a-aa5e-7df949791915
aps_sol_from = map(sol_from) do sol
	CircularAperture(sol.xcenter, sol.ycenter, ap_radius)
end

# ╔═╡ 1150fd19-ece7-4fd0-91db-a4df982d1e8e
md"""
### Step 4. Compute transform

Now that we have our point-to-point correspondence, we can compute our affine transformation needed to produce our aligned image.
"""

# ╔═╡ 6646cf68-daf0-4a83-b3a8-43415ee8f97f
point_map = map(sol_to, sol_from) do source_to, source_from
	[source_from.xcenter, source_from.ycenter] => [source_to.xcenter, source_to.ycenter]
end

# ╔═╡ 9db16b0e-1e1e-40a5-b7f4-56f819f4e0b1
tfm = kabsch(last.(point_map) => first.(point_map); scale=false)

# ╔═╡ 3779aed1-a02d-4370-8d56-37a2a5d374bf
md"""
We can now hand off this transformation to an image transformation library like `JuliaAstroImages.ImageTransformations` to view our final results. This should match our results returned by `Astroalign.align_frame` in the [Usage](#Usage) example.
"""

# ╔═╡ 7990c8be-9425-47d0-a913-9e2bb4fbefd1
img_aligned_from = shareheader(img_from, warp(img_from, tfm, axes(img_to)));

# ╔═╡ dd9296e8-0112-41e1-9ccc-4a3e813e2836
md"""
# 🔧 Notebook setup
"""

# ╔═╡ 1cf184a4-ec99-4cd2-8559-5d52b41ec629
function circ(ap, line_color=:lightgreen)
	circle(
		ap.x - ap.r, # x_min
		ap.x + ap.r, # x_max
		ap.y - ap.r, # y_min
		ap.y + ap.r; # y_max
		line_color,
	)
end

# ╔═╡ 5495dc08-2a7e-49c2-b0ee-7f6a816d584e
function circ2(phot; line_color=:lightgreen, r=16)
	circle(
		phot.xcenter - r, # x_min
		phot.xcenter + r, # x_max
		phot.ycenter - r, # y_min
		phot.ycenter + r; # y_max
		line_color,
	)
end

# ╔═╡ 5e09f7eb-a4af-4d94-8684-96857e716747
TableOfContents(; depth=4)

# ╔═╡ d00e04d9-7a12-481b-b3b3-5c1f7e31a1a7
# Global colorbar lims
const ZMIN, ZMAX = let
	# lims = Zscale(contrast=0.4).((img₁, img₂))
	lims = Percent(99.5).((img_to, img_from))
	minimum(first, lims), maximum(last, lims)
end

# ╔═╡ b461aadf-f88c-4195-8715-35e1e24a9bb4
function trace_hm(img; colorbar_x=0)
	imgv = copy(img)
	# Restriction prescription from AstroImages.jl/Images.jl
	# so plotting doesn't blow up for large images
	while length(eachindex(imgv)) > 10^6
		imgv = restrict(imgv)
	end
	imgv = permutedims(imgv)
	
	# zmin, zmax = Zscale(contrast=0.4)(img)
	return heatmap(x=dims(imgv, X).val, y=dims(imgv, Y).val, z=Matrix(imgv);
		zmin = ZMIN,
		zmax = ZMAX,
		colorscale = :Cividis,
		colorbar = attr(x=colorbar_x, thickness=10, title="Counts"),
	)
end

# ╔═╡ ada7cfc1-43a6-4469-8d23-7cbe37f22301
let
	l = Layout(;
		xaxis = attr(title="X"),
		yaxis = attr(title="Y"),
	)
	p = plot(trace_hm(img; colorbar_x=1.0), l)
	relayout!(p; shapes=circ.(aps))
	p
end

# ╔═╡ 05537e5b-347a-4198-80e9-7eeed85b08ca
 let
	l = Layout(;
		xaxis = attr(title="X"),
		yaxis = attr(title="Y"),
	)
	p = plot(trace_hm(img_to; colorbar_x=1.0), l)
	relayout!(p; shapes=circ.(aps_sol_to, [:magenta, :lightgreen, :blue]))
	p
end

# ╔═╡ 0612c049-c6d1-4e6a-a44a-b2f93a39a2c6
 let
	l = Layout(;
		xaxis = attr(title="X"),
		yaxis = attr(title="Y"),
	)
	p = plot(trace_hm(img_from; colorbar_x=1.0), l)
	relayout!(p; shapes=circ.(aps_sol_from, [:magenta, :lightgreen, :blue]))
	p
end

# ╔═╡ de7ff589-99c0-4625-8a10-86aa702d2510
function plot_pair(img₁, img₂; column_titles=["img_to", "img_from"])
	# Set up some subplots
	fig = make_subplots(;
		rows = 1,	
		cols = 2,
		shared_xaxes = true,
		shared_yaxes = true,
		column_titles,
	)
	
	# Make the subplot titles a smidgen bit smaller
	update_annotations!(fig, font_size=14)
	
	# Manually place the colorbars so they don't clash
	p1 = add_trace!(fig, trace_hm(img₁; colorbar_x=0.45), col=1)
	p2 = add_trace!(fig, trace_hm(img₂; colorbar_x=1), col=2)

	# Keep the images true to size
	update_xaxes!(fig, matches="x", scaleanchor=:y, title="X (pixels)")
	update_yaxes!(fig, matches="y", scaleanchor=:x)

	# Add a shared y-label
	relayout!(fig, Layout(yaxis_title="Y (pixels)"), font_size=10, template="plotly_white", margin=attr(t=20), uirevision=1)

	# Display
	return fig
end

# ╔═╡ f128f050-b716-4a79-8bb6-640708d1bc88
plot_pair(img_to, img_from)

# ╔═╡ 8769216b-00d4-44bd-97fd-7aa89cf19c23
plot_pair(img_to, arr_aligned_from)

# ╔═╡ 066210ea-b5b3-4f73-8fc1-503625fc32ce
fig = plot_pair(img_to, img_aligned_from)

# ╔═╡ Cell order:
# ╟─9e130a37-1073-4d0f-860a-0ec8d164dde1
# ╟─fa1180d4-c1ea-4a1b-8476-0e8d185d5622
# ╟─40c14093-3806-401f-aedf-f6435f785eb4
# ╟─5c4155fa-92bd-4260-a9d4-cea9dc5f3d93
# ╠═f128f050-b716-4a79-8bb6-640708d1bc88
# ╟─b51e47f6-af8e-478a-a716-af74e33c9e99
# ╟─8769216b-00d4-44bd-97fd-7aa89cf19c23
# ╠═445a0d35-2b49-42cc-8529-176778b0e090
# ╟─a1cb22fc-e956-4cf7-aafc-0168da23e556
# ╟─b8323ad8-c26b-4cc7-9891-caa05c128fb1
# ╟─eff56f6e-ab01-4371-a75f-f44bdde7cfd6
# ╠═dc01eaaa-f1d0-4bc6-884f-778d848918c6
# ╠═78c0bf28-bb96-4aea-8bf5-5929ef45adc1
# ╠═d97c367c-4db1-4dd0-8066-3f12e08d2f01
# ╠═0ae46a86-dd86-4092-9d34-05f643ec08af
# ╠═95531bde-8386-4d51-8c83-ffb796a41e90
# ╠═5882adec-7591-4d93-98e2-efb81496c54d
# ╠═f7639401-1fc9-4cb1-824c-4335a4bb8b25
# ╟─a2ed7b77-1277-41a3-8c29-a9814b124d09
# ╟─2bc269e1-dbe3-4c68-9a30-8c6054bc3a82
# ╟─fe518d92-fbfd-4d6f-ba71-0b7b23a73fd7
# ╠═00116e4a-8d9d-46dd-b09f-005a19ddf4ee
# ╠═fb0efcf4-26d4-4554-a5cf-b1136f5a6c17
# ╠═8afa31f0-ee57-4628-bedf-dd2b79faef72
# ╠═b3c689bd-1778-4568-96b9-869f2e6a83c0
# ╠═07abbeb9-15a4-4086-86ca-093e5475c0db
# ╟─ada7cfc1-43a6-4469-8d23-7cbe37f22301
# ╟─c6ef3b26-ccc1-401b-ba9b-88276d4c5067
# ╟─b0ad71b1-3a3c-481b-a08e-2ee558e8e1c5
# ╠═4e1c0615-d26d-4147-a096-d20940b8046a
# ╟─fcb02cf0-4fb5-4e31-bab9-d19a0755def9
# ╟─9109a7a0-4a37-4dca-a923-16a9302556ee
# ╟─3da14f39-9fad-412e-824b-c3db190700aa
# ╠═68139ad3-cf00-4286-b9eb-a435dd20aca2
# ╠═0083d7bb-07f2-45e6-b4f8-44099ff1a0bf
# ╟─35befaff-e36c-4741-b28f-3589afe596cd
# ╟─c73692f2-178d-4b35-badc-e9e682551989
# ╟─0603752b-5fcc-4e14-ae41-292cc49c6711
# ╟─0d4ce3b5-665a-4cc8-8884-90600e99f6ba
# ╟─f2720f8e-8df1-4cfe-a21c-59855e646106
# ╟─255cb3ee-2ac4-4b20-8d4f-785ca9400668
# ╟─cdba7937-eea8-409a-b9e3-714e4516486c
# ╠═c46335bc-ae9a-4257-8a85-b4ccb94d1744
# ╟─d1d3f995-b901-4aab-86cd-e2d6f2393190
# ╠═ad82de06-50f8-4e30-80b9-e4821e845162
# ╟─23a1364a-4ba0-42af-93bf-b6f900b9a13d
# ╟─dffa0f3c-100f-4916-96c7-90274c0df5f2
# ╠═dc2a101a-36d7-4402-b543-c576aba987ea
# ╠═bd2d9faf-7e0c-4a46-91e9-b3984dd3090e
# ╠═7f0b20db-e369-4e6a-aa5e-7df949791915
# ╟─05537e5b-347a-4198-80e9-7eeed85b08ca
# ╟─0612c049-c6d1-4e6a-a44a-b2f93a39a2c6
# ╟─1150fd19-ece7-4fd0-91db-a4df982d1e8e
# ╠═6646cf68-daf0-4a83-b3a8-43415ee8f97f
# ╠═9db16b0e-1e1e-40a5-b7f4-56f819f4e0b1
# ╟─3779aed1-a02d-4370-8d56-37a2a5d374bf
# ╠═7990c8be-9425-47d0-a913-9e2bb4fbefd1
# ╠═066210ea-b5b3-4f73-8fc1-503625fc32ce
# ╟─dd9296e8-0112-41e1-9ccc-4a3e813e2836
# ╟─1cf184a4-ec99-4cd2-8559-5d52b41ec629
# ╟─5495dc08-2a7e-49c2-b0ee-7f6a816d584e
# ╟─b461aadf-f88c-4195-8715-35e1e24a9bb4
# ╟─de7ff589-99c0-4625-8a10-86aa702d2510
# ╠═5e09f7eb-a4af-4d94-8684-96857e716747
# ╠═d00e04d9-7a12-481b-b3b3-5c1f7e31a1a7
# ╠═d8d4c414-64a0-11f0-15a3-0d566872a687
