### A Pluto.jl notebook ###
# v0.20.13

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
	Pkg.activate(Base.current_project())

	# External packages
	using AstroImages, PlutoPlotly, PlutoUI, PSFModels, Revise, Rotations, Photometry
	AstroImages.set_cmap!(:cividis)

	# Internal packages
	using Astroalign

	Pkg.instantiate()
end

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
## Usage

Here is a brief usage example aligning `img_from` onto `img_to` with the exported `align` function from Astroalign.jl.
"""

# ╔═╡ 58b2a3ab-9a1b-469c-8c2e-2f4e1740d6d5
md"""
The available parameters to adjust are:

* `box_size`: The size of the grid cells (in pixels) used to extract point sources. Defaults to a tenth of the GCD of the dimensions of the first image. See [Photometry.jl > Source Detection Algorithms](https://juliaastro.org/Photometry/stable/detection/algs/#Source-Detection-Algorithms) for more.
* `ap_radius`: The radius of the apertures (in pixel) to place around each point source. Defaults to 60% of the `box_size`. See [Photometry.jl > Aperture Photometry](https://juliaastro.org/Photometry/stable/apertures/) for more.
* `min_fwhm`: The minimum FWHM (in pixels) that an extracted point source must have to be considered as a control point. Defaults to a fifth of the width of the first image. See [PSFModels.jl > Fitting data](https://juliaastro.org/PSFModels/stable/introduction/#Fitting-data) for more.
* `nsigma`: The number of standard deviations above the estimated background that a source must be to be considered as a control point. Defaults to 1. See [Photometry.jl > Source Detection Algorithms](https://juliaastro.org/Photometry/stable/detection/algs/#Source-Detection-Algorithms) for more.
"""

# ╔═╡ a1cb22fc-e956-4cf7-aafc-0168da23e556
md"""
For even more control, each step of the alignment process has an associated API that can be used from Astroalign.jl, which we show in the rest of this notebook.
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

# ╔═╡ f7639401-1fc9-4cb1-824c-4335a4bb8b25
# Modified from
# https://github.com/JuliaAstro/PSFModels.jl/blob/main/test/fitting.jl
function generate_model(model, params, inds)
	cartinds = CartesianIndices(inds)
	psf = model.(cartinds; params..., amp=30_000)
	return psf .+ rand(1000:3000, size(psf))
end

# ╔═╡ 7d48608c-e97f-4a80-b3f3-19d2b9abedca
md"""
### View

Click the button below to generate a new star field.
"""

# ╔═╡ e1434564-9864-4ea2-9223-2b3b5aa0093a
@bind new_data Button("Generate new data")

# ╔═╡ 0ae46a86-dd86-4092-9d34-05f643ec08af
begin
	new_data
	fwhms = rand(2:3:20, N_sources)
	positions_to = rand(20:minimum(fwhms):280, N_sources, 2)
end

# ╔═╡ 5aa9cd26-6d5d-46df-b76b-61c6b506af86
positions_from = round.(Int, positions_to * RotMatrix2(π/8) .+  [-80 120])

# ╔═╡ 95531bde-8386-4d51-8c83-ffb796a41e90
sources_to = map(zip(eachrow(positions_to), fwhms)) do ((x, y), fwhm)
	generate_model(gaussian, (; x, y, fwhm), img_size)
end;

# ╔═╡ 81224d9e-df5f-472a-b42b-4b0d3a3e227e
sources_from = map(zip(eachrow(positions_from), fwhms)) do ((x, y), fwhm)
	generate_model(gaussian, (; x, y, fwhm), img_size)
end

# ╔═╡ 379aab23-f25b-41a6-ab89-a07058319306
img_to, img_from = sum.((sources_to, sources_from)) .|> AstroImage

# ╔═╡ 226863a9-9f6e-40c4-aeb0-452f6a1acd53
img_to, img_from

# ╔═╡ b51e47f6-af8e-478a-a716-af74e33c9e99
md"""
In this particular case, `img_from` is rotated clockwise, and shifted vertically upwards and horizontally to the left relative to `img_to` in the above plot. Let's fix it.
"""

# ╔═╡ a2ed7b77-1277-41a3-8c29-a9814b124d09
md"""
## Align -- ✨
"""

# ╔═╡ c0063ec1-a52f-4bd2-a4e0-9c218fbe6a72
const box_size = let
	w = gcd(size(img_to)...) ÷ 10
	iseven(w) ? w + 1 : w 
end

# ╔═╡ 14f73b08-3cb6-48c8-9c00-02e53ffc589f
const ap_radius = 0.6*box_size

# ╔═╡ 445a0d35-2b49-42cc-8529-176778b0e090
img_aligned_from, point_map, tfm, ℳ_to, ℳ_from = align(img_to, img_from;
	box_size,
	ap_radius,
	min_fwhm = first(box_size) ÷ 5,
	nsigma = 1,
);

# ╔═╡ a8bb3c7a-5ff8-4742-b2ab-25350e0efc5e
img = img_to;

# ╔═╡ fb0efcf4-26d4-4554-a5cf-b1136f5a6c17
# Sources, background subtracted image, background
sources, subt, errs = get_sources(img; box_size);

# ╔═╡ 07abbeb9-15a4-4086-86ca-093e5475c0db
aps = CircularAperture.(sources.y, sources.x, ap_radius)

# ╔═╡ 4e1c0615-d26d-4147-a096-d20940b8046a
phot = first(get_photometry(img; box_size, ap_radius), N_sources)

# ╔═╡ 9109a7a0-4a37-4dca-a923-16a9302556ee
md"""
Below is a quick visual check that compares our observed point source with its fitted PSF model:
"""

# ╔═╡ 3da14f39-9fad-412e-824b-c3db190700aa
@bind i Slider(eachindex(phot); show_value=x -> "Source $(x)")

# ╔═╡ 0083d7bb-07f2-45e6-b4f8-44099ff1a0bf
# And "truth" fwhms for comparison
sort(fwhms; rev=true)

# ╔═╡ 35befaff-e36c-4741-b28f-3589afe596cd
function inspect_psf(phot)
	psf_data, psf_model = phot.aperture_f.psf_data, phot.aperture_f.psf_model
	@debug phot.aperture_f.psf_P
	return AstroImage(psf_data), imview(psf_model.(CartesianIndices(psf_data)))
end

# ╔═╡ 68139ad3-cf00-4286-b9eb-a435dd20aca2
inspect_psf(phot[i])

# ╔═╡ 825d7483-18a1-4e9b-88c8-707636b129bd
md"""
### Match
"""

# ╔═╡ 733b2b72-ee6d-41f2-bb87-1cf5ad1114f9


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

# ╔═╡ c485cd62-f315-44e8-88bd-e97318c32923
md"""
## Plotly convenience functions 🖌️
"""

# ╔═╡ 1cf184a4-ec99-4cd2-8559-5d52b41ec629
function circ(ap; line_color=:lightgreen)
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

# ╔═╡ dd9296e8-0112-41e1-9ccc-4a3e813e2836
md"""
# 🔧 Notebook setup
"""

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
	add_trace!(fig, trace_hm(img₁; colorbar_x=0.45), col=1)
	add_trace!(fig, trace_hm(img₂; colorbar_x=1), col=2)

	# Keep the images true to size
	update_xaxes!(fig, matches="x", scaleanchor=:y, title="X (pixels)")
	update_yaxes!(fig, matches="y", scaleanchor=:x)

	# Add a shared y-label
	relayout!(fig, Layout(yaxis_title="Y (pixels)"), font_size=10, template="plotly_white", margin=attr(t=20), uirevision=1)

	# Display
	fig
end

# ╔═╡ 8769216b-00d4-44bd-97fd-7aa89cf19c23
plot_pair(img_to, AstroImage(img_aligned_from))

# ╔═╡ f128f050-b716-4a79-8bb6-640708d1bc88
plot_pair(img_to, img_from)

# ╔═╡ 066210ea-b5b3-4f73-8fc1-503625fc32ce
plot_pair(img_to, AstroImage(img_aligned_from))

# ╔═╡ Cell order:
# ╟─9e130a37-1073-4d0f-860a-0ec8d164dde1
# ╟─fa1180d4-c1ea-4a1b-8476-0e8d185d5622
# ╟─40c14093-3806-401f-aedf-f6435f785eb4
# ╠═226863a9-9f6e-40c4-aeb0-452f6a1acd53
# ╠═445a0d35-2b49-42cc-8529-176778b0e090
# ╟─58b2a3ab-9a1b-469c-8c2e-2f4e1740d6d5
# ╠═8769216b-00d4-44bd-97fd-7aa89cf19c23
# ╟─a1cb22fc-e956-4cf7-aafc-0168da23e556
# ╟─b8323ad8-c26b-4cc7-9891-caa05c128fb1
# ╟─eff56f6e-ab01-4371-a75f-f44bdde7cfd6
# ╠═dc01eaaa-f1d0-4bc6-884f-778d848918c6
# ╠═78c0bf28-bb96-4aea-8bf5-5929ef45adc1
# ╠═0ae46a86-dd86-4092-9d34-05f643ec08af
# ╠═5aa9cd26-6d5d-46df-b76b-61c6b506af86
# ╠═95531bde-8386-4d51-8c83-ffb796a41e90
# ╠═81224d9e-df5f-472a-b42b-4b0d3a3e227e
# ╠═f7639401-1fc9-4cb1-824c-4335a4bb8b25
# ╟─7d48608c-e97f-4a80-b3f3-19d2b9abedca
# ╟─e1434564-9864-4ea2-9223-2b3b5aa0093a
# ╠═379aab23-f25b-41a6-ab89-a07058319306
# ╠═f128f050-b716-4a79-8bb6-640708d1bc88
# ╟─b51e47f6-af8e-478a-a716-af74e33c9e99
# ╠═a2ed7b77-1277-41a3-8c29-a9814b124d09
# ╠═c0063ec1-a52f-4bd2-a4e0-9c218fbe6a72
# ╠═14f73b08-3cb6-48c8-9c00-02e53ffc589f
# ╠═a8bb3c7a-5ff8-4742-b2ab-25350e0efc5e
# ╠═fb0efcf4-26d4-4554-a5cf-b1136f5a6c17
# ╠═07abbeb9-15a4-4086-86ca-093e5475c0db
# ╟─ada7cfc1-43a6-4469-8d23-7cbe37f22301
# ╠═4e1c0615-d26d-4147-a096-d20940b8046a
# ╟─9109a7a0-4a37-4dca-a923-16a9302556ee
# ╠═3da14f39-9fad-412e-824b-c3db190700aa
# ╠═68139ad3-cf00-4286-b9eb-a435dd20aca2
# ╠═0083d7bb-07f2-45e6-b4f8-44099ff1a0bf
# ╟─35befaff-e36c-4741-b28f-3589afe596cd
# ╟─825d7483-18a1-4e9b-88c8-707636b129bd
# ╠═733b2b72-ee6d-41f2-bb87-1cf5ad1114f9
# ╟─cdba7937-eea8-409a-b9e3-714e4516486c
# ╟─066210ea-b5b3-4f73-8fc1-503625fc32ce
# ╟─c485cd62-f315-44e8-88bd-e97318c32923
# ╟─1cf184a4-ec99-4cd2-8559-5d52b41ec629
# ╟─5495dc08-2a7e-49c2-b0ee-7f6a816d584e
# ╟─b461aadf-f88c-4195-8715-35e1e24a9bb4
# ╟─de7ff589-99c0-4625-8a10-86aa702d2510
# ╟─dd9296e8-0112-41e1-9ccc-4a3e813e2836
# ╠═5e09f7eb-a4af-4d94-8684-96857e716747
# ╠═d00e04d9-7a12-481b-b3b3-5c1f7e31a1a7
# ╠═d8d4c414-64a0-11f0-15a3-0d566872a687
