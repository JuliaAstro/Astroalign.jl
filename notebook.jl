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

Like in the days of overhead transparencies
"""

# ╔═╡ b8323ad8-c26b-4cc7-9891-caa05c128fb1
md"""
## Load images 📷

Here are two sample images that we would like to align with each other. Plate solving is expensive, so instead we will try a quicker approach using good ol' triangles; no WCS required. Throughout this notebook, we will use the convention that we transform `from` the second image `to` the first image.

Credit: [_Beroiz, M., Cabral, J. B., & Sanchez, B. (2020)_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract)
"""

# ╔═╡ f7639401-1fc9-4cb1-824c-4335a4bb8b25
# Modified from
# https://github.com/JuliaAstro/PSFModels.jl/blob/main/test/fitting.jl
function generate_model(model, params, inds)
	cartinds = CartesianIndices(inds)
	psf = model.(cartinds; params..., amp=30_000)
	return psf .+ rand(1000:3000, size(psf))
end

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
fwhms = rand(2:3:20, N_sources)

# ╔═╡ caaef504-d0be-44c2-aa1a-57dac7e6ddf3
positions_to = rand(20:minimum(fwhms):280, N_sources, 2)

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

# ╔═╡ 7d48608c-e97f-4a80-b3f3-19d2b9abedca
md"""
### View

We view our fake data below. Rerun the `fwhm` and/or `positions_to` field to generate new star fields.
"""

# ╔═╡ 379aab23-f25b-41a6-ab89-a07058319306
img_to = sum(sources_to) |> AstroImage

# ╔═╡ 49d2aa49-6a9b-4c4b-9349-ce75a2daec6d
img_from = sum(sources_from) |> AstroImage

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
	w = maximum(fwhms)
	iseven(w) ? w + 1 : w 
end

# ╔═╡ 14f73b08-3cb6-48c8-9c00-02e53ffc589f
const ap_radius = sum(fwhms) / length(fwhms)

# ╔═╡ a8bb3c7a-5ff8-4742-b2ab-25350e0efc5e
img = img_to;

# ╔═╡ fb0efcf4-26d4-4554-a5cf-b1136f5a6c17
# Sources, background subtracted image, background
sources, subt, errs = get_sources(img; box_size);

# ╔═╡ 07abbeb9-15a4-4086-86ca-093e5475c0db
aps = CircularAperture.(sources.y, sources.x, ap_radius)

# ╔═╡ 4e1c0615-d26d-4147-a096-d20940b8046a
phot = get_photometry(img; box_size = 21)

# ╔═╡ 9109a7a0-4a37-4dca-a923-16a9302556ee
md"""
Below is a quick visual check that compares our observed point source with its fitted PSF model:
"""

# ╔═╡ 3da14f39-9fad-412e-824b-c3db190700aa
@bind i Slider(eachindex(phot); show_value=x -> "Source $(x)")

# ╔═╡ 35befaff-e36c-4741-b28f-3589afe596cd
function inspect_psf(phot)
	psf_data, psf_model = phot.aperture_f.psf_data, phot.aperture_f.psf_model
	@info phot.aperture_f.psf_P
	return AstroImage(psf_data), imview(psf_model.(CartesianIndices(psf_data)))
end

# ╔═╡ 68139ad3-cf00-4286-b9eb-a435dd20aca2
inspect_psf(phot[i])

# ╔═╡ 0083d7bb-07f2-45e6-b4f8-44099ff1a0bf
# And "truth" fwhms for comparison
sort(fwhms; rev=true)

# ╔═╡ 825d7483-18a1-4e9b-88c8-707636b129bd
md"""
## Match
"""

# ╔═╡ 733b2b72-ee6d-41f2-bb87-1cf5ad1114f9
img_aligned_from, point_map, tfm, ℳ_to, ℳ_from = align(img_to, img_from;
	box_size = 21,
	# fwhm_init = 5,
);

# ╔═╡ cdba7937-eea8-409a-b9e3-714e4516486c
let
	l = Layout(;
		xaxis = attr(title="L3/L2"),
		yaxis = attr(title="L2/L1"),
	)
	p1 = scatter(x=ℳ_to[1, :], y=ℳ_to[2, :];
		mode = :markers,
		name = "img₁",
	)
	p2 = scatter(x=ℳ_from[1, :], y=ℳ_from[2, :];
		mode = :markers,
		marker = attr(symbol="circle-open", size=10),
		name = "img₂",
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

# ╔═╡ f128f050-b716-4a79-8bb6-640708d1bc88
plot_pair(img_to, img_from)

# ╔═╡ 066210ea-b5b3-4f73-8fc1-503625fc32ce
plot_pair(img_to, AstroImage(img_aligned_from))

# ╔═╡ dd9296e8-0112-41e1-9ccc-4a3e813e2836
md"""
# 🔧 Notebook setup
"""

# ╔═╡ 5e09f7eb-a4af-4d94-8684-96857e716747
TableOfContents(; depth=4)

# ╔═╡ Cell order:
# ╟─9e130a37-1073-4d0f-860a-0ec8d164dde1
# ╟─b8323ad8-c26b-4cc7-9891-caa05c128fb1
# ╠═d8d4c414-64a0-11f0-15a3-0d566872a687
# ╟─eff56f6e-ab01-4371-a75f-f44bdde7cfd6
# ╠═f7639401-1fc9-4cb1-824c-4335a4bb8b25
# ╠═dc01eaaa-f1d0-4bc6-884f-778d848918c6
# ╠═78c0bf28-bb96-4aea-8bf5-5929ef45adc1
# ╠═0ae46a86-dd86-4092-9d34-05f643ec08af
# ╠═caaef504-d0be-44c2-aa1a-57dac7e6ddf3
# ╠═5aa9cd26-6d5d-46df-b76b-61c6b506af86
# ╠═95531bde-8386-4d51-8c83-ffb796a41e90
# ╠═81224d9e-df5f-472a-b42b-4b0d3a3e227e
# ╟─7d48608c-e97f-4a80-b3f3-19d2b9abedca
# ╠═379aab23-f25b-41a6-ab89-a07058319306
# ╠═49d2aa49-6a9b-4c4b-9349-ce75a2daec6d
# ╠═f128f050-b716-4a79-8bb6-640708d1bc88
# ╟─b51e47f6-af8e-478a-a716-af74e33c9e99
# ╟─a2ed7b77-1277-41a3-8c29-a9814b124d09
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
# ╠═35befaff-e36c-4741-b28f-3589afe596cd
# ╠═0083d7bb-07f2-45e6-b4f8-44099ff1a0bf
# ╟─825d7483-18a1-4e9b-88c8-707636b129bd
# ╠═733b2b72-ee6d-41f2-bb87-1cf5ad1114f9
# ╟─cdba7937-eea8-409a-b9e3-714e4516486c
# ╠═066210ea-b5b3-4f73-8fc1-503625fc32ce
# ╟─c485cd62-f315-44e8-88bd-e97318c32923
# ╟─1cf184a4-ec99-4cd2-8559-5d52b41ec629
# ╟─5495dc08-2a7e-49c2-b0ee-7f6a816d584e
# ╟─b461aadf-f88c-4195-8715-35e1e24a9bb4
# ╟─de7ff589-99c0-4625-8a10-86aa702d2510
# ╠═d00e04d9-7a12-481b-b3b3-5c1f7e31a1a7
# ╟─dd9296e8-0112-41e1-9ccc-4a3e813e2836
# ╠═5e09f7eb-a4af-4d94-8684-96857e716747
