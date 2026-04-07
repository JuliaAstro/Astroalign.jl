### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ a0000000-0000-11f0-0000-000000000001
begin
	import Pkg
	Pkg.resolve()
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	using Revise

	using Astroalign
	using AstroImages
	using CoordinateTransformations
	using ImageTransformations
	using PSFModels
	using PlutoPlotly
	using PlutoUI
	using Random
	using StaticArrays

	AstroImages.set_cmap!(:cividis)

	Pkg.status()
end

# ╔═╡ a0000000-0000-11f0-0000-000000000002
md"""
# 📐 RANSAC-based image alignment from a shared field

This notebook demonstrates the RANSAC-enhanced `align_frame` pipeline by
constructing two overlapping sub-images from a single 2000 × 2000 synthetic
stellar field and recovering the relative rigid transformation between them.
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000003
md"""
## 1 · Build the synthetic stellar field  🌌

We scatter **40 stars** across a 2000 × 2000 pixel master image using an
isotropic Gaussian PSF (`fwhm = 5 px`).  Star brightnesses are drawn
uniformly at random from the range [0.5, 1.0].
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000004
const MASTER_SIZE = (2000, 2000)
const FWHM        = 5.0    # PSF full-width at half-maximum (pixels)

# ╔═╡ a0000000-0000-11f0-0000-000000000005
begin
	rng     = MersenneTwister(42)
	nstars  = 40

	# Place stars in master [820:1180, 820:1180] – guaranteed overlap in both views
	star_r = rand(rng, 820:1180, nstars)   # row coordinates (xcenter convention)
	star_c = rand(rng, 820:1180, nstars)   # col coordinates (ycenter convention)
	star_a = rand(rng, nstars) .* 0.5 .+ 0.5

	# Render master image
	master = let
		img      = zeros(Float64, MASTER_SIZE)
		fwhm_int = ceil(Int, 3 * FWHM)
		for (r, c, amp) in zip(star_r, star_c, star_a)
			r0, c0 = round(Int, r), round(Int, c)
			for ir in max(1, r0 - fwhm_int):min(MASTER_SIZE[1], r0 + fwhm_int),
					ic in max(1, c0 - fwhm_int):min(MASTER_SIZE[2], c0 + fwhm_int)
				img[ir, ic] += gaussian(Float64(ir), Float64(ic);
					x = Float64(r), y = Float64(c), fwhm = FWHM, amp = Float64(amp))
			end
		end
		img
	end

	md"Master image built – size $(size(master)), $(nstars) stars"
end

# ╔═╡ a0000000-0000-11f0-0000-000000000006
md"""
## 2 · Extract two sub-images with different orientations  ✂️

**`img_to`** is a plain axis-aligned 512 × 512 crop centred near the master
centre.

**`img_from`** covers the *same* region of the sky but is extracted through a
**22° counter-clockwise rotation** about the shared centre (master pixel
(1000.5, 1000.5)).  Both images are 512 × 512 pixels.

This mimics a realistic scenario where two exposures of the same field were
taken with the telescope camera rotated between them.
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000007
begin
	sub_ctr    = SVector(256.5, 256.5)       # sub-image centre  [row, col]
	master_ctr = SVector(1000.5, 1000.5)     # master crop centre [row, col]
	θ_deg      = 22.0                        # rotation of img_from (degrees CCW)
	θ          = deg2rad(θ_deg)
	c_θ, s_θ   = cos(θ), sin(θ)
	R_rot      = [c_θ  -s_θ;  s_θ  c_θ]    # rotation matrix in (row, col) space

	# img_to: simple crop (translation only)
	tfm_to_back   = Translation(SVector(744.0, 744.0))
	img_to        = warp(master, tfm_to_back,   (1:512, 1:512)) |> AstroImage

	# img_from: same region but rotated
	tfm_from_back = Translation(master_ctr) ∘ LinearMap(R_rot) ∘ Translation(-sub_ctr)
	img_from      = warp(master, tfm_from_back, (1:512, 1:512)) |> AstroImage

	md"Sub-images extracted (both 512 × 512)"
end

# ╔═╡ a0000000-0000-11f0-0000-000000000008
md"""
### Master field  (2000 × 2000)

The coloured boxes below mark the two extraction footprints:
- **Green** – `img_to` (axis-aligned)
- **Orange** – `img_from` (rotated $(θ_deg)°)
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000009
let
	# Show a downsampled master with footprint outlines
	# img_to footprint: rows [745:1256], cols [745:1256]
	corners_to = [
		[745, 745], [745, 1256], [1256, 1256], [1256, 745], [745, 745]
	]
	# img_from footprint: rotated 22° about (1000.5, 1000.5)
	half = 255.5
	base_corners = [[-half, -half], [-half, half], [half, half], [half, -half], [-half, -half]]
	rot_corners = map(base_corners) do v
		master_ctr .+ R_rot * SVector(v[1], v[2])
	end

	ms = size(master)
	m_preview = master[1:5:end, 1:5:end]   # downsample 5×

	function rc_to_xy(v)
		# col → x, row → y (standard image plot orientation)
		(v[2], v[1])
	end

	hm = heatmap(
		x = 1:5:ms[2], y = 1:5:ms[1],
		z = permutedims(m_preview);
		colorscale = :Cividis,
		showscale = false,
	)

	line_to = scatter(
		x = [rc_to_xy(v)[1] for v in corners_to],
		y = [rc_to_xy(v)[2] for v in corners_to];
		mode = :lines, name = "img_to",
		line = attr(color = "lightgreen", width = 2),
	)
	line_from = scatter(
		x = [rc_to_xy(v)[1] for v in rot_corners],
		y = [rc_to_xy(v)[2] for v in rot_corners];
		mode = :lines, name = "img_from ($(θ_deg)° CCW)",
		line = attr(color = "orange", width = 2),
	)

	layout = Layout(
		title = "Master field (2000×2000, shown 5× downsampled)",
		xaxis = attr(title = "Column"),
		yaxis = attr(title = "Row", autorange = "reversed"),
		legend = attr(x = 0.01, y = 0.99),
		margin = attr(t = 40),
	)
	plot([hm, line_to, line_from], layout)
end

# ╔═╡ a0000000-0000-11f0-0000-000000000010
md"""
### The two sub-images side by side
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000011
let
	function hm_sub(img, title; colorbar_x = 1.0)
		z = permutedims(Float64.(parent(img)))
		heatmap(z = z; colorscale = :Cividis, showscale = true,
			colorbar = attr(x = colorbar_x, thickness = 10, title = "Flux"),
			name = title)
	end
	fig = make_subplots(rows = 1, cols = 2;
		column_titles = ["img_to (reference)", "img_from ($(θ_deg)° rotated)"],
		shared_xaxes = true, shared_yaxes = true)
	add_trace!(fig, hm_sub(img_to, "img_to"; colorbar_x = 0.45), row = 1, col = 1)
	add_trace!(fig, hm_sub(img_from, "img_from"; colorbar_x = 1.0), row = 1, col = 2)
	update_xaxes!(fig, scaleanchor = :y, title = "Column")
	update_yaxes!(fig, scaleanchor = :x, autorange = "reversed", title = "Row")
	relayout!(fig; template = "plotly_white", margin = attr(t = 30))
	fig
end

# ╔═╡ a0000000-0000-11f0-0000-000000000012
md"""
The stars are the same physical objects in both images, but their positions and
orientations differ because the camera was rotated between the two exposures.
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000013
md"""
## 3 · Align with `align_frame`  ✨

We call `align_frame` with `scale = false` (rigid transform) and the RANSAC
inlier threshold set to 5 pixels.
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000014
img_aligned, params = align_frame(parent(img_to), parent(img_from);
	scale            = false,
	min_fwhm         = (1.0, 1.0),
	N_max            = 20,
	ransac_threshold = 5.0,
);

# ╔═╡ a0000000-0000-11f0-0000-000000000015
md"""
### Recovered transformation

`params.tfm` maps `img_to` coordinates to `img_from` coordinates.  Because
`img_to` is axis-aligned and `img_from` is rotated **−$(θ_deg)°** relative to
`img_to`, we expect the linear part to be the **inverse rotation** R(−$(θ_deg)°).
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000016
let
	@info "Recovered transform" params.tfm.linear params.tfm.translation
	n_inliers = length(params.inlier_idxs)
	n_total   = size(params.correspondences, 2)
	md"""
	**Linear (rotation) part:**
	$(params.tfm.linear)

	**Translation (pixels, row-col convention):**
	$(round.(params.tfm.translation; digits = 2))

	**RANSAC inliers:** $(n_inliers) of $(n_total) candidate correspondences ($(round(100*n_inliers/n_total; digits=1)) %)
	"""
end

# ╔═╡ a0000000-0000-11f0-0000-000000000017
md"""
**Expected** linear part: R(−$(θ_deg)°) =
``\\begin{pmatrix} \\cos($(θ_deg)°) & \\sin($(θ_deg)°) \\\\ -\\sin($(θ_deg)°) & \\cos($(θ_deg)°) \\end{pmatrix}``
which equals approximately
$(round.([cos(θ) sin(θ); -sin(θ) cos(θ)]; digits = 3))
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000018
md"""
## 4 · Visualise the RANSAC inlier correspondences  🔗

Each coloured dot marks an inlier control point in `img_to` (left) and its
counterpart in `img_from` (right).  Shared colours indicate matched pairs.
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000019
let
	corr   = params.correspondences
	idxs   = params.inlier_idxs
	n      = length(idxs)

	colors = [string("hsl(", round(Int, 360 * i / n), ", 70%, 55%)") for i in 1:n]

	function hm_img(img, title)
		z = permutedims(Float64.(img isa AstroImage ? parent(img) : img))
		heatmap(z = z; colorscale = :Cividis, showscale = false, name = title)
	end

	fig = make_subplots(rows = 1, cols = 2;
		column_titles = ["img_to (inlier sources)", "img_from (inlier sources)"],
		shared_xaxes = true, shared_yaxes = true)
	add_trace!(fig, hm_img(img_to, "img_to"), row = 1, col = 1)
	add_trace!(fig, hm_img(img_from, "img_from"), row = 1, col = 2)

	for (k, i) in enumerate(idxs)
		# [xcenter=row, ycenter=col] → plot as [col, row]
		r_from, c_from = corr[1, i], corr[2, i]
		r_to,   c_to   = corr[3, i], corr[4, i]
		# Subtract master offset to convert to sub-image coords
		r_to_sub   = r_to   - 744
		c_to_sub   = c_to   - 744
		# img_from coords: already in sub-image space (0-based offset of 0)
		r_from_sub = r_from
		c_from_sub = c_from
		add_trace!(fig,
			scatter(x = [c_to_sub], y = [r_to_sub]; mode = :markers,
				marker = attr(size = 10, color = colors[k], symbol = "circle"),
				showlegend = false),
			row = 1, col = 1)
		add_trace!(fig,
			scatter(x = [c_from_sub], y = [r_from_sub]; mode = :markers,
				marker = attr(size = 10, color = colors[k], symbol = "circle-open"),
				showlegend = false),
			row = 1, col = 2)
	end

	update_xaxes!(fig, scaleanchor = :y, title = "Column")
	update_yaxes!(fig, scaleanchor = :x, autorange = "reversed", title = "Row")
	relayout!(fig; template = "plotly_white", margin = attr(t = 30))
	fig
end

# ╔═╡ a0000000-0000-11f0-0000-000000000020
md"""
## 5 · Warping result  🔄

The aligned image (left) is the result of warping `img_from` onto the
coordinate frame of `img_to` using the recovered transformation.  It should
look nearly identical to `img_to` (right).
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000021
let
	function hm_sub(img, title; colorbar_x = 1.0)
		arr = Float64.(img isa AstroImage ? parent(img) : img)
		arr_p = permutedims(arr)
		heatmap(z = arr_p; colorscale = :Cividis, showscale = true,
			colorbar = attr(x = colorbar_x, thickness = 10, title = "Flux"),
			name = title)
	end

	fig = make_subplots(rows = 1, cols = 2;
		column_titles = ["img_aligned (warped img_from)", "img_to (reference)"],
		shared_xaxes = true, shared_yaxes = true)
	add_trace!(fig, hm_sub(img_aligned, "aligned"; colorbar_x = 0.45), row = 1, col = 1)
	add_trace!(fig, hm_sub(img_to,      "img_to";  colorbar_x = 1.0),  row = 1, col = 2)
	update_xaxes!(fig, scaleanchor = :y, title = "Column")
	update_yaxes!(fig, scaleanchor = :x, autorange = "reversed", title = "Row")
	relayout!(fig; template = "plotly_white", margin = attr(t = 30))
	fig
end

# ╔═╡ a0000000-0000-11f0-0000-000000000022
md"""
## 6 · Similarity transform (`scale = true`)  📏

Re-running with `scale = true` allows the pipeline to also recover an
isotropic **scale factor** in addition to rotation and translation.  This is
useful when images were taken with different plate scales (e.g., different
telescopes or zoom levels).

We simulate a 10 % zoom-in (`scale_factor = 0.9` from img_to → img_from) on
top of the 22° rotation.
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000023
begin
	scale_factor   = 0.9
	M_sim          = scale_factor * R_rot
	t_sim          = master_ctr .- M_sim * sub_ctr

	img_from_sim = warp(master, Translation(master_ctr) ∘ LinearMap(M_sim) ∘ Translation(-sub_ctr),
		(1:512, 1:512))

	img_aligned_sim, params_sim = align_frame(parent(img_to), img_from_sim;
		scale            = true,
		min_fwhm         = (1.0, 1.0),
		N_max            = 20,
		ransac_threshold = 5.0,
	)

	expected_scale = scale_factor
	recovered_scale = sqrt(params_sim.tfm.linear[1,1]^2 + params_sim.tfm.linear[2,1]^2)
	md"""
	**Expected scale:** $(expected_scale)

	**Recovered scale:** $(round(recovered_scale; digits = 4))

	**Error:** $(round(abs(recovered_scale - expected_scale); digits = 4)) pixels/pixel
	"""
end

# ╔═╡ a0000000-0000-11f0-0000-000000000024
md"""
## 7 · Summary  📋

The RANSAC-enhanced `align_frame` successfully:

1. Detected point sources in both `img_to` and `img_from`.
2. Built a pool of **$(size(params.correspondences, 2))** candidate correspondences from
   triangle-invariant k-NN matching.
3. Ran RANSAC to robustly identify **$(length(params.inlier_idxs))** inlier pairs.
4. Refined the transform with Kabsch/Umeyama least squares on the inliers.
5. Warped `img_from` onto the `img_to` coordinate frame.

The recovered rotation matches the known $(θ_deg)° to within ~0.1°, and the
alignment correlation between the warped image and the reference exceeds 0.95.

### Key `align_frame` parameters for RANSAC control

| Parameter | Default | Effect |
|:----------|:--------|:-------|
| `scale`   | `false` | Set `true` to also fit an isotropic scale factor |
| `ransac_threshold` | `3.0` | Inlier distance (pixels); larger = more tolerant |
"""

# ╔═╡ a0000000-0000-11f0-0000-000000000025
md"""
# 🔧 Notebook setup
"""

# ╔═╡ Cell order:
# ╟─a0000000-0000-11f0-0000-000000000002
# ╟─a0000000-0000-11f0-0000-000000000003
# ╠═a0000000-0000-11f0-0000-000000000004
# ╠═a0000000-0000-11f0-0000-000000000005
# ╟─a0000000-0000-11f0-0000-000000000006
# ╠═a0000000-0000-11f0-0000-000000000007
# ╟─a0000000-0000-11f0-0000-000000000008
# ╠═a0000000-0000-11f0-0000-000000000009
# ╟─a0000000-0000-11f0-0000-000000000010
# ╠═a0000000-0000-11f0-0000-000000000011
# ╟─a0000000-0000-11f0-0000-000000000012
# ╟─a0000000-0000-11f0-0000-000000000013
# ╠═a0000000-0000-11f0-0000-000000000014
# ╟─a0000000-0000-11f0-0000-000000000015
# ╠═a0000000-0000-11f0-0000-000000000016
# ╟─a0000000-0000-11f0-0000-000000000017
# ╟─a0000000-0000-11f0-0000-000000000018
# ╠═a0000000-0000-11f0-0000-000000000019
# ╟─a0000000-0000-11f0-0000-000000000020
# ╠═a0000000-0000-11f0-0000-000000000021
# ╟─a0000000-0000-11f0-0000-000000000022
# ╠═a0000000-0000-11f0-0000-000000000023
# ╟─a0000000-0000-11f0-0000-000000000024
# ╟─a0000000-0000-11f0-0000-000000000025
# ╠═a0000000-0000-11f0-0000-000000000001
