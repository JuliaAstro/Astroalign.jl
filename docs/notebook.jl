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
	
	using Astroalign, AstroImages, PlutoPlotly, PlutoUI, PSFModels, Rotations, Photometry, ImageTransformations, CoordinateTransformations, LinearAlgebra, Random

	using ConsensusFitting: ransac
	
	AstroImages.set_cmap!(:cividis)

	Pkg.status()
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

Enter [`astroalign.py`](https://github.com/quatrope/astroalign). This neat Python package sidesteps all of this by directly matching common star patterns between images to build this point-to-point correspondence. This notebook outlines a reimplementation of its core functionality in Julia, which we package as [JuliaAstro/Astroalign.jl](https://github.com/JuliaAstro/Astroalign.jl).
"""

# ╔═╡ 40c14093-3806-401f-aedf-f6435f785eb4
md"""
### Usage

Here is a brief usage example aligning `img_from` onto `img_to` with the exported `align_frame` function from Astroalign.jl. Select a star field below to get started:

"""

# ╔═╡ c5bfce23-d050-42e3-8af2-f1181adaaa2d
@bind seed PlutoUI.Radio([i => "Star field $(i)" for i in 1:5]; default = 1)

# ╔═╡ 7c1942c2-f61c-4c17-a0a5-0701c19d3d4f
md"""
That's it!

The rest of this notebook will walk through how this works behind the scenes and the various knobs that you can turn.
"""

# ╔═╡ fde0d2e4-e8ce-4861-8d53-43d58c9f8fe1
md"""
### Recovered transformation

As a quick check, here is the transformation object `tfm` returned by `Astroalign.align_frame`:
"""

# ╔═╡ 0c7b43c9-0456-433c-800d-1234b66f54a0
md"""
Taking its inverse (to define the mapping from `img_from => img_to`), and decomposing it into its scale (`S`), rotation (`R`), and translation (`T`) components then gives (rounded for display):
"""

# ╔═╡ 25b3db66-d22d-4372-a15b-02031aeb01d4
md"""
This gives the following transformation parameters:
"""

# ╔═╡ 6e44a52d-cc2a-45eb-ade3-001488cd2f49
function decompose_tfm(tfm)
	M = tfm.linear
	S = sqrt(M'M)
	R = M * inv(S)
	T = tfm.translation
	return (; S, R, T)
end

# ╔═╡ c84e1690-2176-427f-b8b2-eb5feacdcc2d
p_diff(x, x0) = round(100 * (x - x0) / x0; digits = 3)

# ╔═╡ f47dd317-6ac6-4f5f-95f8-eaca3d2820a6
p_diff(x::AbstractVector, x0::AbstractVector) = round(100 * norm(x - x0) / norm(x0); digits = 3)

# ╔═╡ c5658a61-99e2-4008-a542-9e12bf70ee9b
md"""
## Overview

[_Beroiz, Cabral, & Sanchez_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract) use the fact that triangles can be uniquely characterized to match sets of three stars (asterisms) between images. This point-to-point correspondence then gives us everything we need to compute the affine transformation between them.

For this implementation, they use the invariant ``\mathscr M`` (the pair of two independent ratios of a triangle's side lengths, ``L_i``) to define this unique characterization:

```math
\begin{align}
&\mathscr M = (L_3/L_2, L_2/L_1), \\

&\text{where}\ L_3 > L_2 > L_1\quad.
\end{align}
```

Astroalign.jl accomplishes this in the following steps:

1. Identify the `N_max` brightest point-like sources in `img_from` and `img_to`.
2. Calculate all triangular asterisms formed from these sources.
3. Build a `2 × 3 × 2 × N` array of candidate triangle-level correspondences
   by matching each from-triangle to its nearest to-triangle in
   the invariant ``\mathscr M`` space defined by [Beroiz et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract).
   Vertices are assigned via a canonical ordering that is invariant under
   rotation, so the positional correspondence between matched triangles is
   geometrically consistent.  The axes are `[coord, vertex, frame, match]`
   where `coord ∈ {x, y}`, `vertex ∈ {1,2,3}`, and `frame ∈ {from, to}`.
4. Run RANSAC ([Fischler & Bolles, 1981](https://dl.acm.org/doi/10.1145/358669.358692))
   on the triangle matches to robustly identify the largest set of mutually
   consistent correspondences ("inliers").  Each hypothesis is a Kabsch fit to
   one randomly sampled triangle match (3 over-determined constraints), which
   prevents cross-triangle vertex mixing.
5. Refine the transformation via the Kabsch / Umeyama least-squares algorithm
   applied to all vertex pairs from all inlier triangle matches.
6. Finally, warp `img_from` to the coordinates of `img_to`.
"""

# ╔═╡ c0b252bb-e621-45b6-987f-85f7a0211271
md"""
## Details
"""

# ╔═╡ a2ed7b77-1277-41a3-8c29-a9814b124d09
md"""
### Step 1: Identify control points

This step is done solely on the Photometry.jl side, which Astroalign.jl calls with some reasonable defaults.
"""

# ╔═╡ 2bc269e1-dbe3-4c68-9a30-8c6054bc3a82
md"""
#### Detect

`astroalign.py` uses [`sep`](https://github.com/quatrope/astroalign/blob/d7463b4ca48fc35f3d86a72343015491cdf20d6a/astroalign.py#L537) under the hood for its source extraction, so we'll use a combination of [`Photometry.extract_sources`](https://juliaastro.org/Photometry/stable/detection/#Photometry.Detection.extract_sources) to pull out the regions around the brightest pixels, and [`PSFModels.fit`](https://juliaastro.org/PSFModels/stable/api/#PSFModels.fit) to just pick out the ones that look like stars (vs. hot pixels, artifacts, etc.).

!!! todo
	Explore other point source detection algorithms in Julia.
"""

# ╔═╡ fe518d92-fbfd-4d6f-ba71-0b7b23a73fd7
md"""
#### Source extraction

Starting with `Astroalign.get_sources`, we identify candidate sources (`sources_to`) in the image that we would like to align to (`img_to`):
"""

# ╔═╡ c6ef3b26-ccc1-401b-ba9b-88276d4c5067
md"""
Since these are just locations of the brightest points in our image, some could be hot pixels or other artifacts. To address this, we can next filter them out by fitting a PSF to each source and only taking ones that meet a minimum FWHM.
"""

# ╔═╡ b0ad71b1-3a3c-481b-a08e-2ee558e8e1c5
md"""
#### Source characterization

[`Photometry.photometry`](https://juliaastro.org/Photometry/stable/apertures/#Photometry.Aperture.photometry) automatically computes aperture sums and returns them in a nice table for us. We also pass a function, `Astroalign.PSF`, to compute some PSF statistics for each source and stores them in the table as well.

!!! note
	Some PSF model fits may not converge for really noisy sources.
"""

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

# ╔═╡ 1a53b727-2553-468f-9105-134f682249a2
md"""
!!! note
	PSF centers are relative to the aperture, while `xcenter` and `ycenter` are relative to the whole image. Astroalign.jl performs the necessary conversions from the former to the latter in `Astroalign.to_subpixel` before reporting the final fitted values.
"""

# ╔═╡ 35befaff-e36c-4741-b28f-3589afe596cd
function inspect_psf(phot)
	psf_data, psf_model = phot.aperture_f.psf_data, phot.aperture_f.psf_model
	@debug (; phot.xcenter, phot.ycenter)
	@debug phot.aperture_f.psf_params
	return AstroImage(psf_data), imview(psf_model.(CartesianIndices(psf_data)))
end

# ╔═╡ c73692f2-178d-4b35-badc-e9e682551989
md"""
Looks to be fitting alright! From here, results can be sorted and filtered as needed. By default, Astrolign.jl sorts from largest to smallest FWHM.
"""

# ╔═╡ 255cb3ee-2ac4-4b20-8d4f-785ca9400668
md"""
### Step 2: Calculate invariants

This is done internally in `Astroalign.align_frame`, but the computed invariants ``\mathscr M_i`` can be exposed with `Astroalign.triangle_invariants` for plotting and debugging. Below is a plot comparing the compents of the computed invariants for all control points in our `from` and `to` images. Overlapping sections indicates similar triangle between images found by Astroalign.jl. Compare to Fig 1. in [Beroiz, M., Cabral, J. B., & Sanchez, B. (2020)](https://arxiv.org/pdf/1909.02946).
"""

# ╔═╡ d1d3f995-b901-4aab-86cd-e2d6f2393190
md"""
This can also be accessed through the named tuple returned by `Astroalign.align_frame` in the [Usage](#Usage) example. We will use this to get the corresponding invariants for our `from` image;
"""

# ╔═╡ 23a1364a-4ba0-42af-93bf-b6f900b9a13d
md"""
Note that the number of combinations of triangles in each frame can differ if the number of control points detected in each image is not the same. This can happen when sources drift towards the edge or off of the frame between images. All we need is one match though, which is helped by the additional combinations available.
"""

# ╔═╡ dffa0f3c-100f-4916-96c7-90274c0df5f2
md"""
### Step 3: Build candidates

We next build our list of candidate correspondences in this invariant space via a nearest neighbors search.
"""

# ╔═╡ 5dd82850-91d7-4b57-81df-e32dcc28eab9
md"""
### Step 4: Refine candidate list

Next, we find the largest set of mutually consistent correspondences (inliers) via a RANSAC pass with [JuliaAstro/ConsensusFitting.jl](https://github.com/ConsensusFitting.jl).
"""

# ╔═╡ 1c6b9f26-a418-4e47-8f6a-50a78f627ba8
# TODO: Come up with a better name?
function step4(correspondences; scale = false, ransac_threshold = 3.0)		
	fittingfn = scale ? Astroalign._fit_minimal_similarity_triangle : Astroalign._fit_minimal_rigid_triangle
    
	fwd_tfm, inlier_idxs = ransac(
        correspondences, fittingfn, Astroalign._triangle_distfn, 1, ransac_threshold;
    )
end

# ╔═╡ 1150fd19-ece7-4fd0-91db-a4df982d1e8e
md"""
### Step 5. Refine transformation

The previous step returns an initial proposed transformation and corresponding set of inliers, which we then successively refine using all detected control points. This lets us capture previously-missed inliers while dropping incorrectly assigned inliers in previous passes.
"""

# ╔═╡ e116588d-5793-446f-b5b9-1a13bc2733ad
md"""
We visualize the final set of corresponding control points below:
"""

# ╔═╡ 614bc7c4-6ba6-448b-9e82-aad968133622
function step_5(correspondences, fwd_tfm_initial, inlier_idxs_initial;
	scale = false,
	ransac_threshold = 3.0,
)
	fwd_tfm = fwd_tfm_initial
	inlier_idxs = inlier_idxs_initial
	for _ in 1:3
		isempty(inlier_idxs) && break
		# 2 × 3·N_inliers
		pts_from = reshape(correspondences[:, :, 1, inlier_idxs], 2, :)
		pts_to = reshape(correspondences[:, :, 2, inlier_idxs], 2, :)
		# 2 × 3·N_inliers
		new_fwd = kabsch(pts_from => pts_to; scale)   # from => to for scoring
		new_idxs, _ = Astroalign._triangle_distfn([new_fwd], correspondences, ransac_threshold)
		isempty(new_idxs) && break
		fwd_tfm = new_fwd
		inlier_idxs = new_idxs
	end
		
	# point_map: from-vertex => to-vertex for each vertex of each inlier match
	point_map = mapreduce(vcat, inlier_idxs) do i
		[correspondences[:, v, 1, i] => correspondences[:, v, 2, i] for v in 1:3]
	end
	
	return unique(point_map), inv(fwd_tfm)
end

# ╔═╡ 463f4963-4b5c-40f2-baaa-6f1180988990
md"""
### Step 6: Apply transformation

We can now hand off this transformation to an image transformation library like [JuliaAstroImages/ImageTransformations.jl](https://github.com/JuliaImages/ImageTransformations.jl) to view our final results. This should match our results returned by `Astroalign.align_frame` in the [Usage](#Usage) example.
"""

# ╔═╡ dd9296e8-0112-41e1-9ccc-4a3e813e2836
md"""
# 🔧 Notebook setup
"""

# ╔═╡ af8acb41-4cc8-4665-95f3-baaed36eed9d
# "Truth" values for transformation
const SCALE_0, ROT_0, TRANS_0 = 0.8, π/8, [10, 7]

# ╔═╡ f553bc81-dcc4-4e04-8173-beae8fe96249
"""
In this particular case, `img_from` is i) **scaled by a factor of $(SCALE_0)**, ii) **rotated counter-clockwise by $(round(rad2deg(ROT_0); digits = 1))° ($(round(ROT_0; digits = 3)) rad)**, and iii) **translated to the right $(first(TRANS_0)) pixels and up $(last(TRANS_0)) pixels** to arrive at `img_to` in the above plot. Let's fix it.
""" |> Markdown.parse

# ╔═╡ 39b81373-8029-4e9c-9ea4-732722cf645e
tfm_fwd_0 = Translation(TRANS_0...) ∘
	LinearMap(RotMatrix2(ROT_0)) ∘
	LinearMap(SCALE_0 * I)

# ╔═╡ dc01eaaa-f1d0-4bc6-884f-778d848918c6
const N_sources = 12

# ╔═╡ 6f943630-9a40-425e-8920-2911653e11d9
const RNG = Xoshiro(seed)

# ╔═╡ beee8408-60e9-444e-bcfa-19acf91a8171
const FWHMS = [rand(RNG, 1:10) for _ in 1:N_sources]

# ╔═╡ 78c0bf28-bb96-4aea-8bf5-5929ef45adc1
img_size = (1:300, 1:300)

# ╔═╡ eff56f6e-ab01-4371-a75f-f44bdde7cfd6
md"""
## Star field generator ✨

For simplicity, we'll just create $(N_sources) Gaussian point sources placed randomly in a $(length(first(img_size))) x $(length(last(img_size))) grid with some noise over the whole image. We can then check our fitted values against these "truth" values at the end.
"""

# ╔═╡ f7639401-1fc9-4cb1-824c-4335a4bb8b25
# Modified from
# https://github.com/JuliaAstro/PSFModels.jl/blob/main/test/fitting.jl
function generate_model(rng, model, params, inds)
	cartinds = CartesianIndices(inds)
	psf = model.(cartinds; params..., amp = 30_000)
	noise = rand(rng, 1000:3000, size(psf))
    return psf .+ noise
end

# ╔═╡ 95531bde-8386-4d51-8c83-ffb796a41e90
img_to = let
	# Uncomment for extended sources
	# fwhms = [(rand(rng, 1:20), rand(rng, 1:20)) for _ in 1:N_sources]
	pad = 10 # Minimum space etween the stars (in pixels)
	positions_to = rand(RNG, 1+pad:pad:300-pad, N_sources, 2)
	
	map(zip(eachrow(positions_to), FWHMS)) do ((x, y), fwhm)
		generate_model(RNG, gaussian, (; x, y, fwhm), img_size)
	end
end |> sum |> AstroImage;

# ╔═╡ fb0efcf4-26d4-4554-a5cf-b1136f5a6c17
# Sources, background subtracted image, background.
# Guards against extraneous matches by
# only selecting brightest 10 sources by default.
sources_to, subt_to, errs_to = get_sources(img_to);

# ╔═╡ 8afa31f0-ee57-4628-bedf-dd2b79faef72
begin
# Common to both images
box_size = Astroalign._compute_box_size(img_to)
ap_radius = 0.6 * first(box_size)
end; box_size, ap_radius

# ╔═╡ 07abbeb9-15a4-4086-86ca-093e5475c0db
aps_to = CircularAperture.(sources_to.y, sources_to.x, ap_radius)

# ╔═╡ 4e1c0615-d26d-4147-a096-d20940b8046a
phot_to = let
	phot = photometry(aps_to, subt_to; f = Astroalign.PSF())
	phot = Astroalign.to_subpixel(phot, aps_to)
	sort!(phot; by = x -> hypot(x.aperture_f.psf_params.fwhm...), rev = true)
end

# ╔═╡ 3da14f39-9fad-412e-824b-c3db190700aa
@bind i Slider(eachindex(phot_to); show_value=x -> "Source $(x)")

# ╔═╡ 68139ad3-cf00-4286-b9eb-a435dd20aca2
inspect_psf(phot_to[i])

# ╔═╡ c46335bc-ae9a-4257-8a85-b4ccb94d1744
C_to, ℳ_to = triangle_invariants(phot_to)

# ╔═╡ 5882adec-7591-4d93-98e2-efb81496c54d
img_from = let
	warp(img_to, tfm_fwd_0, axes(img_to);
		 fillvalue = ImageTransformations.Periodic(),
	)
end |> AstroImage;

# ╔═╡ 445a0d35-2b49-42cc-8529-176778b0e090
arr_from_aligned, params_aligned = align_frame(img_from, img_to;
	# box_size = (3, 3),
	# ap_radius = 20,
	# f = Astroalign.PSF(),
	# min_fwhm = box_size .÷ 5,
	# nsigma = 1,
	# N_max,
	scale = true,
	# ransac_threshold = 3.0,
	# final_iters = 3,
	# use_fitpos = true,
);

# ╔═╡ 30c3ecfc-f676-4bad-8a04-cc54fa3cf0c2
tfm_aligned = params_aligned.tfm

# ╔═╡ f72f1cd6-b3ff-4737-b7c2-bbca3e6c5b0f
PlutoUI.ExperimentalLayout.hbox(
	[
		"""
		Linear part:
		
		```julia
		$(repr("text/plain", tfm_aligned.linear))
		```
		""",
				
		"""
		Translation part:
		
		```julia
		$(repr("text/plain", tfm_aligned.translation))
		```
		"""
	] .|> Markdown.parse; style = Dict("gap" => "1rem"),
)

# ╔═╡ 94974b07-81b5-46dd-8643-6b70449ca912
S, R, T = decompose_tfm(inv(tfm_aligned))

# ╔═╡ 2cd7bbd0-8ced-4e8c-9c05-a16e9710bcc8
PlutoUI.ExperimentalLayout.hbox(
	[
		"""
		Scale:
		
		```julia
		$(repr("text/plain", round.(S; digits = 3)))
		```
		""",
				
		"""
		Rotation:
		
		```julia
		$(repr("text/plain", round.(R; digits = 3)))
		```
		""",

		"""
		Translation:
		
		```julia
		$(repr("text/plain", round.(T; digits = 3)))
		```
		"""
	] .|> Markdown.parse; style = Dict("gap" => "1rem"),
)

# ╔═╡ ab647cad-f3e3-4e1d-b9b2-e9d31612e9fc
params_tfm = (scale = S[1], rot = atan(R[2, 1], R[1, 1]), trans = T)

# ╔═╡ 1a099207-213c-4326-9b38-5ba4a8bf70b8
let
	n_inliers = length(params_aligned.inlier_idxs)
	n_total = size(params_aligned.correspondences, 4)
	perc = round(100 * n_inliers / n_total; digits = 1)

	"""
	which are within **$(p_diff(params_tfm.scale, SCALE_0))%, $(p_diff(params_tfm.rot, ROT_0))%, and $(p_diff(params_tfm.trans, TRANS_0))%** of our scale, rotation, and translation parameters used to originally transform `img_from` to `img_to`, respectively.
	
	Taking a look at our RANSAC pass, these final transformation values were determined from $(n_inliers) out of $(n_total) detected correspondences ($(perc) %) 
	
	For even more control and instrospection, each step of the alignment process has an associated API that can be used from Astroalign.jl, along with additional parameters returned by `Astroalign.align_frame`, which we show in the rest of this notebook.
	""" |> Markdown.parse
end

# ╔═╡ ad82de06-50f8-4e30-80b9-e4821e845162
(; C_from, ℳ_from) = params_aligned; C_from, ℳ_from

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

# ╔═╡ 2c3e0706-556a-4fdb-bbc5-85b5f90b3649
correspondences = Astroalign._build_correspondences(C_from, ℳ_from, C_to, ℳ_to)

# ╔═╡ 12d45273-f8c0-4b59-bcd6-296b4ebbb978
fwd_tfm_initial, inlier_idxs_initial = step4(correspondences;
	scale = true,
	ransac_threshold = 3.0,
)

# ╔═╡ 5041a969-a40e-49ee-8467-e1a38f81b7a6
# TODO: Come up with a better name?
point_map, tfm = step_5(correspondences, fwd_tfm_initial, inlier_idxs_initial;
	scale = true,
	ransac_threshold = 3.0,
)

# ╔═╡ bd2d9faf-7e0c-4a46-91e9-b3984dd3090e
aps_sol_from = map(point_map) do sol
	CircularAperture(sol.first[1], sol.first[2], ap_radius)
end;

# ╔═╡ 7f0b20db-e369-4e6a-aa5e-7df949791915
aps_sol_to = map(point_map) do sol
	CircularAperture(sol.second[1], sol.second[2], ap_radius)
end;

# ╔═╡ 7990c8be-9425-47d0-a913-9e2bb4fbefd1
img_aligned_from = shareheader(img_from, warp(img_from, tfm, axes(img_to)));

# ╔═╡ 1e8aaba0-645e-48c0-b4e1-b9e8f4c81c86
md"""
## Plotly helpers 🎨
"""

# ╔═╡ d00e04d9-7a12-481b-b3b3-5c1f7e31a1a7
# Global colorbar lims
const ZMIN, ZMAX = let
	# lims = Zscale(contrast=0.4).((img₁, img₂))
	lims = Percent(99.5).((img_to, img_from))
	minimum(first, lims), maximum(last, lims)
end

# ╔═╡ 1cf184a4-ec99-4cd2-8559-5d52b41ec629
function circ(ap; xref = :x, yref = :y, line_color = :lightgreen)
	circle(
		ap.x - ap.r, # x_min
		ap.x + ap.r, # x_max
		ap.y - ap.r, # y_min
		ap.y + ap.r; # y_max
		line_color,
		xref,
		yref
	)
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

# ╔═╡ 0612c049-c6d1-4e6a-a44a-b2f93a39a2c6
function make_aps(aps; line_color = nothing, xref= :x, yref = :y)
	N = length(aps)
	
	line_colors = if isnothing(line_color)
		[
			string("hsl(", round(Int, 360 * i / N), ", 70%, 55%)")
			for i in 1:N
		]
	else
		fill(line_color, N)
	end
	
	return [
		circ(ap; line_color, xref, yref)
		for (ap, line_color) in zip(aps, line_colors)
	]
end

# ╔═╡ de7ff589-99c0-4625-8a10-86aa702d2510
function plot_pair(img_left, img_right; column_titles=["img_left", "img_right"], aps_left = nothing, aps_right = nothing, aps_left_line_color = nothing, aps_right_line_color = nothing)
	# Set up some subplots
	fig = make_subplots(;
		rows = 1,	
		cols = 2,
		shared_xaxes = true,
		shared_yaxes = true,
		column_titles,
	)
	
	# Make the subplot titles a smidgen bit smaller
	update_annotations!(fig, font_size = 14)
	
	# Manually place the colorbars so they don't clash
	p1 = add_trace!(fig, trace_hm(img_left; colorbar_x = 0.45), col = 1)
	p2 = add_trace!(fig, trace_hm(img_right; colorbar_x = 1), col = 2)

	
	# Keep the images true to size
	update_xaxes!(fig, matches = "x", scaleanchor = :y, title = "X (pixels)")
	update_yaxes!(fig, matches = "y", scaleanchor = :x)

	# Add a shared y-label
	relayout!(fig, Layout(yaxis_title = "Y (pixels)");
		font_size = 10,
		template = "plotly_white",
		margin = attr(t = 20),
		uirevision = 1,
		shapes = [
			(isnothing(aps_left) ? [] : make_aps(aps_left;  line_color=aps_left_line_color, xref = :x,  yref = :y))...,
			(isnothing(aps_right) ? [] : make_aps(aps_right; line_color=aps_right_line_color, xref= :x2, yref = :y2))...,
    	]
	)

	# Display
	return fig
end

# ╔═╡ f128f050-b716-4a79-8bb6-640708d1bc88
plot_pair(img_from, img_to; column_titles = ["img_from", "img_to"])

# ╔═╡ 8769216b-00d4-44bd-97fd-7aa89cf19c23
plot_pair(arr_from_aligned, img_to; column_titles = ["img_from (aligned)", "img_to"])

# ╔═╡ 391d8c47-21f4-43b8-85ce-7c2e943fde9d
plot_pair(img_from, img_to;
	column_titles = ["img_from", "img_to"],
	# aps_left = aps_sol_from,
	# aps_left_line_color = :lightgreen,
	aps_right = aps_to,
	aps_right_line_color = :lightgreen,
)

# ╔═╡ 41136c70-f0ed-435b-b449-5d71e04e9c35
plot_pair(img_from, img_to;
	column_titles = ["img_from", "img_to"],
	aps_left = aps_sol_from,
	# aps_left_line_color = :lightgreen,
	aps_right = aps_sol_to,
	# aps_right_line_color = :lightgreen,
)

# ╔═╡ 066210ea-b5b3-4f73-8fc1-503625fc32ce
fig = plot_pair(img_aligned_from, img_to;
	column_titles = ["img_aligned_from", "img_to"]
)

# ╔═╡ 84ce90fc-f8a9-47ac-8f3f-c83899027a4d
md"""
## Packages 📦
"""

# ╔═╡ 5e09f7eb-a4af-4d94-8684-96857e716747
TableOfContents(; depth = 4)

# ╔═╡ Cell order:
# ╟─9e130a37-1073-4d0f-860a-0ec8d164dde1
# ╟─fa1180d4-c1ea-4a1b-8476-0e8d185d5622
# ╟─40c14093-3806-401f-aedf-f6435f785eb4
# ╟─c5bfce23-d050-42e3-8af2-f1181adaaa2d
# ╟─f128f050-b716-4a79-8bb6-640708d1bc88
# ╟─f553bc81-dcc4-4e04-8173-beae8fe96249
# ╟─8769216b-00d4-44bd-97fd-7aa89cf19c23
# ╠═445a0d35-2b49-42cc-8529-176778b0e090
# ╟─7c1942c2-f61c-4c17-a0a5-0701c19d3d4f
# ╟─fde0d2e4-e8ce-4861-8d53-43d58c9f8fe1
# ╠═30c3ecfc-f676-4bad-8a04-cc54fa3cf0c2
# ╟─f72f1cd6-b3ff-4737-b7c2-bbca3e6c5b0f
# ╟─0c7b43c9-0456-433c-800d-1234b66f54a0
# ╟─2cd7bbd0-8ced-4e8c-9c05-a16e9710bcc8
# ╠═94974b07-81b5-46dd-8643-6b70449ca912
# ╟─25b3db66-d22d-4372-a15b-02031aeb01d4
# ╠═ab647cad-f3e3-4e1d-b9b2-e9d31612e9fc
# ╟─1a099207-213c-4326-9b38-5ba4a8bf70b8
# ╟─6e44a52d-cc2a-45eb-ade3-001488cd2f49
# ╟─c84e1690-2176-427f-b8b2-eb5feacdcc2d
# ╟─f47dd317-6ac6-4f5f-95f8-eaca3d2820a6
# ╟─c5658a61-99e2-4008-a542-9e12bf70ee9b
# ╟─c0b252bb-e621-45b6-987f-85f7a0211271
# ╟─a2ed7b77-1277-41a3-8c29-a9814b124d09
# ╟─2bc269e1-dbe3-4c68-9a30-8c6054bc3a82
# ╟─fe518d92-fbfd-4d6f-ba71-0b7b23a73fd7
# ╠═fb0efcf4-26d4-4554-a5cf-b1136f5a6c17
# ╠═8afa31f0-ee57-4628-bedf-dd2b79faef72
# ╠═07abbeb9-15a4-4086-86ca-093e5475c0db
# ╠═391d8c47-21f4-43b8-85ce-7c2e943fde9d
# ╟─c6ef3b26-ccc1-401b-ba9b-88276d4c5067
# ╟─b0ad71b1-3a3c-481b-a08e-2ee558e8e1c5
# ╠═4e1c0615-d26d-4147-a096-d20940b8046a
# ╟─fcb02cf0-4fb5-4e31-bab9-d19a0755def9
# ╟─9109a7a0-4a37-4dca-a923-16a9302556ee
# ╟─3da14f39-9fad-412e-824b-c3db190700aa
# ╠═68139ad3-cf00-4286-b9eb-a435dd20aca2
# ╟─1a53b727-2553-468f-9105-134f682249a2
# ╟─35befaff-e36c-4741-b28f-3589afe596cd
# ╟─c73692f2-178d-4b35-badc-e9e682551989
# ╟─255cb3ee-2ac4-4b20-8d4f-785ca9400668
# ╟─cdba7937-eea8-409a-b9e3-714e4516486c
# ╠═c46335bc-ae9a-4257-8a85-b4ccb94d1744
# ╟─d1d3f995-b901-4aab-86cd-e2d6f2393190
# ╠═ad82de06-50f8-4e30-80b9-e4821e845162
# ╟─23a1364a-4ba0-42af-93bf-b6f900b9a13d
# ╟─dffa0f3c-100f-4916-96c7-90274c0df5f2
# ╠═2c3e0706-556a-4fdb-bbc5-85b5f90b3649
# ╟─5dd82850-91d7-4b57-81df-e32dcc28eab9
# ╠═12d45273-f8c0-4b59-bcd6-296b4ebbb978
# ╟─1c6b9f26-a418-4e47-8f6a-50a78f627ba8
# ╟─1150fd19-ece7-4fd0-91db-a4df982d1e8e
# ╠═5041a969-a40e-49ee-8467-e1a38f81b7a6
# ╟─e116588d-5793-446f-b5b9-1a13bc2733ad
# ╠═bd2d9faf-7e0c-4a46-91e9-b3984dd3090e
# ╠═7f0b20db-e369-4e6a-aa5e-7df949791915
# ╠═41136c70-f0ed-435b-b449-5d71e04e9c35
# ╟─614bc7c4-6ba6-448b-9e82-aad968133622
# ╟─463f4963-4b5c-40f2-baaa-6f1180988990
# ╠═7990c8be-9425-47d0-a913-9e2bb4fbefd1
# ╠═066210ea-b5b3-4f73-8fc1-503625fc32ce
# ╟─dd9296e8-0112-41e1-9ccc-4a3e813e2836
# ╟─eff56f6e-ab01-4371-a75f-f44bdde7cfd6
# ╠═af8acb41-4cc8-4665-95f3-baaed36eed9d
# ╠═39b81373-8029-4e9c-9ea4-732722cf645e
# ╠═dc01eaaa-f1d0-4bc6-884f-778d848918c6
# ╠═beee8408-60e9-444e-bcfa-19acf91a8171
# ╠═6f943630-9a40-425e-8920-2911653e11d9
# ╠═78c0bf28-bb96-4aea-8bf5-5929ef45adc1
# ╠═95531bde-8386-4d51-8c83-ffb796a41e90
# ╠═5882adec-7591-4d93-98e2-efb81496c54d
# ╠═f7639401-1fc9-4cb1-824c-4335a4bb8b25
# ╟─1e8aaba0-645e-48c0-b4e1-b9e8f4c81c86
# ╠═d00e04d9-7a12-481b-b3b3-5c1f7e31a1a7
# ╟─1cf184a4-ec99-4cd2-8559-5d52b41ec629
# ╟─b461aadf-f88c-4195-8715-35e1e24a9bb4
# ╟─0612c049-c6d1-4e6a-a44a-b2f93a39a2c6
# ╟─de7ff589-99c0-4625-8a10-86aa702d2510
# ╟─84ce90fc-f8a9-47ac-8f3f-c83899027a4d
# ╠═5e09f7eb-a4af-4d94-8684-96857e716747
# ╠═d8d4c414-64a0-11f0-15a3-0d566872a687
