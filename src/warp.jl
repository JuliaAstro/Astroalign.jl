"""
    function align_frame(img_from, img_to; [warp_function], kwargs...)

Align `img_from` onto `img_to`.

Accepts an optional `warp_function` to specify the coordinate transformation (warping) function to use. The function maintains the call signature `warp_function(img_from, inv(tfm), axes(img_to))`, with the input image `img_from`, the transform to apply `inv(tfm)` and the `axes()` of the destination `img_to`. By default [`ImageTransformations.warp`](https://juliaimages.org/ImageTransformations.jl/stable/reference/#ImageTransformations.warp) is used. Note that `warp_function` can potentially modify inputs provided via Julia's closure mechanism.

Additional keyword arguments are forwarded to [`find_transform`](@ref).

# Extended help

Alignment algorithm:

1. Identify the `N_max` brightest point-like sources in `img_from` and `img_to`.
2. Calculate all triangular asterisms formed from these sources.
3. Build a `2 × 3 × 2 × N` array of candidate triangle-level correspondences
   by matching each from-triangle to its nearest to-triangle in
   the invariant ``ℳ`` space defined by [Beroiz et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract).
   Vertices are assigned via a canonical ordering that is invariant under
   rotation, so the positional correspondence between matched triangles is
   geometrically consistent. The axes are `[coord, vertex, frame, match]`
   where `coord ∈ {x, y}`, `vertex ∈ {1,2,3}`, and `frame ∈ {from, to}`.
4. Run RANSAC ([Fischler & Bolles, 1981](https://dl.acm.org/doi/10.1145/358669.358692))
   on the triangle matches to robustly identify the largest set of mutually
   consistent correspondences ("inliers"). Each hypothesis is a Kabsch fit to
   one randomly sampled triangle match (3 over-determined constraints), which
   prevents cross-triangle vertex mixing.
5. Refine the transformation via the Kabsch / Umeyama least-squares algorithm
   applied to all vertex pairs from all inlier triangle matches.
6. Finally, warp `img_from` to the coordinates of `img_to`.
"""
function align_frame(img_from, img_to; warp_function = warp, kwargs...)
    # Steps 1 - 5
    tfm, tfm_params = find_transform(img_from, img_to; kwargs...)

    # Step 6: Apply the transform (from => to)
    warp_img = warp_function(img_from, inv(tfm), axes(img_to))

    return warp_img, (; tfm, tfm_params...)
end

"""
    _ransac(correspondences; scale, ransac_threshold)

    RANSAC on triangle matches to find the largest set of mutually consistent correspondences (inliers).
"""
function _ransac(correspondences; scale, ransac_threshold)
    fittingfn = scale ? _fit_minimal_similarity_triangle : _fit_minimal_rigid_triangle
    tfm, inlier_idxs = ransac(correspondences, fittingfn, _triangle_distfn, 1, ransac_threshold)
    return tfm, inlier_idxs
end

"""
    _refine_transform(tfm, inlier_idxs, correspondences; final_iters, scale, ransac_threshold)

Finalize matches returned by [`Astroalign._ransac`](@ref) by iteratively refining the transform.
"""
function _refine_transform(tfm, inlier_idxs, correspondences; final_iters, scale, ransac_threshold)
    for _ in 1:final_iters
        isempty(inlier_idxs) && break
        pts_from = reshape(correspondences[:, :, 1, inlier_idxs], 2, :)  # 2 × 3·N_inliers
        pts_to = reshape(correspondences[:, :, 2, inlier_idxs], 2, :)  # 2 × 3·N_inliers
        new_fwd = kabsch(pts_from => pts_to; scale)   # from => to for scoring
        new_idxs, _ = _triangle_distfn([new_fwd], correspondences, ransac_threshold)
        isempty(new_idxs) && break
        tfm = new_fwd
        inlier_idxs = new_idxs
    end

    # point_map: from-vertex => to-vertex for each vertex of each inlier match
    point_map = mapreduce(vcat, inlier_idxs) do i
        [correspondences[:, v, 1, i] => correspondences[:, v, 2, i] for v in 1:3]
    end |> unique

    return tfm, inlier_idxs, point_map
end

"""
    function find_transform(img_from, img_to;
        [box_size],
        [ap_radius],
        [f],
        [min_fwhm],
        [nsigma],
        [N_max],
        [scale],
        [ransac_threshold],
        [final_iters],
        [use_fitpos],
    )

Compute the transformation needed to align `img_from` onto `img_to`, assuming both images are related via a rigid
(or similarity, when `scale = true`) transformation. Automatically called by [`align_frame`](@ref).

# Parameters

- `box_size`: The size of the grid cells (in pixels) used to extract candidate point sources to use for alignment. Defaults to (3, 3) pixels. See [Photometry.jl > Source Detection Algorithms](@extref Photometry Source-Detection-Algorithms) for more.
- `ap_radius`: The radius of the apertures (in pixel) to place around each point source. Defaults to 9 pixels. See [Photometry.jl > Aperture Photometry](@extref Photometry Aperture-Photometry) for more.
- `f`: The function to compute within each aperture. Defaults to a 2D Gaussian fitted to the aperture center, with default FWHM of 1.5 pixels. See the [Source characterization](https://juliaastro.org/Astroalign.jl/notebook.html#Source-characterization) section of the accompanying Pluto.jl notebook for more.
- `min_fwhm`: The minimum FWHM (in pixels) that an extracted point source must have to be considered as a control point. Defaults to 2 pixels. See [PSFModels.jl > Fitting data](@extref PSFModels Fitting-data) for more. Set to `nothing` to use all identified sources as control points.
- `nsigma`: The number of standard deviations above the estimated background that a source must be to be considered as a control point. Defaults to 1. See [Photometry.jl > Source Detection Algorithms](@extref Photometry Source-Detection-Algorithms) for more.
- `N_max`: Maximal Number of (brightest) sources to consider for alignment (default is 10).
- `scale`: If `true`, fit a similarity transformation (rotation + isotropic scale + translation) instead of a rigid transformation (rotation + translation only). Defaults to `false`.
- `ransac_threshold`: Pixel-distance threshold below which a correspondence is classified as an inlier by RANSAC. Defaults to `3.0`.
- `final_iters`: Number of iterative-refinement passes after RANSAC (default `3`). Each pass fits a new transform from the current inlier set and re-scores all correspondences to admit new inliers or drop old ones.
- `use_fitpos`: if `true` (default), the fit results are used in the position estimate for the triangles and thus the alignment.
"""
function find_transform(img_from, img_to;
    box_size = (3, 3),
    ap_radius = 9,
    f = PSF(),
    min_fwhm = 2.0,
    nsigma = 1,
    N_max = 10,
    use_fitpos = true,
    scale = false,
    ransac_threshold = 3.0,
    final_iters = 3,
)
    # Step 1: Identify control points
    phot_from, phot_from_params = _photometry(img_from; box_size, ap_radius, f, min_fwhm, nsigma, N_max, use_fitpos)
    phot_to, phot_to_params = _photometry(img_to; box_size, ap_radius, f, min_fwhm, nsigma, N_max, use_fitpos)

    # Step 2: Calculate invariants
    C_from, ℳ_from = _triangle_invariants(phot_from)
    C_to, ℳ_to = _triangle_invariants(phot_to)

    # Step 3: Build candidate correspondence pool via nearest neighbors triangle matching
    correspondences = _build_correspondences(C_from, ℳ_from, C_to, ℳ_to)

    size(correspondences, 4) < 1 &&
        error("align_frame: not enough candidate correspondences ($(size(correspondences, 4))); " *
              "ensure both images contain at least 3 detectable point sources")

    # Step 4: RANSAC on triangle matches to find the largest set of mutually consistent correspondences (inliers)
    fwd_tfm_initial, inlier_idxs_initial = _ransac(correspondences; scale, ransac_threshold)

    # Step 5: Finalize the result by iteratively refining the transform.
    # Each pass: fit a new forward (from => to) transform on the current inlier set,
    # then re-score all correspondences to update inlier_idxs. Using the full
    # array (not the previous inlier subset) lets previously-missed inliers be
    # recovered and incorrectly accepted ones drop out.
    # Note that _triangle_distfn expects a from => to transform.
    tfm, inlier_idxs, point_map = _refine_transform(fwd_tfm_initial, inlier_idxs_initial, correspondences; final_iters, scale, ransac_threshold)

    return tfm, (; point_map, correspondences, inlier_idxs, C_from, ℳ_from, C_to, ℳ_to, phot_from_params, phot_to_params)
end
