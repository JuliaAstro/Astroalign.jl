"""
    function align_frame(img_to, img_from;
        [box_size],
        [ap_radius],
        [f],
        [min_fwhm],
        [nsigma],
        [N_max],
        [scale],
        [ransac_threshold],
        [k_nearest],
        [final_iters],
        [use_fitpos],
    )

Align `img_from` onto `img_to`, assuming both images are related via a rigid
(or similarity, when `scale = true`) transformation.

This is achieved via the following algorithm:

1. Identify the `N_max` brightest point-like sources in `img_from` and `img_to`.
2. Calculate all triangular asterisms formed from these sources.
3. Build a `2 × 3 × 2 × N` array of candidate triangle-level correspondences
   by matching each from-triangle to its `k_nearest` nearest to-triangles in
   the invariant ``\\mathscr M`` space defined by [Beroiz et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract).
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

# Parameters

- `box_size`: The size of the grid cells (in pixels) used to extract candidate point sources to use for alignment. Defaults to a tenth of the greatest common denominator of the dimensions of `img_to`. See [Photometry.jl > Source Detection Algorithms](@extref Photometry Source-Detection-Algorithms) for more.
- `ap_radius`: The radius of the apertures (in pixel) to place around each point source. Defaults to 60% of `first(box_size)`. See [Photometry.jl > Aperture Photometry](@extref Photometry Aperture-Photometry) for more.
- `f`: The function to compute within each aperture. Defaults to a 2D Gaussian fitted to the aperture center. See the [Source characterization](https://juliaastro.org/Astroalign.jl/notebook.html#Source-characterization) section of the accompanying Pluto.jl notebook for more.
- `min_fwhm`: The minimum FWHM (in pixels) that an extracted point source must have to be considered as a control point. Defaults to a fifth of the width of the first image. See [PSFModels.jl > Fitting data](@extref PSFModels Fitting-data) for more.
- `nsigma`: The number of standard deviations above the estimated background that a source must be to be considered as a control point. Defaults to 1. See [Photometry.jl > Source Detection Algorithms](@extref Photometry Source-Detection-Algorithms) for more.
- `N_max`: Maximal Number of (brightest) sources to consider for alignment (default is 10).
- `scale`: If `true`, fit a similarity transformation (rotation + isotropic scale + translation) instead of a rigid transformation (rotation + translation only). Defaults to `false`.
- `ransac_threshold`: Pixel-distance threshold below which a correspondence is classified as an inlier by RANSAC. Defaults to `3.0`.
- `k_nearest`: Number of nearest triangles (in invariant space) to consider per from-triangle when building the correspondence pool. Larger values increase robustness at the cost of a larger RANSAC data set. Defaults to `5`.
- `final_iters`: Number of iterative-refinement passes after RANSAC (default `3`). Each pass fits a new transform from the current inlier set and re-scores all correspondences to admit new inliers or drop old ones.
- `use_fitpos`: if `true` (default), the fit results are used in the position estimate for the triangles and thus the alignment.
"""
function align_frame(img_to, img_from;
    box_size = _compute_box_size(img_to),
    ap_radius = 0.6 * first(box_size),
    f = PSF(),
    min_fwhm = box_size .÷ 5,
    nsigma = 1,
    N_max = 10,
    scale::Bool = false,
    ransac_threshold::Real = 3.0,
    k_nearest::Integer = 5,
    final_iters::Int = 3,
    use_fitpos = true,
)
    ransac_threshold = float(ransac_threshold)

    # Step 1: Identify control points
    phot_to = _photometry(img_to, box_size, ap_radius, min_fwhm, nsigma, f; N_max, filter_fwhm = true, use_fitpos)
    phot_from = _photometry(img_from, box_size, ap_radius, min_fwhm, nsigma, f; N_max, filter_fwhm = true, use_fitpos)

    # Step 2: Calculate invariants
    C_to,   ℳ_to   = triangle_invariants(phot_to)
    C_from, ℳ_from = triangle_invariants(phot_from)

    # Step 3: Build candidate correspondence pool via k-NN triangle matching
    correspondences = _build_correspondences(C_to, ℳ_to, C_from, ℳ_from; k = k_nearest)

    size(correspondences, 4) < 1 &&
        error("align_frame: not enough candidate correspondences ($(size(correspondences, 4))); " *
              "ensure both images contain at least 3 detectable point sources")

    # Step 4: RANSAC on triangle matches to find the largest set of mutually consistent correspondences (inliers)
    fittingfn = scale ? _fit_minimal_similarity_triangle : _fit_minimal_rigid_triangle
    fwd_tfm, inlier_idxs = ransac(
        correspondences, fittingfn, _triangle_distfn, 1, ransac_threshold;
    )

    # Step 5: Finalize the result by iteratively refining the transform.
    # Each pass: fit a new forward (from→to) transform on the current inlier set,
    # then re-score all correspondences to update inlier_idxs. Using the full
    # array (not the previous inlier subset) lets previously-missed inliers be
    # recovered and incorrectly accepted ones drop out.
    # Note that _triangle_distfn expects a from→to transform.
    for _ in 1:final_iters
        isempty(inlier_idxs) && break
        pts_from = reshape(correspondences[:, :, 1, inlier_idxs], 2, :)  # 2 × 3·N_inliers
        pts_to   = reshape(correspondences[:, :, 2, inlier_idxs], 2, :)  # 2 × 3·N_inliers
        new_fwd  = kabsch(pts_from => pts_to; scale)   # from→to for scoring
        new_idxs, _ = _triangle_distfn([new_fwd], correspondences, ransac_threshold)
        isempty(new_idxs) && break
        fwd_tfm     = new_fwd
        inlier_idxs = new_idxs
    end

    # Invert the transform (to→from) for warp
    tfm = inv(fwd_tfm)

    # point_map: from-vertex => to-vertex for each vertex of each inlier match
    point_map = mapreduce(vcat, inlier_idxs) do i
        [correspondences[:, v, 1, i] => correspondences[:, v, 2, i] for v in 1:3]
    end

    # Step 6: Apply transformation
    return (
        warp(img_from, tfm, axes(img_to)),
        (;
            point_map,
            tfm,
            correspondences,
            inlier_idxs,
            C_to,
            ℳ_to,
            C_from,
            ℳ_from,
        )
    )
end
