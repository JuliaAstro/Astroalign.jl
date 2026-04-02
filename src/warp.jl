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
    )

Align `img_from` onto `img_to`, assuming both images are related via a rigid
(or similarity, when `scale = true`) transformation.

This is achieved via the following algorithm:

1. Identify the `N_max` brightest point-like sources in `img_from` and `img_to`.
2. Calculate all triangular asterisms formed from these sources.
3. Build a pool of candidate point correspondences by matching each
   from-triangle to its `k_nearest` nearest to-triangles in the invariant
   ``\\mathscr M`` space defined by [Beroiz et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract).
   Vertices are assigned via a canonical ordering that is invariant under
   rotation, so the positional correspondence between matched triangles is
   geometrically consistent.
4. Run RANSAC ([Fischler & Bolles, 1981](https://dl.acm.org/doi/10.1145/358669.358692))
   to robustly identify the largest set of mutually consistent correspondences
   ("inliers").  Each RANSAC hypothesis is determined analytically from two
   randomly sampled correspondences, which is the minimal sample needed to
   fix a rigid (or similarity) transform in 2-D.
5. Refine the transformation via the Kabsch / Umeyama least-squares algorithm
   applied to all inlier correspondences.
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
)

    # Step 1: Identify control points
    phot_to   = _photometry(img_to,   box_size, ap_radius, min_fwhm, nsigma, f; filter_fwhm = true)
    phot_from = _photometry(img_from, box_size, ap_radius, min_fwhm, nsigma, f; filter_fwhm = true)

    # Step 2: Calculate invariants
    C_to,   ℳ_to   = triangle_invariants(phot_to)
    C_from, ℳ_from = triangle_invariants(phot_from)

    # Step 3: Build candidate correspondence pool via k-NN triangle matching
    correspondences = _build_correspondences(C_to, ℳ_to, C_from, ℳ_from; k = k_nearest)

    size(correspondences, 2) < 2 &&
        error("align_frame: not enough candidate correspondences ($(size(correspondences, 2))); " *
              "ensure both images contain at least 3 detectable point sources")

    # Step 4: RANSAC – robustly identify the largest consensus set
    fittingfn = scale ? _fit_minimal_similarity : _fit_minimal_rigid
    best_M_fwd, inlier_idxs = ransac(
        correspondences, fittingfn, _correspondence_distfn, 2, Float64(ransac_threshold);
    )

    # Step 5: Refine solution with Kabsch / Umeyama on all inliers
    #   kabsch(to_pts => from_pts) gives backward transform: to → from (needed by warp)
    pts_from = correspondences[1:2, inlier_idxs]
    pts_to   = correspondences[3:4, inlier_idxs]
    tfm = kabsch(pts_to => pts_from; scale)

    # point_map mirrors the old format: [x_from, y_from] => [x_to, y_to] per inlier
    point_map = map(inlier_idxs) do i
        correspondences[1:2, i] => correspondences[3:4, i]
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
