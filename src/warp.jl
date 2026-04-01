"""
    function align_frame(img_to, img_from;
        [box_size],
        [ap_radius],
        [f],
        [min_fwhm],
        [nsigma],
    )

Align `img_from` onto `img_to`. See below for keyword arguments currently available to control this process.

* `box_size`: The size of the grid cells (in pixels) used to extract candidate point sources to use for alignment. Defaults to a tenth of the greatest common denominator of the dimensions of `img_to`. See [Photometry.jl > Source Detection Algorithms](@extref Photometry Source-Detection-Algorithms) for more.
* `ap_radius`: The radius of the apertures (in pixel) to place around each point source. Defaults to 60% of `first(box_size)`. See [Photometry.jl > Aperture Photometry](@extref Photometry Aperture-Photometry) for more.
* `f`: The function to compute within each aperture. Defaults to a 2D Gaussian fitted to the aperture center. See the [Source characterization](https://juliaastro.org/Astroalign.jl/notebook.html#Source-characterization) section of the accompanying Pluto.jl notebook for more.
* `min_fwhm`: The minimum FWHM (in pixels) that an extracted point source must have to be considered as a control point. Defaults to a fifth of the width of the first image. See [PSFModels.jl > Fitting data](@extref PSFModels Fitting-data) for more.
* `nsigma`: The number of standard deviations above the estimated background that a source must be to be considered as a control point. Defaults to 1. See [Photometry.jl > Source Detection Algorithms](@extref Photometry Source-Detection-Algorithms) for more.
"""
function align_frame(img_to, img_from;
        box_size = _compute_box_size(img_to),
        ap_radius = 0.6 * first(box_size),
        f = PSF(),
        min_fwhm = box_size .÷ 5,
        nsigma = 1,
    )
    # Step 1: Identify control points
    phot_to = _photometry(img_to, box_size, ap_radius, min_fwhm, nsigma, f; filter_fwhm=true)
    phot_from = _photometry(img_from, box_size, ap_radius, min_fwhm, nsigma, f; filter_fwhm=true)

    # Step 2: Calculate invariants
    C_to, ℳ_to = triangle_invariants(phot_to)
    C_from, ℳ_from = triangle_invariants(phot_from)

    # Step 3: Select nearest
    sol_to, sol_from = find_nearest(C_to, ℳ_to, C_from, ℳ_from)

    # Transform
    point_map = map(sol_to, sol_from) do source_to, source_from
        [source_from.xcenter, source_from.ycenter] => [source_to.xcenter, source_to.ycenter]
    end

    tfm = kabsch(last.(point_map) => first.(point_map); scale=false)

    return (
        warp(img_from, tfm, axes(img_to)),
        (;
            point_map,
            tfm,
            C_to,
            ℳ_to,
            C_from,
            ℳ_from,
       )
    )
end
