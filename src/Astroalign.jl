module Astroalign

using Combinatorics: combinations
using CoordinateTransformations: kabsch
using Distances: euclidean
using ImageTransformations: warp
using NearestNeighbors: nn, KDTree
using PSFModels: gaussian, fit
using Photometry: estimate_background,
                  extract_sources,
                  photometry,
                  sigma_clip,
                  CircularAperture,
                  PeakMesh

export align_frame, get_sources, find_nearest, triangle_invariants


"""
    get_sources(img; box_size = nothing, nsigma = 1)

Extract candidate sources in `img` according to [`Photometry.Detection.extract_sources`](@extref). By default, `img` is first sigma clipped and then background subtracted before the candidate sources are extracted. `box_size` is passed to [`Photometry.Background.estimate_background`](@extref), and `nsigma` is passed to [`Photometry.Detection.extract_sources`](@extref). See the [Photometry.jl](@extref) documentation for more.

TODO: Pass more options to clipping, background estimating, and extraction methods in [Photometry.jl](@extref).
"""
function get_sources(img; box_size = _compute_box_size(img), nsigma = 1, N_max = 10)
    # Background subtract `img`
    clipped = sigma_clip(img, 1, fill = NaN)
    bkg, bkg_rms = estimate_background(clipped, box_size)
    subt = img .- bkg[axes(img)...]

    return (
        # Sort detected sources from brightest to darkest
        first(extract_sources(PeakMesh(; box_size, nsigma), subt, bkg, true), N_max),
        # And also return the inputs, handy for debugging and data viz
        subt,
        bkg,
    )
end

Base.@kwdef struct PSF
    model = gaussian
    params = (;)
    func_kwargs = (;)
    kwargs = (; x_abstol = 2e-6)
end

(p::PSF)(img) = fit_psf(img, p)

function fit_psf(img_ap, p)
    # Normalize
    psf_data = collect(Float32, img_ap)
    psf_data ./= maximum(psf_data)

    # Set params
    if isempty(p.params)
        y, x = Tuple(argmax(psf_data))
        fwhm = _compute_box_size(img_ap)
        params = (; x, y, fwhm)
    else
        params = p.params
    end

    # Fit
    psf_params, psf_model = fit(p.model, params, psf_data; func_kwargs=p.func_kwargs, p.kwargs...)

    return (; psf_params, psf_model, psf_data)
end

# Internal function used by `align`
# Calls to `Photometry.photometry` with reasonable defaults
function _photometry(img, box_size, ap_radius, min_fwhm, nsigma, f; filter_fwhm)
    # Sources, background subtracted image, background
    sources, subt, _ = get_sources(img; box_size, nsigma)

    # Define apertures
    aps = CircularAperture.(sources.y, sources.x, ap_radius)

    phot = photometry(aps, subt; f)

    if filter_fwhm
        filter!(phot) do source
            hypot(min_fwhm...) ≤ hypot(source.aperture_f.psf_params.fwhm...)
        end
    end

    sort!(phot; by = x -> hypot(x.aperture_f.psf_params.fwhm...), rev = true)

    return phot
end

"""
    triangle_invariants(phot)

Returns all combinations (``C``) of three candidate point sources from the table of sources `phot` returned by [`Photometry.Aperture.photometry`](@extref), and the computed invariant ``\\mathscr M`` for each according to Eq. 3 from [_Beroiz, M., Cabral, J. B., & Sanchez, B. (2020)_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract).
"""
function triangle_invariants(phot)
    C = combinations(phot, 3)
    ℳ = map(C) do (pa, pb, pc)
        a, b, c = (
            (pa.ycenter, pa.xcenter),
            (pb.ycenter, pb.xcenter),
            (pc.ycenter, pc.xcenter),
        )
        Ls = sort!([euclidean(a, b), euclidean(b, c), euclidean(a, c)])
        (Ls[3] / Ls[2], Ls[2] / Ls[1])
    end |> stack
    return C, ℳ
end

"""
    find_nearest(C_to, ℳ_to, C_from, ℳ_from)

Return the closes pair of three points between the `from` and `to` frames in the invariant ``\\mathscr M`` space as computed by [`Astroalign.triangle_invariants`](@ref).
"""
function find_nearest(C_to, ℳ_to, C_from, ℳ_from)
    idxs, dists = nn(KDTree(ℳ_to), ℳ_from)
    idx_from = argmin(dists)
    idx_to = idxs[idx_from]
    sol_to = collect(C_to)[idx_to]
    sol_from = collect(C_from)[idx_from]
    return sol_to, sol_from
end

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

function _compute_box_size(img)
    w = gcd(size(img)...) ÷ 10
    box_width = iseven(w) ? w + 1 : w
    return (box_width, box_width)
end

end # module
