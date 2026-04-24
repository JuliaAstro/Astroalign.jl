"""
    _get_sources(img; box_size = nothing, nsigma = 1, N_max = 10)

Extract candidate sources in `img` according to [`Photometry.Detection.extract_sources`](@extref). By default, `img` is first sigma clipped and then background subtracted before the candidate sources are extracted. `box_size` is passed to [`BackgroundMeshes.estimate_background`](@extref), and `nsigma` is passed to [`Photometry.Detection.extract_sources`](@extref). See the [Photometry.jl](@extref) documentation for more.

TODO: Pass more options to clipping, background estimating, and extraction methods in [Photometry.jl](@extref).
"""
function _get_sources(img; box_size, nsigma, N_max)
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
        bkg_rms,
    )
end

"""
    _photometry(img; box_size, ap_radius, min_fwhm, nsigma, f, N_max, use_fitpos)

Internal function used by [`align_frame`](@ref). Calls to [`Photometry.Aperture.photometry`](@extref) with reasonable defaults.

See [`align_frame`](@ref) for keyword arguments.
"""
function _photometry(img; box_size, ap_radius, min_fwhm, nsigma, f, N_max, use_fitpos)
    # Sources, background subtracted image, background
    sources, subt, bkg, bkg_rms = _get_sources(img; box_size, nsigma, N_max)

    # Define apertures
    aps = CircularAperture.(sources.y, sources.x, ap_radius)

    # Fit using the PSF model
    phot = photometry(aps, subt; f)

    # Improve the coordinates estimate with the fit results
    if use_fitpos
        phot = to_subpixel(phot, aps)
    end

    if !isnothing(min_fwhm)
        filter!(phot) do source
            hypot(source.aperture_f.psf_params.fwhm...) ≥ hypot(min_fwhm...)
        end
    end

    sort!(phot; by = x -> hypot(x.aperture_f.psf_params.fwhm...), rev = true)

    return phot, (; sources, subt, bkg, bkg_rms, aps)
end

"""
    to_subpixel(phot, aps)

Creates a new photometry table that is identical to `phot`, but with the x and y centers replaced with their associated fitted values.

Requires the list of apertures `aps` that were used for the initial photometry to do the necessary conversion from aperture coordinates to image coordinates.
"""
function to_subpixel(phot, aps)
    # Widen column type for x and y coords
    t = Table(phot; xcenter = float.(phot.xcenter), ycenter = float.(phot.ycenter))

    # Update xcenter and ycenter using fitted values
    for (i, (ap_f, ap)) in enumerate(zip(t.aperture_f, aps))
        psf_params = ap_f.psf_params
        t.xcenter[i] += psf_params.x - (size(ap, 1) ÷ 2 + 1)
        t.ycenter[i] += psf_params.y - (size(ap, 2) ÷ 2 + 1)
    end

    return t
end
