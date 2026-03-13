"""
    get_sources(img; box_size = nothing, nsigma = 1, N_max = 10)

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
    y, x = if !hasproperty(p.params, :x) && !hasproperty(p.params, :y)
        Tuple(argmax(psf_data))
    else
        p.params.x, p.params.y
    end

    fwhm = if !hasproperty(p.params, :fwhm)
        _compute_box_size(img_ap)
    else
        p.params.fwhm
    end

    amp = if !hasproperty(p.params, :amp)
        1.0
    else
        p.params.amp
    end

    # TODO: Generalize to other PSFs
    # params = (; x, y, fwhm, amp)
    params = (; x, y, fwhm)

    # Fit
    psf_params, psf_model = fit(p.model, params, psf_data; func_kwargs=p.func_kwargs, p.kwargs...)

    return (; psf_params, psf_model, psf_data)
end

# Internal function used by `align`
# Calls to `Photometry.photometry` with reasonable defaults
function _photometry(img, box_size, ap_radius, min_fwhm, nsigma, f; N_max=10, filter_fwhm, use_fitpos=true)
    # Sources, background subtracted image, background
    sources, subt, _ = get_sources(img; box_size, nsigma, N_max)

    # Define apertures
    aps = CircularAperture.(sources.y, sources.x, ap_radius)

    phot = photometry(aps, subt; f)
    # improve the coordinates estimate with the fit results
    phot = (use_fitpos) ? [(p..., xcenter = p.xcenter + p.aperture_f.psf_params.x - (size(ap)[1]÷2+1),
            ycenter = p.ycenter + p.aperture_f.psf_params.y - (size(ap)[2]÷2+1)) 
            for (p, ap) in zip(phot, aps)] : phot;

    if filter_fwhm
        filter!(phot) do source
            hypot(min_fwhm...) ≤ hypot(source.aperture_f.psf_params.fwhm...)
        end
    end

    sort!(phot; by = x -> hypot(x.aperture_f.psf_params.fwhm...), rev = true)

    return phot
end
