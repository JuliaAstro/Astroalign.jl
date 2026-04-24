Base.@kwdef struct PSF
    model = gaussian
    params = (; fwhm = 1.5)
    func_kwargs = (;)
    kwargs = (; x_abstol = 2e-6)
end

"""
    (p::PSF)(img)

Callable for [PSFModels.fit](@extref) used by [Astroalign._photometry](@ref).
"""
(p::PSF)(img) = fit_psf(img, p)

"""
    fit_psf(img_ap, p)

Fits a point spread function model to a region of interest (`img_ap`) using the Optim.jl toolbox.
"""
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

    fwhm = p.params.fwhm

    params = (; x, y, fwhm)

    # Fit
    psf_params, psf_model = fit(p.model, params, psf_data; func_kwargs = p.func_kwargs, p.kwargs...)

    return (; psf_params, psf_model, psf_data)
end

"""
    com_psf(img_ap)

Determine peak parameters via a fast, non-iterative center-of-mass approach.
"""
function com_psf(img_ap; rel_thresh=0.1f0)
    psf_data = collect(Float32, img_ap)
    psf_data ./= maximum(psf_data)

    x_coords = axes(psf_data, 1)
    y_coords = axes(psf_data, 2)'
    # use the average of the edge of the ROI as background estimate
    bg = (sum(psf_data[1,:])+sum(psf_data[end,:])+sum(psf_data[:,1])+sum(psf_data[:,end])) / 
        (2*size(psf_data,1)+2*size(psf_data,2))

    # clamp at zero before com calculation
    psf_data = max.(0, psf_data .- bg .- rel_thresh)
    sum_psf = sum(psf_data)
    x = sum(x_coords .* psf_data) / sum_psf
    y = sum(y_coords .* psf_data) / sum_psf

    var_x = sum(abs2.(x_coords.-x) .* psf_data) / sum_psf
    var_y = sum(abs2.(y_coords.-y) .* psf_data) / sum_psf
    fac = 2*sqrt(2*log(2))
    fwhm = (sqrt(var_x)*fac, sqrt(var_y)*fac)

    psf_params = (; x, y, fwhm)
    psf_model = "com"
    return (; psf_params, psf_model, psf_data)
end
