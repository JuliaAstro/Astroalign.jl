module Astroalign

using Combinatorics: combinations
using CoordinateTransformations: kabsch
using Distances: euclidean
using ImageTransformations: warp
using NearestNeighbors: nn, KDTree
using Photometry: estimate_background,
                  extract_sources,
                  photometry,
                  sigma_clip,
                  CircularAperture,
                  PeakMesh
using PSFModels: gaussian, fit
using LinearAlgebra: norm

export align, find_nearest, get_photometry, get_sources, triangle_invariants

function get_sources(img; box_size = nothing, nsigma = 1)
    if isnothing(box_size)
        box_size = _compute_box_size(img)
    end

    # Background subtract `img`
    clipped = sigma_clip(img, 1, fill = NaN)
    bkg, bkg_rms = estimate_background(clipped, box_size)
    subt = img .- bkg[axes(img)...]

    return (
        # Sort detected sources from brightest to darkest
        extract_sources(PeakMesh(; box_size, nsigma), subt, bkg, true),
        # And also return the inputs, handy for debugging and data viz
        subt,
        bkg,
    )
end

function fit_psf(img_ap)
    # Normalize
    psf_data = collect(Float32, img_ap)
    psf_data ./= maximum(psf_data)

    # Set params
    xcen, ycen = Tuple(argmax(psf_data))
    fwhm = _compute_box_size(img_ap)
    params = (x=ycen, y=xcen, fwhm)

    # Fit
    psf_params, psf_model = fit(gaussian, params, psf_data; x_abstol=2e-6)

    return (; psf_params, psf_model, psf_data)
end

# Internal function used by `align`
# Calls to `Photometry.photometry` with reasonable defaults
function _photometry(img, box_size, ap_radius, min_fwhm, nsigma; filter_fwhm=false)
    # Sources, background subtracted image, background
    sources, subt, _ = get_sources(img; box_size, nsigma)

    # Define apertures
    aps = CircularAperture.(sources.y, sources.x, ap_radius)

    phot = photometry(aps, subt; f = fit_psf)

    if filter_fwhm
        filter!(phot) do source
            norm(min_fwhm) ≤ norm(source.aperture_f.psf_params.fwhm)
        end
    end

    sort!(phot; by = x -> hypot(x.aperture_f.psf_params.fwhm...), rev = true)

    return phot
end

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

function find_nearest(C_to, ℳ_to, C_from, ℳ_from)
    idxs, dists = nn(KDTree(ℳ_to), ℳ_from)
    idx_from = argmin(dists)
    idx_to = idxs[idx_from]
    sol_to = collect(C_to)[idx_to]
    sol_from = collect(C_from)[idx_from]
    return sol_to, sol_from
end

function align(img_to, img_from; box_size = nothing, ap_radius = nothing, min_fwhm = nothing, nsigma = 1)
    if isnothing(box_size)
        box_size = _compute_box_size(img_to)
    end

    if isnothing(ap_radius)
        ap_radius = 0.6 * first(box_size)
    end

    if isnothing(min_fwhm)
        min_fwhm = box_size .÷ 5
    end

    # Step 1: Identify control points
    phot_to = _photometry(img_to, box_size, ap_radius, min_fwhm, nsigma; filter_fwhm=true)
    phot_from = _photometry(img_from, box_size, ap_radius, min_fwhm, nsigma; filter_fwhm=true)

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
