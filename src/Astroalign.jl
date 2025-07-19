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

export align, get_photometry, get_sources

function get_sources(img; box_size = (3, 3), nsigma = 1)
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

function fit_psf(img_ap; fwhm=2, kernel=gaussian)
    # Normalize
    psf_data = collect(Float32, img_ap)
    psf_data ./= maximum(psf_data)

    # Fit
    xcen, ycen = Tuple(argmax(psf_data))
    params = (x=ycen, y=xcen, fwhm)
    psf_P, psf_model = fit(kernel, params, psf_data)
    return (; psf_P, psf_model, psf_data)
end

function get_photometry(aps, subt; f = fit_psf, sort_fwhm = true)
    phot = photometry(aps, subt; f)
    sort_fwhm && sort!(phot; by = x -> x.aperture_f.psf_P.fwhm, rev = true)
end

function get_photometry(img; box_size = (3, 3), ap_radius = 10, sort_fwhm = true, min_fwhm = 1.5)
    # Sources, background subtracted image, background
    sources, subt, _ = get_sources(img; box_size)

    # Define apertures
    aps = CircularAperture.(sources.y, sources.x, ap_radius)

    return filter!(get_photometry(aps, subt; sort_fwhm)) do phot
        min_fwhm ≤ phot.aperture_f.psf_P.fwhm
    end
end

function triangle_invariants(C)
    map(C) do (pa, pb, pc)
        a, b, c = (
            (pa.ycenter, pa.xcenter),
            (pb.ycenter, pb.xcenter),
            (pc.ycenter, pc.xcenter),
        )
        Ls = sort!([euclidean(a, b), euclidean(b, c), euclidean(a, c)])
        ℳ = (Ls[3] / Ls[2], Ls[2] / Ls[1])
    end |> stack
end

function align(img_to, img_from; box_size = (3, 3))
    # Step 1: Identify control points
    phot_to = get_photometry(img_to; box_size)
    phot_from = get_photometry(img_from; box_size)

    # Step 2: Calculate invariants
    C_to = combinations(phot_to, 3)
    C_from = combinations(phot_from, 3)
    ℳ_to = triangle_invariants(C_to)
    ℳ_from = triangle_invariants(C_from)

    # Step 3: Select nearest
    idxs, dists = nn(KDTree(ℳ_to), ℳ_from)
    idx_from = argmin(dists)
    idx_to = idxs[idx_from]
    sol_to = collect(C_to)[idx_to]
    sol_from = collect(C_from)[idx_from]

    # Transform
    point_map = map(sol_to, sol_from) do source_to, source_from
        [source_from.xcenter, source_from.ycenter] => [source_to.xcenter, source_to.ycenter]
    end

    tfm = kabsch(last.(point_map) => first.(point_map); scale=false)

    return (
        warp(img_from, tfm, axes(img_to)),
        point_map,
        tfm,
        ℳ_to,
        ℳ_from,
    )
end

end # module
