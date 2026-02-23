# helper function. Transforms source coordinates and copies values
function warp_assign!(result, weights, src, x,y, tfm, supersample)
    src_pos = SVector{2}(x,y) # [x, y]
    # Forward transform source coord to dest coord
    dest_coord = tfm(src_pos)
    
    # Map to supersampled grid
    px = round(Int, dest_coord[1]* supersample)
    py = round(Int, dest_coord[2]* supersample)
            
    if checkbounds(Bool, result, px, py)
        @inbounds result[px, py] += src[x, y]
        @inbounds weights[px, py] += one(eltype(weights))
    end
end

"""
    forward_warp!(result, weights, src, tfm, dest_size; supersample=1)

Forward-mode warp: iterate over source, scatter to destination with accumulation.
"""
function forward_warp!(result, weights, src::AbstractMatrix{T}, tfm; supersample=1) where T    
    warp_assign!.(Ref(result), Ref(weights), Ref(src), axes(src, 1), transpose(axes(src, 2)), Ref(tfm), supersample)
    return result, weights
end

"""
    forward_warp(src, tfm, dest_size; supersample=1)

Forward-mode warp: iterate over source, scatter to destination with accumulation.
"""
function forward_warp(src::AbstractMatrix{T}, tfm, dest_size; supersample=1) where T
    out_H, out_W = dest_size .* supersample
    result = zeros(eltype(src), out_H, out_W)
    weights = zeros(Int, out_H, out_W)
    forward_warp!(result, weights, src, tfm; supersample=supersample)
end

function drizzle_warp(bayer_mosaic, tfm; supersample=2.0, bayer_index=(1,2,2,3), result=nothing, drizzle_mask=nothing)
    sindex_x = (1,2,1,2)
    sindex_y = (1,1,2,2)
    for bayer_pix in 1:4
        sx = sindex_x[bayer_pix] # determines the offsets
        sy = sindex_y[bayer_pix]
        src_mat = @view bayer_mosaic[sx:2:end, sy:2:end]
        dst_mat = @view result[:,:,bayer_index[bayer_pix]]
        dst_mask_mat = @view drizzle_mask[:,:,bayer_index[bayer_pix]]
        my_src_shift = Translation([sx-2, sy-2])
        my_zoom = AffineMap([supersample 0;0 supersample],[0, 0])
        tfm_both = compose(my_src_shift, compose(my_zoom, inv(tfm)))
        forward_warp!(dst_mat, dst_mask_mat, src_mat, tfm_both)
    end 
    return result
end

"""
    function align_frame(img_to, img_from;
        [box_size],
        [ap_radius],
        [f],
        [min_fwhm],
        [nsigma],
        [use_fitpos],
        [N_max] 
        [dist_limit],
        [use_fitpos],
        [drizzle_supersampling],
        [mosaic],
        [drizzle_mask],
        [verbose]
    )

Align `img_from` onto `img_to`.
This is achieved by 
1) determining the brightest `N_max` stars in img_from and img_to
2) calculating all triangles connecting these stars
3) finding the best matching triangle between the two sets via the ratio of hypotenuses
4) determining the rigid tranform via the kabsch alsgorithm (allowing for scale)
5) warping `img_from` to the coordinates of `img_to`
6) if `result` is provided, the warped image is added to the result and result is returned, else only the warped image is returned.
7) if `mosaic` is provided, the warping is performed using a `forward_warp` and considering the input is a bayer pattern (default: "RGGB")

# Parameters

* `box_size`: The size of the grid cells (in pixels) used to extract candidate point sources to use for alignment. Defaults to a tenth of the greatest common denominator of the dimensions of `img_to`. See [Photometry.jl > Source Detection Algorithms](@extref Photometry Source-Detection-Algorithms) for more.
* `ap_radius`: The radius of the apertures (in pixel) to place around each point source. Defaults to 60% of `first(box_size)`. See [Photometry.jl > Aperture Photometry](@extref Photometry Aperture-Photometry) for more.
* `f`: The function to compute within each aperture. Defaults to a 2D Gaussian fitted to the aperture center. See the [Source characterization](https://juliaastro.org/Astroalign.jl/notebook.html#Source-characterization) section of the accompanying Pluto.jl notebook for more.
* `min_fwhm`: The minimum FWHM (in pixels) that an extracted point source must have to be considered as a control point. Defaults to a fifth of the width of the first image. See [PSFModels.jl > Fitting data](@extref PSFModels Fitting-data) for more.
* `nsigma`: The number of standard deviations above the estimated background that a source must be to be considered as a control point. Defaults to 1. See [Photometry.jl > Source Detection Algorithms](@extref Photometry Source-Detection-Algorithms) for more.
* `use_fitpos`: if `true` (default), the fit results are used in the position estimate for the triangles and thus the alignment.
* `N_max`: Maximal Number of (brightest) emitters to consider for alignment (default is 10) 
* `dist_limit`: If larger than zero, this will trigger a second stage of alignment considering not only 3 matching emitters (default) but all emitters in the list matching approximately up to this distance in pixels.
* `drizzle_supersampling`: If provided this factor (e.g. Float64) will determine the supersampling to use for the drizzle warp function
* `mosaic = nothing`: If provided, the mosaic, bayer pattern will be used during warping (not for alignment!), default is nothing
* `verbose = true`:  If true (default) some status information is provided during computations
* `ref_info`: if provided, the registration information for the `img_to` does not need to be recalculated (useful for stack operations)
"""
function align_frame(img_to, img_from;
        box_size = _compute_box_size(img_to),
        ap_radius = 0.6 * first(box_size),
        f = PSF(),
        min_fwhm = (2.0, 2.0),
        nsigma = 1,
        N_max= 10, 
        use_fitpos=true,
        drizzle_supersampling = nothing,
        mosaic = nothing,
        dist_limit = 0,
        verbose = true,
        ref_info = nothing)
    # Step 1: Identify control points

    phot_to = isnothing(ref_info) ? _photometry(img_to, box_size, ap_radius, min_fwhm, nsigma, f; N_max= N_max, filter_fwhm=true, use_fitpos=use_fitpos) : ref_info[1]
    phot_from = _photometry(img_from, box_size, ap_radius, min_fwhm, nsigma, f; N_max= N_max, filter_fwhm=true, use_fitpos=use_fitpos)

    # Step 2: Calculate invariants
    C_to, ℳ_to = isnothing(ref_info) ? triangle_invariants(phot_to) : ref_info[2]
    C_from, ℳ_from = triangle_invariants(phot_from)

    # Step 3: Select nearest
    sol_to, sol_from = find_nearest(C_to, ℳ_to, C_from, ℳ_from)

    # Transform
    point_map = map(sol_to, sol_from) do source_to, source_from
        [source_from.xcenter, source_from.ycenter] => [source_to.xcenter, source_to.ycenter]
    end
    # Determine a rigid transform.
    tfm = kabsch(last.(point_map) => first.(point_map); scale=false)

    stars_used = -1
    if true # (dist_limit > 0)
        good_list = []
        dist2_limit = (dist_limit==0) ? 1.0 : abs2(dist_limit)
        # sort!(phot_to; by = x -> x.aperture_f.psf_params.amp, rev = true)
        for pt in phot_to
            best_dist2 = Inf
            best_coord = [0.0, 0.0]
            n=1
            src_coord = tfm([pt.xcenter, pt.ycenter])
            for pf in phot_from
                dist2 = sum(abs2.(src_coord .- [pf.xcenter, pf.ycenter]))
                if (dist2 < best_dist2)
                    best_dist2 = dist2
                    best_coord = [pf.xcenter, pf.ycenter]
                end
            end
            if (best_dist2 < dist2_limit)
                push!(good_list, best_coord => [pt.xcenter, pt.ycenter])
            end
        end
        if (verbose)
            println("$(length(good_list))/$(N_max), $(length(phot_to)) valid stars, corresponding star pairs with distance smaller than $(dist_limit).")
            # @show good_list
        end
        stars_used = length(good_list)

        # rerun kabsch, with more point pairs
        if ((dist_limit > 0) && length(good_list) > 0) 
            tfm = kabsch(last.(good_list) => first.(good_list); scale=false)
        end
    end

    reduced_size = size(img_to)[1:2]

    drizzle_mask = nothing # since it is returned
    if isnothing(drizzle_supersampling) 
        if isnothing(mosaic)
            warped = warp(img_from, tfm, axes(img_to))
        else
            if (size(mosaic,3) == 1)
                warped = warp(mosaic, tfm, axes(img_to))
            else
                tmp = warp(mosaic[:,:,1], tfm, axes(img_to))
                warped = similar(tmp, (size(tmp)..., size(mosaic,3)))
                warped .= 0
                warped[:,:,1] .= tmp
                for n in 2:size(mosaic, 3)
                    warped[:,:,n] = warp(mosaic[:,:,n], tfm, axes(img_to))
                end
            end
        end
    else
        dst_size = round.(Int, ((reduced_size .* drizzle_supersampling)...,3))
        isnothing(mosaic) && error("For drizzle you need to provide a drizzle_supersample! and a mosaic input")
        drizzle_mask = similar(mosaic, Int, dst_size)
        drizzle_mask .= 0
        result = similar(mosaic, dst_size)
        result .= 0
        warped = drizzle_warp(mosaic, tfm, result=result, drizzle_mask=drizzle_mask, supersample=drizzle_supersampling)
    end
    ref_info = isnothing(ref_info) ? (phot_to, (C_to, ℳ_to)) : ref_info

    return (
        warped,
        (;
            point_map,
            tfm,
            C_to,
            ℳ_to,
            C_from,
            ℳ_from,
            drizzle_mask,
            stars_used,
            ref_info, # for convenience
       )
    )
end
