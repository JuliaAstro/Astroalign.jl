# here you find the code of the stacking algorithms

function stack_many(img_stack; verbose = true, ref_slice = size(img_stack,3)÷2+1, min_sigma=2.0, kwargs...)
    Nimgs = size(img_stack,3)
    dst_size = (size(img_stack)[1:2]...,3, Nimgs)
    all_params = []
    all_results = similar(img_stack, dst_size)
    all_results .= 0
    ref_img = sum((@view img_stack[:,:,ref_slice,:]), dims=4)[:,:,1,1]
    # ref_img = @view img_stack[:,:,ref_slice,:]
    ref_info = nothing;
    n=1
    for (src_color, res_slice) in zip(eachslice(img_stack, dims=3), eachslice(all_results, dims=4))
        src_mono = sum(src_color, dims=4)[:,:,1,1]; # sum over colors
        myres, params = align_frame(ref_img, src_mono; mosaic=src_color, kwargs...)
        ref_info = params[:ref_info]
        push!(all_params, params)
        res_slice .= myres
        if (verbose)
            tfm = params[:tfm]
            a = atan(tfm.linear[1,2], tfm.linear[1,1]) * 180/pi
            println("stacking frame $n, angle: $a deg, shift: $(tfm.translation)")
        end
        n += 1
    end

    result = nothing
    stack_dim = 4
    if (min_sigma > 0)
        @show size(all_results)
        @show min_sigma
        result = remove_outliers(all_results; verbose=verbose, stack_dim=stack_dim, min_sigma=min_sigma)
    else
        all_masks = .!isnan.(all_results)
        # eliminate the NaNs
        all_results[.!all_masks] .= 0
        result = sum(all_results, dims=stack_dim) ./ max.(1, sum(all_masks, dims=stack_dim))
    end
    return result, all_params
end

function stack_many_drizzle(mosaic_stack; drizzle_supersampling = 2.0, min_sigma = 2.0,  
                verbose=true, ref_slice = size(mosaic_stack,3)÷2+1, kwargs...)    
    # sum over colors (for alignment only)
    ref_mono = bin_mono(@view mosaic_stack[:,:,ref_slice]);

    reduced_size = size(ref_mono)[1:2]

    Nimgs = size(mosaic_stack,3)
    dst_size = round.(Int, ((reduced_size .* drizzle_supersampling)...,3, Nimgs))
    all_params = []
    all_results = similar(mosaic_stack, dst_size)
    all_masks = similar(mosaic_stack, Int, dst_size)
    n=1
    ref_info = nothing;

    for (src, res_slice, mymask) in zip(eachslice(mosaic_stack, dims=3), eachslice(all_results, dims=4), eachslice(all_masks, dims=4))
        src_mono = bin_mono(src); # sum over colors
        myres, params = align_frame(ref_mono, src_mono;
                drizzle_supersampling = drizzle_supersampling,
                mosaic = src, ref_info = ref_info, kwargs...)
        ref_info = params[:ref_info]
        push!(all_params, params)
        res_slice .= myres
        mymask .= params[:drizzle_mask]
        if (verbose)
            tfm = params[:tfm]
            a = atan(tfm.linear[1,2], tfm.linear[1,1]) * 180/pi
            println("drizzle-stacking: $n, angle: $(round(a, sigdigits=3)) deg, shift: $(round.(tfm.translation, sigdigits=4))")
        end
        n += 1
    end

    result = nothing # since it is returned
    stack_dim = 4
    if (min_sigma > 0)
        result = remove_outliers(all_results, all_masks; verbose=verbose, stack_dim=stack_dim, min_sigma=min_sigma)
    else
        divisor = max.(1,sum(all_masks, dims=stack_dim))
        result = sum(all_results,dims=stack_dim) ./ divisor
    end
    # normalize drizzle result
    # result ./= max.(1,all_params[end][:drizzle_mask])
    return result, all_params
end

function remove_outliers(all_results; kwargs...)
    all_masks = .!isnan.(all_results)
    # eliminate the NaNs
    all_results[.!all_masks] .= 0
    return remove_outliers(all_results, all_masks; kwargs...)
end

function remove_outliers(all_results, all_masks; verbose=true, stack_dim=4, min_sigma=2.0)
        (verbose) && println("... summing results")
        divisor = max.(1, sum(all_masks, dims=stack_dim))
        result = sum(all_results,dims=stack_dim) ./ divisor

        (verbose) && println("... removing outliers")
        outliers = (all_masks .!= 0) .&& abs.(all_results .- result) .> min_sigma .* weighted_std(all_results, all_masks, dims=stack_dim)
        (verbose) && println("... outliers found: $(sum(outliers)), $(round(100*sum(outliers)/length(outliers), sigdigits=3)) %")
        all_results[outliers] .= 0
        all_masks[outliers] .= 0
        divisor = max.(1,sum(all_masks, dims=stack_dim))
        return sum(all_results, dims=stack_dim)./ divisor
end