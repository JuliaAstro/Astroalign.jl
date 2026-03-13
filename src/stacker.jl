"""
    stack_many(img_stack; verbose = true, ref_slice = size(img_stack,3)÷2+1, min_sigma=2.0, kwargs...)

Stacks many image frames (`img_stack`) stacked along the 3rd dimension into a single result image.

# Parameters

* `img_stack`: input stack to align and sum in the stacking operation. This input stack should have the individual images stacked along dimension 3. It can have multiple colors, stacked along the 4th dimension.
* `verbose`: prints diagnostic output, if `true`. (default: `true`)
* `ref_slice`: an integer indicating the slice to use as a reference image. (default: middle of the stack to minimize field rotation effects).
* `min_sigma`: minimum number of standard deviations a single pixel needs to be away from the mean of that pixel to be excluded. 
               If this number is set to zero, the outlier-exclusion algorithm will not be run.

For other (optional) arguments, see the documentation of `align_frame`.

# Example

```julia
using IndexFunArrays # For gaussian blob generation

# Create Initial star coordinates
sz = (100,100); mid_pos = [sz...] ./ 2
N = 60
star_pos = sz .* rand(2,N)
star_amp = rand(N)
star_shape = (0.3 .+ rand(2, N))

# Create star image
y1 = gaussian(sz; offset = star_pos, weight = star_amp, sigma = star_shape)
rot_mat(alpha) = [cos(alpha) sin(alpha); -sin(alpha) cos(alpha)]

# Rotate the stars by random angles
frames = []
coords = []
N = 10

for alpha in 33*rand(N)
    shift_vec = 10.0 .* (rand(2).-0.5)
    new_pos = rot_mat(alpha * pi/180) * (star_pos .- mid_pos) .+ mid_pos .+ shift_vec
    frame = gaussian(sz; offset = new_pos, weight = star_amp, sigma = star_shape);
    push!(frames, frame); push!(coords, new_pos)
end

frames = cat(frames...; dims=3)
```
"""
function stack_many(img_stack; verbose = true, ref_slice = size(img_stack,3)÷2 + 1, min_sigma = 2.0, kwargs...)
    num_cols = size(img_stack, 4)
    Nimgs = size(img_stack, 3)
    dst_size = (size(img_stack)[1:2]..., num_cols, Nimgs)
    all_params = []
    all_results = similar(img_stack, dst_size)
    all_results .= 0

    # Keep the alignment images in mono
    # Do NOT use a @view in the line below, as this leads to errors!
    ref_img = sum(img_stack[:, :, ref_slice:ref_slice, :]; dims = 4)[:, :, 1, 1]
    # ref_img = @view img_stack[:,:,ref_slice,:]
    ref_info = nothing
    n=1

    for (src_color, res_slice) in zip(eachslice(img_stack; dims = 3), eachslice(all_results; dims = 4))
        # sum over colors. Note that eachslice removes dimension 3
        src_mono = sum(src_color; dims = 3)[:, :, 1, 1]
        myres, params = align_frame(ref_img, src_mono; to_warp = src_color, kwargs...)

        if haskey(params, :ref_info)
           ref_info = params[:ref_info]
        else
            @warn "ignoring slice $(n)"
            n += 1
            continue # ignore this entry
        end

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
        result = remove_outliers(all_results; verbose, stack_dim, min_sigma)
    else
        all_masks = .!isnan.(all_results)
        # Eliminate the NaNs
        all_results[.!all_masks] .= 0
        result = sum(all_results; dims=stack_dim) ./ max.(1, sum(all_masks; dims = stack_dim))
    end

    return result, all_params
end

"""
    stack_many_drizzle(mosaic_stack; drizzle_supersampling = 2.0, min_sigma = 2.0,
                verbose = true, ref_slice = size(mosaic_stack, 3)÷2 + 1, kwargs...)

Stacks many image frames (`mosaic_stack`) stacked along the 3rd dimension into a single result image.

# Parameters

* `mosaic_stack`: input stack to align and sum in the stacking operation. This needs to be a bayer-pattern mosaic. This input stack should have the individual images stacked along dimension 3. 
    Internally first a binned version is calculated and then the transformation parameters are used to transform the original data.
* `drizzle_supersampling`: This is the supersampling factor in comparison to one original (red) color sampling. 
    The default of `2` means that the result size will be equal to the original size, but interpolation free.
    It is important to stack enough images such that no holes remain in the stacked image.
* `min_sigma`: minimum number of standard deviations a single pixel needs to be away from the mean of that pixel to be excluded. 
               If this number is set to zero, the outlier-exclusion algorithm will not be run.
* `verbose`: prints diagnostic output, if `true`. (default: `true`)
* `ref_slice`: an integer indicating the slice to use as a reference image. (default: middle of the stack to minimize field rotation effects).
* `bayer_pattern`: a string of size 4 characters, indicating the order of colors. The default ("RGGB") corresponds to this pattern (starting from the top left corner of `mosaic_stack`):

    ```
    R G R G
    G B R B
    R G R G
    G B R B 
    ```

For other possible (optional) arguments, see the documentation of `align_frame`.
"""
function stack_many_drizzle(mosaic_stack; drizzle_supersampling = 2.0, min_sigma = 2.0,
                verbose = true, ref_slice = size(mosaic_stack,3)÷2 + 1, kwargs...)
    # Sum over colors (for alignment only)
    ref_mono = bin_mono(@view mosaic_stack[:, :, ref_slice])[:, :, 1]

    reduced_size = size(ref_mono)[1:2]

    Nimgs = size(mosaic_stack, 3)
    dst_size = round.(Int, ((reduced_size .* drizzle_supersampling)..., 3, Nimgs))
    all_params = []
    all_results = similar(mosaic_stack, dst_size)
    all_masks = similar(mosaic_stack, Int, dst_size)
    n = 1
    ref_info = nothing

    for (src, res_slice, mymask) in zip(eachslice(mosaic_stack; dims = 3), eachslice(all_results, dims = 4), eachslice(all_masks, dims = 4))
        src_mono = bin_mono(src)[:, :, 1]; # Sum over colors
        myres, params = align_frame(ref_mono, src_mono;
            drizzle_supersampling = drizzle_supersampling,
            to_warp = src,
            ref_info = ref_info,
            kwargs...
        )

        if haskey(params, :ref_info)
           ref_info = params[:ref_info]
        else
            @warn "ignoring slice $(n)"
            n += 1
            continue # Ignore this entry
        end

        push!(all_params, params)
        res_slice .= myres
        mymask .= params[:drizzle_mask]

        if (verbose)
            tfm = params[:tfm]
            a = atan(tfm.linear[1, 2], tfm.linear[1, 1]) * 180/pi
            println("drizzle-stacking: $n, angle: $(round(a; sigdigits = 3)) deg, shift: $(round.(tfm.translation; sigdigits = 4))")
        end

        n += 1
    end

    result = nothing # Since it is returned
    stack_dim = 4

    if (min_sigma > 0)
        result = remove_outliers(all_results, all_masks; verbose, stack_dim, min_sigma)
    else
        divisor = max.(1, sum(all_masks; dims = stack_dim))
        result = sum(all_results, dims = stack_dim) ./ divisor
    end

    # Normalize drizzle result
    # result ./= max.(1, all_params[end][:drizzle_mask])
    return result, all_params
end

function remove_outliers(all_results; kwargs...)
    all_masks = .!isnan.(all_results)
    # Eliminate the NaNs
    all_results[.!all_masks] .= 0
    return remove_outliers(all_results, all_masks; kwargs...)
end

function remove_outliers(all_results, all_masks; verbose = true, stack_dim = 4, min_sigma = 2.0)
        verbose && println("... summing results")
        divisor = max.(1, sum(all_masks; dims = stack_dim))
        result = sum(all_results; dims = stack_dim) ./ divisor

        verbose && println("... removing outliers")
        outliers = (all_masks .!= 0) .&& abs.(all_results .- result) .> min_sigma .* weighted_std(all_results, all_masks; dims = stack_dim)
        verbose && println("... outliers found: $(sum(outliers)), $(round(100*sum(outliers)/length(outliers); sigdigits=3)) %")
        all_results[outliers] .= 0
        all_masks[outliers] .= 0
        divisor = max.(1, sum(all_masks; dims = stack_dim))
        return sum(all_results; dims = stack_dim) ./ divisor
end
