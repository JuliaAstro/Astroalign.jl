function get_bayer_index(pattern="RGGB")
    return ntuple((c)-> (pattern[c]=='R') ? 1 : (pattern[c]=='G') ? 2 : (pattern[c]=='B') ? 3 : 0, 4)
end

function bin_mono(data; DT=eltype(data))
    return sum(bin_rgb(data), dims=ndims(data)+1)
end

"""
    bin_rgb(data; DT=eltype(data), bayer_pattern="RGGB")

Bins bayer-patternd data into red, green and blue channels. The result has half the pixel numbers in each dimension but the 3 RGB color channels are mutually shifted by half pixels in the result grid. The green channel is blurred, since two values from different pixel coordinates are summed.

# Parameters

* `data`: data to split into channels. This can be a single image or a stack.
* `DT`: result datatype (default is the input datatype)
* `bayer_pattern`: a string of size 4 characters, indicating the order of colors. The default ("RGGB") corresponds to this pattern (starting from the top left corner of `mosaic_stack`):

    ```
    R G R G
    G B R B
    R G R G
    G B R B
    ```
"""
function bin_rgb(data; DT = eltype(data), bayer_pattern = "RGGB")
    res = similar(data, DT, (size(data, 1) .÷ 2, size(data,2) .÷ 2, size(data)[3:end]..., 3))
    res .= 0;
    idx_sequence = get_bayer_index(bayer_pattern)
    other_idx = ntuple((d) -> Colon(), ndims(data) - 2)
    res[: , :, other_idx..., idx_sequence[1]] .+= data[1:2:end-1, 1:2:end-1, other_idx...]
    res[: , :, other_idx..., idx_sequence[2]] .+= data[2:2:end  , 1:2:end-1, other_idx...]
    res[: , :, other_idx..., idx_sequence[3]] .+= data[1:2:end-1, 2:2:end  , other_idx...]
    res[: , :, other_idx..., idx_sequence[4]] .+= data[2:2:end  , 2:2:end  , other_idx...]
    reorient_tuple = ntuple((d) -> (d == ndims(res)) ? 3 : 1, ndims(res))
    normfac = reshape([sum(idx_sequence .== 1), sum(idx_sequence .== 2), sum(idx_sequence .== 3)], reorient_tuple)
    res ./= max.(1, normfac)
    return DT.(res)
end

"""
    weighted_std(data, weights; dims=4)

Calculates the standard deviation allowing for (binary) weights indicating which pixels are considered.
"""
function weighted_std(data, weights; dims = 4)
    # mean projection of counting pixels
    mp = sum(data; dims) ./ max.(1, sum(weights; dims))
    myvar = sum(abs2.((data .- mp .* weights).* weights); dims) ./ max.(1, sum(weights; dims))
    return sqrt.(myvar)
end

function mystd(data; dims = 4)
    mp = sum(data; dims) ./ size(data, dims)
    myvar = sum(abs2.(data .- mp); dims) ./ size(data, dims)
    return sqrt.(myvar)
end

# conciders the N brightests stars only
# returns a tuple of fwhm_x and fwhm_y
function median_fwhm(param, idx=1) 
    # the mod operation ensures that this also works for only a single fwhm scalar to fit
    median([p.aperture_f[1].fwhm[mod(idx-1, length(p.aperture_f[1].fwhm))+1] for p in param])
end

"""
    collect_info(all_params)

    returns the following information from the parameters collected during stacking:
    `stars_used`, `shift_x`, `shift_y`, `med_fwhms_x`, `med_fwhms_y`, `rotation`

    This information can be used for plotting
"""
function collect_info(all_params)
    all_stars_used = [p.stars_used for p in all_params]
    all_shift_x = [p.tfm.translation[1] for p in all_params]
    all_shift_y = [p.tfm.translation[2] for p in all_params]
    all_rotation = [atan(p.tfm.linear[1,2], p.tfm.linear[1,1]) for p in all_params]
    all_med_fwhms_x = [p.med_fwhm_x for p in all_params]
    all_med_fwhms_y = [p.med_fwhm_y for p in all_params]
    return all_stars_used, all_shift_x, all_shift_y, all_med_fwhms_x, all_med_fwhms_y, all_rotation
end
