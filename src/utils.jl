function _compute_box_size(img)
    w = gcd(size(img)...) ÷ 10
    box_width = iseven(w) ? w + 1 : w
    return (box_width, box_width)
end

function get_bayer_index(pattern="RGGB")
    return ntuple((c)-> (pattern[c]=='R') ? 1 : (pattern[c]=='G') ? 2 : (pattern[c]=='B') ? 3 : 0, 4)
end

function bin_mono(data; DT=eltype(data))
    return sum(bin_rgb(data), dims=ndims(data)+1)
end

"""
    bin_rgb(data; DT=eltype(data), bayer_pattern="RGGB")

bins bayer-patternd data into red, green and blue channels.
The result has half the pixel numbers in each dimension but the 3 RGB color channels are mutually shifted by
half pixels in the result grid. The green channel is blurred, since two values from different pixel coordinates are summed.

# Parameters
* `data`: data to split into channels. This can be a single image or a stack.
* `DT`: result datatype (default is the input datatype)
* `bayer_pattern`: a string of size 4 characters, indicating the order of colors. The default ("RGGB") corresponds to
This pattern (starting from the top left corner of `mosaic_stack`)
R G R G
G B R B
R G R G
G B R B 
"""
function bin_rgb(data; DT=eltype(data), bayer_pattern="RGGB")
    res = similar(data, DT, (size(data,1).÷2, size(data,2).÷2, size(data)[3:end]..., 3))
    res .= 0;
    idx_sequence = get_bayer_index(bayer_pattern)
    other_idx = ntuple((d)->Colon(), ndims(data)-2)
    res[:,:,other_idx..., idx_sequence[1]] .+= data[1:2:end-1, 1:2:end-1, other_idx...]
    res[:,:,other_idx..., idx_sequence[2]] .+= data[2:2:end, 1:2:end-1, other_idx...]
    res[:,:,other_idx..., idx_sequence[3]] .+= data[1:2:end-1, 2:2:end, other_idx...]
    res[:,:,other_idx..., idx_sequence[4]] .+= data[2:2:end, 2:2:end, other_idx...]
    reorient_tuple = ntuple((d)-> (d == ndims(res)) ? 3 : 1, ndims(res))
    normfac = reshape([sum(idx_sequence .== 1), sum(idx_sequence .== 2), sum(idx_sequence .== 3)], reorient_tuple)
    res ./= max.(1,normfac)
    return DT.(res)
end

"""
    weighted_std(data, weights; dims=4)

calculates the standard deviation allowing for (binary) weights indicating which pixels are considered.
"""
function weighted_std(data, weights; dims=4)
    mp = sum(data .* weights, dims=dims) ./ max.(1, sum(weights, dims=dims))
    myvar = sum(abs2.((data .- mp) .* weights), dims=dims) ./ max.(1, sum(weights, dims=dims))
    return sqrt.(myvar)
end

function mystd(data; dims=4)
    mp = sum(data , dims=dims)./size(data, dims)
    myvar = sum(abs2.(data .- mp), dims=dims)./size(data, dims)
    return sqrt.(myvar)
end

