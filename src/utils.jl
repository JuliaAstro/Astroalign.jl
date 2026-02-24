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

function bin_rgb(data; DT=eltype(data), bayer_pattern="RGGB")
    res = similar(data, DT, (size(data,1).÷2, size(data,2).÷2, size(data)[3:end]..., 3))
    idx_sequence = get_bayer_index(bayer_pattern)
    other_idx = ntuple((d)->Colon(), ndims(data)-2)
    res[:,:,other_idx..., idx_sequence[1]] = data[1:2:end-1, 1:2:end-1, other_idx...]
    res[:,:,other_idx..., idx_sequence[2]] = data[2:2:end, 1:2:end-1, other_idx...]
    res[:,:,other_idx..., idx_sequence[3]] = data[1:2:end-1, 2:2:end, other_idx...]
    res[:,:,other_idx..., idx_sequence[4]] = data[2:2:end, 2:2:end, other_idx...]
    normfac = reorient([sum(idx_sequence .== 1), sum(idx_sequence .== 2), sum(idx_sequence .== 3)], Val(ndims(res)))
    res ./= max.(1,normfac)
    return DT.(res)
end


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

"""
    correct_dark_flat(data, dark_img=nothing, flat_img=nothing, channel=nothing; T=Float32)

corrects data using a `dark_img` and `flat_img` by subtracting the dark and deviding by the normalized, dark-subtracted `flat_img`.

* minval: a minum value for the dark-subtracted flat image to avoid division by zero.
"""
function correct_dark_flat(data, dark_img=nothing, flat_img=nothing, channel=nothing, minval=0.001f0; T=Float32)
    if !isnothing(dark_img)
        dark_img = select_region_view(dark_img, new_size=size(data)[1:2]);
        data = T.(data) .- T.(dark_img);
    end
    if !isnothing(flat_img)
        dark_img = isnothing(dark_img) ? 0 : dark_img
        flat_img = max.(select_region_view(flat_img, new_size=size(data)[1:2]) .- dark_img, minval);
        flat_img ./= T.(sum(flat_img)/length(flat_img))        
        data = T.(T.(data) ./ flat_img);
    end
    data
end
