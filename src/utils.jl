function _compute_box_size(img)
    w = gcd(size(img)...) ÷ 10
    box_width = iseven(w) ? w + 1 : w
    return (box_width, box_width)
end

function bin_mono(data; DT=eltype(data))
    DT.(data[1:2:end,1:2:end]) .+ DT.(data[2:2:end,1:2:end]) .+ DT.(data[1:2:end,2:2:end]) .+ DT.(data[2:2:end,2:2:end])
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