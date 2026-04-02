using NDTools: select_region_view

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
