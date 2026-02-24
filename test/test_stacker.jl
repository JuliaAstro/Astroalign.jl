using Random # to seed the random generator for reproducibility
Random.seed!(42)
# using View5D  # recommended for visualization

@testset "test_stacker" begin

using IndexFunArrays # for gaussian blob generation
# create inital star coordinates
sz = (100,100); mid_pos = [sz...]./2;
N = 60; star_pos = sz .* rand(2,N); star_amp=rand(N);
star_shape = (0.3 .+rand(2,N))

# create star image
y1 = gaussian(sz, offset = star_pos, weight=star_amp, sigma=star_shape);
rot_mat(alpha) = [cos(alpha) sin(alpha); -sin(alpha) cos(alpha)]

# rotate the stars by random angles
frames = []; coords = []; N = 10
for alpha in 33*rand(N)
    shift_vec = 10.0 .* (rand(2).-0.5)
    new_pos = rot_mat(alpha*pi/180) * (star_pos.- mid_pos) .+ mid_pos .+ shift_vec
    frame = gaussian(sz, offset = new_pos, weight=star_amp, sigma=star_shape);
    push!(frames, frame); push!(coords, new_pos)
end
frames = cat(frames..., dims=3)
# @vt frames # visualize all frames as a 3d stack

using Astroalign
# align the first fram in the stack to the reference image
y2_aligned, params = align_frame(y1, frames[:,:,1]);

# @vt y1 y2_aligned # display alignment (toggle between frames using the keys `,` and `.`)

# check alignment by summing at the origninal source positions and the destinatoin positions

y2_aligned[isnan.(y2_aligned)] .= 0
y1_coords = clamp.(round.(Int, star_pos), 1, sz[1]-1)
y2_coords = clamp.(round.(Int, coords[1]), 1, sz[1]-1)
sum_at_true = sum(sum.([y2_aligned[c...] for c in eachslice(y1_coords, dims=2)]))
sum_at_source =sum(sum.([y2_aligned[c...] for c in eachslice(y2_coords, dims=2)]))

@test sum_at_true / sum_at_source > 8

# stack all frames in the stack to the coordinates of the first frame
stacked, all_params = stack_many(frames; ref_slice=1);

@test size(stacked) == (100,100,1,1)
# @vt res
sum_at_true = sum(sum.([stacked[c...] for c in eachslice(y2_coords, dims=2)]))
sum_at_source =sum(sum.([stacked[c...] for c in eachslice(y1_coords, dims=2)]))
@test sum_at_true / sum_at_source > 8

end