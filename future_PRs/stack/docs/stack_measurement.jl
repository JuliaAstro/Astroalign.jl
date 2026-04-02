#  This script stacks measured data and presents the result

using Astroalign
# using Pluto
using Plots
using Images # to display the stacked result
# using View5D # for interactive viewing
using Statistics: mean, median
using NDTools: select_region
using AstroImages # to be able to load FITS data
using FileIO
using MultifileArrays: load_series # add https://github.com/JuliaIO/MultifileArrays.jl.git

include("preprocess_helpers.jl") # for background subtraction and field flatness correction

if (false) # run this code to download the data and store it locally
    # Here is some example data taken on a dwarf 3 telescope:
    using Downloads
    mydir = mktempdir()
    zip_url = "https://cloud.uni-jena.de/s/9dqYsosX2DxTT5G/download"
    # download the data (takes some time, about 500 Mb!)
    Downloads.download(zip_url, joinpath(mydir, "data.zip"))
    # unzip the downloaded data (in Windows only?). This creates the folder examples/example_data
    run(`tar -xf $(joinpath(mydir, "data.zip"))`)  # works on Windows with tar
end

folder = "example_data\\"
file_dark15 = raw"dark_exp_15.000000_gain_60_bin_1_12C_stack_9.fits"
file_flat = raw"flat_gain_2_bin_1_ir_0.fits"
files = raw"M 35_15s60_Astro_20260215-*_16C.fits"

# data = load_series(load, files);
dark = load(joinpath(folder, file_dark15))
flat = load(joinpath(folder, file_flat))
cd("example_data") # No idea why Windows cannot deal with the file path?
data = load_series(load, files);
cd("..")
data = correct_dark_flat(data, dark, flat); # conversion to Float32 seems essential. In Float64 the fits seem to fail! 

box_size = (15, 15)  
ap_radius = 0.6 * first(box_size);
min_fwhm = 1.5
N_max = 30
use_interp = true;
f = Astroalign.PSF()
# f = com_psf
@time stacked_d, all_params_d = stack_many(data; use_interp=use_interp, use_drizzle=true, dist_limit = 2, f=f, N_max=N_max,
        box_size, ap_radius, min_sigma = 2.5, nsigma = 1, min_fwhm = min_fwhm, drizzle_supersampling = 2.0);

all_stars_used, all_shift_x, all_shift_y, all_med_fwhms_x, all_med_fwhms_y, all_rotation = collect_info(all_params_d);
plot(all_shift_x, title="Shifts", xlabel="frame #", ylabel="shift / pixel", label="X");plot!(all_shift_y, label="Y")
plot(all_med_fwhms_x, title="FWHMs", xlabel="frame #", ylabel="shift / pixel", label="X");plot!(all_med_fwhms_y, label="Y")
plot(all_rotation .* (180/pi), title="rotation", xlabel="frame #", ylabel="angle / deg", label="X")

single_channel = data[2:2:end, 1:2:end, :];
@time stacked_s, all_params_s = stack_many(single_channel; use_drizzle=false, dist_limit = 2, f=f, N_max=N_max,
        box_size, ap_radius, min_sigma = 1.5, nsigma = 1, min_fwhm = min_fwhm);

all_stars_used, all_shift_x, all_shift_y, all_med_fwhms_x, all_med_fwhms_y, all_rotation = collect_info(all_params_s);
plot(all_shift_x, title="Shifts", xlabel="frame #", ylabel="shift / pixel", label="X");plot!(all_shift_y, label="Y")
plot(all_med_fwhms_x, title="FWHMs", xlabel="frame #", ylabel="shift / pixel", label="X");plot!(all_med_fwhms_y, label="Y")

prepare_for_viewer(v) = sqrt.(max.(0, reshape(v .- median(v, dims=(1,2)), (size(v)[1:2]...,1,size(v,3)))))
prepare_for_display(v, m=10.0) = m .*colorview(RGB,permutedims(prepare_for_viewer(v), (4, 2, 1, 3))[:,:,:,1])
# display the result as an RGB image

plot(prepare_for_display(stacked_d, 0.08)) 
# @vt prepare_for_viewer(stacked_d)

# bin first and process the binned color data
all_binned = Astroalign.bin_rgb(data);
box_size = (9, 9)
ap_radius = 0.6 * first(box_size);
stacked_c, all_params_c = stack_many(all_binned; dist_limit = 2, 
        box_size, ap_radius, min_sigma = 2.0, nsigma = 0.5, ref_slice = 1);

# @vt prepare_for_viewer(stacked_c)
plot(prepare_for_display(stacked_c, 0.08)) 

# bin and sum colors first and process the binned monochrome data
all_binned_m = Astroalign.bin_mono(data);
box_size = (9, 9)
ap_radius = 0.6 * first(box_size);
stacked_m, all_param_m = stack_many(all_binned_m; dist_limit = 2, 
        box_size, ap_radius, min_sigma = 2.0, nsigma = 0.5, ref_slice = 1);

plot(prepare_for_display(cat(stacked_m,stacked_m,stacked_m, dims=3), 0.08)) 
# heatmap(sqrt.(clamp.(stacked_m[:,:,1,1], 200, 250)))
@vt prepare_for_viewer(stacked_m)
