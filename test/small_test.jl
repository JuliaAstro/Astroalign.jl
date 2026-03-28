# This is a debugging test, using the real star data which failed for some frames.
box_size = (15, 15)
ap_radius = 0.6 * first(box_size);
f = com_psf
@time stacked_d, all_params_d = stack_many_drizzle(data; dist_limit = 2, f=f,
               box_size=box_size, ap_radius=ap_radius, min_sigma = 1.0, min_fwhm=2.0, nsigma = 1, ref_slice = 1, drizzle_supersampling = 2.0);

bad_pos = 3               
@time stacked_d, all_params_d = stack_many_drizzle(data[:,:,[1, bad_pos]]; dist_limit = 2,
box_size=box_size, ap_radius=ap_radius, min_sigma = 1.0, min_fwhm=2.0, nsigma = 1, ref_slice = 1, drizzle_supersampling = 2.0);

img_to = data[1:2:end,1:2:end,1]
img_from = data[1:2:end,1:2:end, bad_pos]
# y2_aligned, params =  align_frame(img_to, img_from; min_fwhm=2.0, nsigma = 1, drizzle_supersampling = 2.0)  
min_fwhm=2.0; nsigma = 1;
f = com_psf
f = PSF()
y2_aligned, params =  align_frame(img_to, img_from; box_size=box_size, ap_radius=ap_radius, min_fwhm=min_fwhm, nsigma = nsigma, f=f);  


using Astroalign: _photometry, PSF
ref_info = nothing; N_max = 20; N_best = 7; use_fitpos = true; 
f=PSF()
phot_to = isnothing(ref_info) ? _photometry(img_to, box_size, ap_radius, min_fwhm, nsigma, f; N_max, filter_fwhm = true, use_fitpos) : ref_info[1]
phot_from = _photometry(img_from, box_size, ap_radius, min_fwhm, nsigma, f; N_max, filter_fwhm = true, use_fitpos)


function show_star_pos(img, phot)
    v = @vv img
    import_marker_lists([[p.xcenter,  p.ycenter] for p in phot])
end