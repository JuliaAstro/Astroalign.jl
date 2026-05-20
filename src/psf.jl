"""
    com_psf(T::Type{<:AbstractFloat}, img_ap, rel_thresh)

Determine peak parameters via a fast, non-iterative center-of-mass approach.
"""
function com_psf(T::Type{<:AbstractFloat}, img_ap::AbstractMatrix, rel_thresh)
    ax, ay = axes(img_ap, 1), axes(img_ap, 2)
    peak = maximum(img_ap)

    # use the average of the edge of the cutout as background estimate
    bg_sum = zero(T)
    @inbounds for j in ay
        bg_sum += img_ap[first(ax),j] + img_ap[last(ax),j]
    end
    @inbounds for i in ax[begin+1:end-1]
        bg_sum += img_ap[i,first(ay)] + img_ap[i,last(ay)]
    end
    threshold = bg_sum / (2*length(ax) + 2*length(ay)) + rel_thresh * peak

    # sum_w: total weight, sum_wx/wy: weighted x/y sums (for centroid),
    # sum_wx2/wy2: weighted x²/y² sums for variance via Konig's theorem
    # var(X) = E(X²) − E(X)²
    sum_w, sum_wx, sum_wy, sum_wx2, sum_wy2 = zero(T), zero(T), zero(T), zero(T), zero(T)
    @inbounds for j in ay
        col_w, col_wx, col_wx2 = zero(T), zero(T), zero(T)
        for i in ax
            v = ifelse(img_ap[i,j] - threshold < 0, zero(T), img_ap[i,j] - threshold)
            col_w += v; col_wx += i*v; col_wx2 += i*i*v
        end
        sum_w += col_w; sum_wx += col_wx; sum_wx2 += col_wx2
        sum_wy += j*col_w; sum_wy2 += j*j*col_w
    end

    x, y = sum_wx / sum_w, sum_wy / sum_w
    var_x = sum_wx2 / sum_w - x*x
    var_y = sum_wy2 / sum_w - y*y
    fac = T(2 * sqrt(2 * log(2)))
    fwhm = (sqrt(var_x) * fac, sqrt(var_y) * fac)
    return (; psf_params=(; x, y, fwhm), psf_model="com", psf_data=img_ap)
end
# Fallback for non-array inputs; Photometry.photometry uses Transducers.jl so we have to collect first
com_psf(T::Type{<:AbstractFloat}, img_ap, rel_thresh) = com_psf(T, collect(T, img_ap), rel_thresh)
"""
   com_psf(img_ap; rel_thresh::T=0.1f0) where T <: AbstractFloat
Forwards to `com_psf(T, img_ap, rel_thresh)` where `T` is inferred from the type of `rel_thresh`.
"""
com_psf(img_ap; rel_thresh::T=0.1f0) where T <: AbstractFloat = com_psf(T, img_ap, rel_thresh)
