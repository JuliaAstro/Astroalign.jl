"""
    com_psf(T::Type{<:AbstractFloat}, img_ap::AbstractMatrix, rel_thresh)

Determine peak parameters via a fast, non-iterative center-of-mass approach.
Return quantities are typed to `promote_type(T, typeof(maximum(img_ap)))`.

The threshold for the center-of-mass calculation is computed as the average
value of the border pixels (a simple background estimate) plus `rel_thresh`
times the peak pixel value.  Pixels below this threshold are clamped to zero.
Larger values of `rel_thresh` isolate the core of the PSF; smaller values
include more of the wings.

Peak parameters are returned as a NamedTuple with fields
 - `psf_params`: NamedTuple containing entries `x`, `y`, and `fwhm`
   (a tuple of x and y FWHM).
 - `psf_model`: `"com"` to indicate center-of-mass
 - `psf_data`: the input image cutout (`img_ap`) for reference
"""
function com_psf(T::Type{<:AbstractFloat}, img_ap::AbstractMatrix, rel_thresh)
    ax, ay = axes(img_ap, 1), axes(img_ap, 2)
    peak = maximum(img_ap)
    Tc = float(promote_type(T, typeof(peak)))

    # use the average of the edge of the cutout as background estimate
    bg_sum = zero(Tc)
    @inbounds for j in ay
        bg_sum += img_ap[first(ax),j] + img_ap[last(ax),j]
    end
    @inbounds for i in ax[begin+1:end-1]
        bg_sum += img_ap[i,first(ay)] + img_ap[i,last(ay)]
    end
    threshold = bg_sum / (2*length(ax) + 2*length(ay) - 4) + Tc(rel_thresh) * peak

    # sum_w: total weight, sum_wx/wy: weighted x/y sums (for centroid),
    # sum_wx2/wy2: weighted x²/y² sums for variance via Konig's theorem
    # var(X) = E(X²) − E(X)²
    sum_w, sum_wx, sum_wy, sum_wx2, sum_wy2 = zero(Tc), zero(Tc), zero(Tc), zero(Tc), zero(Tc)
    @inbounds for j in ay
        col_w, col_wx, col_wx2 = zero(Tc), zero(Tc), zero(Tc)
        for i in ax
            v = img_ap[i,j] - threshold
            v = ifelse(v < 0, zero(Tc), v)
            col_w += v; col_wx += i*v; col_wx2 += i*i*v
        end
        sum_w += col_w; sum_wx += col_wx; sum_wx2 += col_wx2
        sum_wy += j*col_w; sum_wy2 += j*j*col_w
    end

    x, y = sum_wx / sum_w, sum_wy / sum_w
    var_x = max(zero(Tc), sum_wx2 / sum_w - x*x)
    var_y = max(zero(Tc), sum_wy2 / sum_w - y*y)
    fac = Tc(2 * sqrt(2 * log(2)))
    fwhm = (sqrt(var_x) * fac, sqrt(var_y) * fac)
    return (; psf_params=(; x, y, fwhm), psf_model="com", psf_data=img_ap)
end
"""
    com_psf(img_ap; rel_thresh::T=0.1f0) where T <: AbstractFloat
Forwards to `com_psf(T, img_ap, rel_thresh)` where `T` is inferred from the type of
`rel_thresh`; accumulation promotes `T` with the image peak type.
"""
com_psf(img_ap; rel_thresh::T=0.1f0) where T <: AbstractFloat = com_psf(T, img_ap, rel_thresh)
