"""
    triangle_invariants(phot)

Returns all combinations (``C``) of three candidate point sources from the table of sources `phot` returned by [`Photometry.Aperture.photometry`](@extref), and the computed invariant ``\\mathscr M`` for each according to Eq. 3 from [_Beroiz, M., Cabral, J. B., & Sanchez, B. (2020)_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract).
"""
function triangle_invariants(phot)
    C = combinations(phot, 3)
    ℳ = map(C) do (pa, pb, pc)
        a, b, c = (
            (pa.ycenter, pa.xcenter),
            (pb.ycenter, pb.xcenter),
            (pc.ycenter, pc.xcenter),
        )
        Ls = sort!([euclidean(a, b), euclidean(b, c), euclidean(a, c)])
        (Ls[3] / Ls[2], Ls[2] / Ls[1])
    end |> stack
    return C, ℳ
end

"""
    find_nearest(C_to, ℳ_to, C_from, ℳ_from)

Return the closes pair of three points between the `from` and `to` frames in the invariant ``\\mathscr M`` space as computed by [`Astroalign.triangle_invariants`](@ref).
"""
function find_nearest(C_to, ℳ_to, C_from, ℳ_from)
    # Find the nearest neighbors regarding the triangles
    idxs, dists = nn(KDTree(ℳ_to), ℳ_from)
    # Determine the best match:
    idx_from = argmin(dists)
    idx_to = idxs[idx_from]
    sol_to = collect(C_to)[idx_to]
    sol_from = collect(C_from)[idx_from]
    return sol_to, sol_from
end

# Re-order the three vertices of a triangle so that:
# 1. The apex (vertex opposite the longest edge) is last.
# 2. The two base vertices are ordered counter-clockwise (positive cross product).
# This canonical form is preserved under rotation and translation, so corresponding
# triangles in two images receive the same vertex permutation and produce
# geometrically consistent point correspondences.
function _canonical_vertex_order(pa, pb, pc)
    xa, ya = pa.xcenter, pa.ycenter
    xb, yb = pb.xcenter, pb.ycenter
    xc, yc = pc.xcenter, pc.ycenter

    d2_ab = (xb - xa)^2 + (yb - ya)^2
    d2_bc = (xc - xb)^2 + (yc - yb)^2
    d2_ac = (xc - xa)^2 + (yc - ya)^2

    # Identify the two base vertices (endpoints of longest edge) and the apex
    if d2_ab >= d2_bc && d2_ab >= d2_ac
        v1, v2, apex = pa, pb, pc
    elseif d2_bc >= d2_ab && d2_bc >= d2_ac
        v1, v2, apex = pb, pc, pa
    else
        v1, v2, apex = pa, pc, pb
    end

    # Enforce CCW winding so that a rotation does not change the order
    cross = (v2.xcenter - v1.xcenter) * (apex.ycenter - v1.ycenter) -
            (v2.ycenter - v1.ycenter) * (apex.xcenter - v1.xcenter)
    cross < 0 && ((v1, v2) = (v2, v1))

    return (v1, v2, apex)
end

"""
    _build_correspondences(C_to, ℳ_to, C_from, ℳ_from; k = 5)

Build a `4 × N` matrix of candidate point correspondences between the `from`
and `to` frames, where each column is `[x_from; y_from; x_to; y_to]`.

For each triangle in `from`, the `k` nearest triangles in `to` (measured in the
invariant ``\\mathscr M`` space) are retrieved. The three vertex pairs from each
matched triangle pair – ordered canonically via [`_canonical_vertex_order`](@ref) –
contribute three columns to the output.  The resulting pool of candidate
correspondences is suitable as the data matrix `x` for
[`ConsensusFitting.ransac`](@extref).
"""
function _build_correspondences(C_to, ℳ_to, C_from, ℳ_from; k = 5)
    C_to_list   = collect(C_to)
    C_from_list = collect(C_from)

    isempty(C_to_list) || isempty(C_from_list) && return zeros(4, 0)

    k_actual = min(k, size(ℳ_to, 2))
    idxs, _ = knn(KDTree(ℳ_to), ℳ_from, k_actual)

    cols = Vector{Float64}[]

    for i in eachindex(C_from_list)
        canon_from = _canonical_vertex_order(C_from_list[i]...)
        for j in idxs[i]
            canon_to = _canonical_vertex_order(C_to_list[j]...)
            for l in 1:3
                push!(cols, [
                    canon_from[l].xcenter,
                    canon_from[l].ycenter,
                    canon_to[l].xcenter,
                    canon_to[l].ycenter,
                ])
            end
        end
    end

    isempty(cols) && return zeros(4, 0)
    return hcat(cols...)
end

# ──────────────────────────────────────────────────────────────────
# Analytic minimal fitting functions for ConsensusFitting.ransac
# ──────────────────────────────────────────────────────────────────

# Minimum squared distance below which two points are considered coincident
# for the purposes of degeneracy testing in the minimal fitting functions.
const _DEGENERATE_SQ = 1e-8

"""
    _fit_minimal_rigid(x)

Analytically fit a rigid 2-D transformation (rotation + translation, no scale)
to exactly two point correspondences supplied as a `4 × 2` matrix `x`, where
each column is `[x_from; y_from; x_to; y_to]`.

Returns a one-element `Vector{AffineMap}` containing the **forward** transform
(mapping `from`-frame coordinates to `to`-frame coordinates), or an empty
vector when the sample is degenerate (i.e. the two `from`-points coincide).

The rotation angle ``\\theta`` is determined analytically via the complex-number
identity

```math
e^{i\\theta} = \\frac{\\Delta q \\cdot \\overline{\\Delta p}}{|\\Delta p|^2}
```

where ``\\Delta p = p_2 - p_1`` and ``\\Delta q = q_2 - q_1``.
"""
function _fit_minimal_rigid(x)
    p1x, p1y = x[1, 1], x[2, 1]
    p2x, p2y = x[1, 2], x[2, 2]
    q1x, q1y = x[3, 1], x[4, 1]
    q2x, q2y = x[3, 2], x[4, 2]

    dpx, dpy = p2x - p1x, p2y - p1y
    dqx, dqy = q2x - q1x, q2y - q1y

    dp_sq = dpx^2 + dpy^2
    dp_sq < _DEGENERATE_SQ && return AffineMap[]

    # Complex product Δq · conj(Δp) / |Δp|²
    c_num = dqx * dpx + dqy * dpy   # Re
    s_num = dqy * dpx - dqx * dpy   # Im

    # Normalise to a unit rotation (rigid: |R| = 1)
    r_sq = c_num^2 + s_num^2
    r_sq < _DEGENERATE_SQ && return AffineMap[]
    inv_r = inv(sqrt(r_sq))
    cosθ = c_num * inv_r
    sinθ = s_num * inv_r

    R = [cosθ -sinθ; sinθ cosθ]
    t = [q1x - R[1, 1] * p1x - R[1, 2] * p1y,
         q1y - R[2, 1] * p1x - R[2, 2] * p1y]

    return [AffineMap(R, t)]
end

"""
    _fit_minimal_similarity(x)

Analytically fit a similarity 2-D transformation (rotation + isotropic scale +
translation) to exactly two point correspondences supplied as a `4 × 2` matrix
`x`, where each column is `[x_from; y_from; x_to; y_to]`.

Returns a one-element `Vector{AffineMap}` containing the **forward** transform,
or an empty vector for degenerate input.  The scale–rotation complex factor is:

```math
A = \\frac{\\Delta q \\cdot \\overline{\\Delta p}}{|\\Delta p|^2}
\\quad (|A| = s,\\; \\arg A = \\theta)
```
"""
function _fit_minimal_similarity(x)
    p1x, p1y = x[1, 1], x[2, 1]
    p2x, p2y = x[1, 2], x[2, 2]
    q1x, q1y = x[3, 1], x[4, 1]
    q2x, q2y = x[3, 2], x[4, 2]

    dpx, dpy = p2x - p1x, p2y - p1y
    dqx, dqy = q2x - q1x, q2y - q1y

    dp_sq = dpx^2 + dpy^2
    dp_sq < _DEGENERATE_SQ && return AffineMap[]

    # Complex division Δq / Δp = (Δq · conj(Δp)) / |Δp|²
    a = (dqx * dpx + dqy * dpy) / dp_sq   # s·cos θ
    b = (dqy * dpx - dqx * dpy) / dp_sq   # s·sin θ

    # Similarity matrix  M = s·R_θ = [a -b; b a]
    M = [a -b; b a]
    t = [q1x - M[1, 1] * p1x - M[1, 2] * p1y,
         q1y - M[2, 1] * p1x - M[2, 2] * p1y]

    return [AffineMap(M, t)]
end

# ──────────────────────────────────────────────────────────────────
# RANSAC distance / verification function
# ──────────────────────────────────────────────────────────────────

"""
    _correspondence_distfn(M_candidates, x, t)

RANSAC verification function for 2-D point correspondences.

- `M_candidates`: collection of `AffineMap` forward transforms returned by
  the fitting function.
- `x`: `4 × N` data matrix; each column is `[x_from; y_from; x_to; y_to]`.
- `t`: pixel-distance threshold for inlier classification.

Returns `(inliers, best_M)` where `inliers` is a `Vector{Int}` of inlier column
indices and `best_M` is the model with the most inliers.
"""
function _correspondence_distfn(M_candidates, x, t)
    best_inliers = Int[]
    best_M = first(M_candidates)

    t_sq = t^2

    for M in M_candidates
        inliers = Int[]
        A, b = M.linear, M.translation
        for i in axes(x, 2)
            pfx, pfy = x[1, i], x[2, i]
            ptx, pty = x[3, i], x[4, i]
            # Forward transform: p_to_pred = A * p_from + b
            pred_x = A[1, 1] * pfx + A[1, 2] * pfy + b[1]
            pred_y = A[2, 1] * pfx + A[2, 2] * pfy + b[2]
            dx = pred_x - ptx
            dy = pred_y - pty
            dx^2 + dy^2 < t_sq && push!(inliers, i)
        end
        if length(inliers) > length(best_inliers)
            best_inliers = inliers
            best_M = M
        end
    end

    return best_inliers, best_M
end
