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
    _build_correspondences(C_from, ℳ_from, C_to, ℳ_to)

Build a `2 × 3 × 2 × N` array of candidate triangle-level correspondences
between the `from` and `to` frames. The `C` and `ℳ` are the combinations of three
points and their invariants as returned by [`Astroalign.triangle_invariants`](@ref).

The axes are `[coord, vertex, frame, match]`:

- `coord ∈ {1 = x, 2 = y}`
- `vertex ∈ {1, 2, 3}` — canonical vertex index within the triangle
- `frame ∈ {1 = from, 2 = to}`
- `match` — index of the candidate triangle pair (``N`` total)

So `out[:, v, 1, n]` is the `(x, y)` position of vertex `v` in the `from`
frame for match `n`, and `out[:, v, 2, n]` is the corresponding position in
the `to` frame. Vertices are ordered canonically via
[`_canonical_vertex_order`](@ref), so corresponding triangles receive the
same geometric vertex assignment.
"""
function _build_correspondences(C_from, ℳ_from, C_to, ℳ_to)
    C_from_list = collect(C_from)
    C_to_list   = collect(C_to)

    (isempty(C_from_list) || isempty(C_to_list)) && return zeros(2, 3, 2, 0)

    # Get most similar triangle in to-frame for each from-frame triangle, using the invariants as features
    idxs, _ = nn(KDTree(ℳ_to), ℳ_from)

    out = Array{Float64}(undef, 2, 3, 2, length(C_from_list))

    for i in eachindex(C_from_list)
        canon_from = _canonical_vertex_order(C_from_list[i]...)
        canon_to = _canonical_vertex_order(C_to_list[idxs[i]]...)
        for v in 1:3
            out[1, v, 1, i] = canon_from[v].xcenter
            out[2, v, 1, i] = canon_from[v].ycenter
            out[1, v, 2, i] = canon_to[v].xcenter
            out[2, v, 2, i] = canon_to[v].ycenter
        end
    end

    return out
end

# ──────────────────────────────────────────────────────────────────
# Triangle-level fitting functions for ConsensusFitting.ransac
# ──────────────────────────────────────────────────────────────────

# Minimum absolute value of the cross product (= 2 × signed triangle area in
# pixel² units) below which a triangle is considered degenerate (collinear).
const _DEGENERATE_AREA = 1.0

# Internal: fit a rigid or similarity transform to the single triangle match.
# x is a 2×3×2×1 view from ransac: [coord, vertex, frame, 1]
#   frame = 1 → from image,  frame = 2 → to image
function _fit_triangle(x, scale::Bool)
    pts_from = view(x, :, :, 1, 1)   # 2×3 — from-frame vertices
    pts_to   = view(x, :, :, 2, 1)   # 2×3 — to-frame vertices

    # Reject degenerate (collinear) triangles via the cross product of two edges
    v1 = view(pts_from, :, 2) .- view(pts_from, :, 1)
    v2 = view(pts_from, :, 3) .- view(pts_from, :, 1)
    abs(v1[1] * v2[2] - v1[2] * v2[1]) < _DEGENERATE_AREA && return AffineMap[]

    try
        return [kabsch(pts_from => pts_to; scale)]
    catch
        return AffineMap[]
    end
end

"""
    _fit_minimal_rigid_triangle(x)

Fit a rigid 2-D transformation (rotation + translation) to the single triangle
correspondence in the `2 × 3 × 2 × 1` RANSAC sample view `x`.

Axes are `[coord, vertex, frame, 1]` where `frame = 1` is the `from` image and
`frame = 2` is the `to` image.  The three vertex pairs are fitted in a
least-squares sense via the Kabsch algorithm.

Returns a one-element `Vector{AffineMap}` (forward: `from` → `to`), or an
empty vector when the from-vertices are collinear.
"""
_fit_minimal_rigid_triangle(x)      = _fit_triangle(x, false)

"""
    _fit_minimal_similarity_triangle(x)

Fit a similarity 2-D transformation (rotation + isotropic scale + translation)
to the single triangle correspondence in the `2 × 3 × 2 × 1` RANSAC sample
view `x`.  See [`_fit_minimal_rigid_triangle`](@ref) for the data layout and
fitting procedure.
"""
_fit_minimal_similarity_triangle(x) = _fit_triangle(x, true)

# ──────────────────────────────────────────────────────────────────
# RANSAC distance / verification function (triangle level)
# ──────────────────────────────────────────────────────────────────

"""
    _triangle_distfn(M_candidates, x, t)

RANSAC verification function for triangle-level correspondences.

`x` is a `2 × 3 × 2 × N` array with axes `[coord, vertex, frame, match]`
where `frame = 1` is the `from` image and `frame = 2` is the `to` image.

A triangle match is classified as an inlier when **all three** of its vertex
pairs satisfy `‖A·p_from + b − p_to‖ < t`.

Returns `(inliers, best_M)` where `inliers` is a vector of match indices.
"""
function _triangle_distfn(M_candidates, x, t)
    # x is 2×3×2×N — [coord, vertex, frame, match]
    best_inliers = Int[]
    best_M = first(M_candidates)
    t_sq = t^2

    for M in M_candidates
        inliers = Int[]
        A, b = M.linear, M.translation
        for i in axes(x, 4)
            ok = true
            for v in 1:3
                pfx, pfy = x[1, v, 1, i], x[2, v, 1, i]
                ptx, pty = x[1, v, 2, i], x[2, v, 2, i]
                pred_x = A[1, 1] * pfx + A[1, 2] * pfy + b[1]
                pred_y = A[2, 1] * pfx + A[2, 2] * pfy + b[2]
                dx, dy = pred_x - ptx, pred_y - pty
                if dx^2 + dy^2 ≥ t_sq
                    ok = false
                    break
                end
            end
            ok && push!(inliers, i)
        end
        if length(inliers) > length(best_inliers)
            best_inliers = inliers
            best_M = M
        end
    end

    return best_inliers, best_M
end