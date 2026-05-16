"""
    _triangle_invariants(phot)

Returns all combinations (``C``) of three candidate point sources from the table of sources `phot` returned by [`Photometry.Aperture.photometry`](@extref), and the computed invariant ``\\mathscr M`` for each according to Eq. 3 from [_Beroiz, M., Cabral, J. B., & Sanchez, B. (2020)_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract).
"""
function _triangle_invariants(phot)
    @inline function sort3(a, b, c)
        a, b = minmax(a, b)
        b, c = minmax(b, c)
        a, b = minmax(a, b)
        return a, b, c  # sorted ascending: small, mid, large
    end
    xs = phot.xcenter
    ys = phot.ycenter
    n = length(phot)
    ntriangles = binomial(n, 3)
    ℳ = Matrix{Float64}(undef, 2, ntriangles)
    C = Vector{NTuple{3,Int}}(undef, ntriangles)

    # Enumerate all combinations(1:n,3) triangles via nested loops over strictly increasing index triples
    # (i < j < l), storing results in pre-allocated C and ℳ indexed by the flat counter k.
    k = 0
    for i in 1:n-2
        xi, yi = xs[i], ys[i]
        for j in i+1:n-1
            xj, yj = xs[j], ys[j]
            dx_ji, dy_ji = xj - xi, yj - yi
            d2_ji = dx_ji^2 + dy_ji^2  # squared distance between sources i and j
            for l in j+1:n
                k += 1
                xl, yl = xs[l], ys[l]
                dx_lj, dy_lj = xl - xj, yl - yj
                dx_li, dy_li = xl - xi, yl - yi
                d2_lj = dx_lj^2 + dy_lj^2  # squared distance between sources j and l
                d2_li = dx_li^2 + dy_li^2  # squared distance between sources i and l
                
                small2, mid2, large2 = sort3(d2_ji, d2_lj, d2_li)
                ℳ[1, k] = sqrt(large2 / mid2)
                ℳ[2, k] = sqrt(mid2 / small2)
                C[k] = _canonical_vertex_order(i, j, l, d2_ji, d2_lj, d2_li, xs, ys)
            end
        end
    end

    return C, ℳ
end

"""
    _canonical_vertex_order(i, j, l, d2_ji, d2_lj, d2_li, xs, ys)

Re-order the three vertices of a triangle with `x` and `y` coordinates
`x = (xs[i], xs[j], xs[l])` and `y = (ys[i], ys[j], ys[l])`
into a canonical form so that:

1. The apex (vertex opposite the longest edge) is last.
2. The two base vertices are ordered counter-clockwise (positive cross product).

This canonical form is preserved under rotation and translation, so corresponding triangles
in two images receive the same vertex permutation and produce geometrically consistent point
correspondences.

The squared edge lengths `d2_ji`, `d2_lj`, `d2_li` are accepted as arguments rather than
recomputed, since they are already available at the call site in [`_triangle_invariants`](@ref).
"""
@inline function _canonical_vertex_order(i, j, l, d2_ji, d2_lj, d2_li, xs, ys)
    # Identify apex (vertex opposite the longest edge) and base vertices
    if d2_ji >= d2_lj && d2_ji >= d2_li
        v1, v2, apex = i, j, l   # longest edge is i-j, apex is l
    elseif d2_lj >= d2_ji && d2_lj >= d2_li
        v1, v2, apex = j, l, i   # longest edge is j-l, apex is i
    else
        v1, v2, apex = i, l, j   # longest edge is i-l, apex is j
    end

    # Enforce CCW winding: swap base vertices if cross product is negative
    cross = (xs[v2] - xs[v1]) * (ys[apex] - ys[v1]) - (ys[v2] - ys[v1]) * (xs[apex] - xs[v1])
    cross < 0 && ((v1, v2) = (v2, v1))

    return (v1, v2, apex)
end

"""
    _build_correspondences(C_from, ℳ_from, phot_from, C_to, ℳ_to, phot_to)

Build a `2 × 3 × 2 × N` array of candidate triangle-level correspondences
between the `from` and `to` frames. The `C` and `ℳ` are the combinations of three
points and their invariants as returned by [`_triangle_invariants`](@ref), which
guarantees that both `C_from` and `C_to` are already in canonical vertex order
(base vertices CCW, apex last).

The axes are `[coord, vertex, frame, match]`:

- `coord ∈ {1 = x, 2 = y}`
- `vertex ∈ {1, 2, 3}` — canonical vertex index within the triangle
- `frame ∈ {1 = from, 2 = to}`
- `match` — index of the candidate triangle pair (``N`` total)

So `out[:, v, 1, n]` is the `(x, y)` position of vertex `v` in the `from`
frame for match `n`, and `out[:, v, 2, n]` is the corresponding position in
the `to` frame. Because both frames share the same canonical vertex ordering,
corresponding vertices across frames are geometrically consistent and suitable
for direct use in transform estimation.
"""
function _build_correspondences(C_from, ℳ_from, phot_from, C_to, ℳ_to, phot_to)
    (isempty(C_from) || isempty(C_to)) && return zeros(2, 3, 2, 0)

    # Build the KD-tree once; query per-triangle inside the loop to avoid
    # materialising the full idxs vector
    tree = KDTree(ℳ_to)

    out = Array{Float64}(undef, 2, 3, 2, length(C_from))
    xs_from, ys_from = phot_from.xcenter, phot_from.ycenter
    xs_to, ys_to = phot_to.xcenter, phot_to.ycenter

    for i in eachindex(C_from)
        # Find nearest neighbor of ℳ_from[:, i] in ℳ_to, returning the index of the corresponding triangle in the to-frame
        idx, _ = nn(tree, view(ℳ_from, :, i))
        C_from_i = C_from[i]
        C_to_i = C_to[idx]
        for v in 1:3
            out[1, v, 1, i] = xs_from[C_from_i[v]]
            out[2, v, 1, i] = ys_from[C_from_i[v]]
            out[1, v, 2, i] = xs_to[C_to_i[v]]
            out[2, v, 2, i] = ys_to[C_to_i[v]]
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

"""
    _fit_triangle(x, scale::Bool)

Internal: fit a rigid or similarity transform to the single triangle match.

`x` is a 2×3×2×1 view from ransac: [coord, vertex, frame, 1]

frame = 1 => from image, frame = 2 => to image
"""
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
_fit_minimal_rigid_triangle(x) = _fit_triangle(x, false)

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

    return (inliers = best_inliers, model = best_M)
end
