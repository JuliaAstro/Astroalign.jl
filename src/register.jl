"""
    triangle_invariants(phot)

Returns all combinations (``C``) of three candidate point sources from the table of sources `phot` returned by [`Photometry.Aperture.photometry`](@extref), and the computed invariant ``\\mathscr M`` for each according to Eq. 3 from [_Beroiz, M., Cabral, J. B., & Sanchez, B. (2020)_](https://ui.adsabs.harvard.edu/abs/2020A%26C....3200384B/abstract).
and a map ℳ, of two side ratios between smallest and mid and mid and largest.
"""
function triangle_invariants(phot)
    # changed this to be parity-conserving
    # TODO : allow only small variations in size
    # TODO : prefer large triangles over small ones

    C = combinations(phot, 3)
    ℳ = map(C) do (pa, pb, pc)
        # a, b, c = (
        #     (pa.ycenter, pa.xcenter),
        #     (pb.ycenter, pb.xcenter),
        #     (pc.ycenter, pc.xcenter),
        # )
        # Ls = sort!([euclidean(a, b), euclidean(b, c), euclidean(a, c)])
        Ls = triangle_distances(pa, pb, pc)
        (Ls[1] / Ls[2], Ls[2] / Ls[3])
    end |> stack
    return C, ℳ
end

# function triangle_distances(pa,pb,pc)
#         a, b, c = (
#             (pa.ycenter, pa.xcenter),
#             (pb.ycenter, pb.xcenter),
#             (pc.ycenter, pc.xcenter),
#         )
#     return sort!([euclidean(a, b), euclidean(b, c), euclidean(a, c)])
# end

"""
    ensure_clockwise!(a, b, c)

ensures the peaks are always ordered clockwise via inspecting their coordinates.
returns the ordered tuple of the input tuples
"""
function ensure_clockwise(a, b, c)
    cross = (b[1]-a[1])*(c[2]-a[2]) - (b[2]-a[2])*(c[1]-a[1])
    if cross > 0
        return (a, c, b)  # or mutate whichever container
    else
        return (a, b, c)
    end
end

function ensure_clockwise_largest!(sol)
    pa = sol[1]
    pb = sol[2]
    pc = sol[3]
    cross = (pb.xcenter-pa.xcenter)*(pc.ycenter-pa.ycenter) - (pb.ycenter-pa.ycenter)*(pc.xcenter-pa.xcenter)
    # reorder, if needed
    if cross > 0
        sol[1] = pa
        sol[2] = pc
        sol[3] = pb
    end
    a, b, c = (
            (sol[1].ycenter, sol[1].xcenter),
            (sol[2].ycenter, sol[2].xcenter),
            (sol[3].ycenter, sol[3].xcenter),
        )
    # start with the largest distance
    ab = euclidean(a, b); bc = euclidean(b, c); ca = euclidean(c, a)
    di = get_largest_distance_ids(ab,bc,ca)
    sol[1:3] .= sol[[di...]]
    sol
end

function get_largest_distance_ids(ab,bc,ca)
    if (ab > bc)
        if (bc > ca || ab > ca) # ab is largest
            return (1,2,3)
        else # ca is largest
            return (3,1,2)
        end
    else
        if (bc > ca) # bc is largest
            return (2,3,1)
        else # ca is largest
            return (3,1,2)
        end
    end
end

function triangle_distances(pa,pb,pc)
    a, b, c = (
            (pa.ycenter, pa.xcenter),
            (pb.ycenter, pb.xcenter),
            (pc.ycenter, pc.xcenter),
        )
    # ensure a right-handed order of all peaks (a,b,c)
    a,b,c = ensure_clockwise(a,b,c)
    ab = euclidean(a, b); bc = euclidean(b, c); ca = euclidean(c, a)
    di = get_largest_distance_ids(ab,bc,ca)
    # return a,b,c cyclically shifted to start with the largest distance
    return (ab,bc,ca)[[di...]]
end

"""
    find_nearest(C_to, ℳ_to, C_from, ℳ_from)

Return the closes pair of three points between the `from` and `to` frames
in the invariant ``\\mathscr M`` space as computed by [`Astroalign.triangle_invariants`](@ref).
"""
function find_nearest(C_to, ℳ_to, C_from, ℳ_from)
    # find the nearest neighbors regarding the triangles
    idxs, dists = nn(KDTree(ℳ_to), ℳ_from)
    # determine the best match:
    # TODO: ensure minimum conditions are met regarding largest distance, largest aspect ratio and minimum non-symmetry
    idx_from = argmin(dists)
    idx_to = idxs[idx_from]
    sol_to = collect(C_to)[idx_to]
    sol_from = collect(C_from)[idx_from]

    # ensure the triangles are ordered clockwise, starting from the largest distance
    # ensure a right-handed order of all peaks (a,b,c) and start at the largest distance
    ensure_clockwise_largest!(sol_to)
    ensure_clockwise_largest!(sol_from)

    # println("D1 =$(triangle_distances(sol_to...)))")
    # println("D2 =$(triangle_distances(sol_from...))")
    # println("Dsum1=$(sum(triangle_distances(sol_to...)))")
    # println("Dsum2=$(sum(triangle_distances(sol_from...)))")
    return sol_to, sol_from
end

