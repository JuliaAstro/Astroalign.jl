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
    idxs, dists = nn(KDTree(ℳ_to), ℳ_from)
    idx_from = argmin(dists)
    idx_to = idxs[idx_from]
    sol_to = collect(C_to)[idx_to]
    sol_from = collect(C_from)[idx_from]
    return sol_to, sol_from
end
