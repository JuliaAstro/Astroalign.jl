
function render_stars(positions_rc_amp, img_size, fwhm)
    img = zeros(Float32, img_size)
    fwhm_int = ceil(Int, 2.5 * fwhm)
    for (r, c, amp) in positions_rc_amp
        r0, c0 = round(Int, r), round(Int, c)
        for ir in max(1, r0 - fwhm_int):min(img_size[1], r0 + fwhm_int),
                ic in max(1, c0 - fwhm_int):min(img_size[2], c0 + fwhm_int)
            # PSFModels convention: first arg → row coord (x), second → col (y)
            img[ir, ic] += gaussian(Float32(ir), Float32(ic);
                x = Float32(r), y = Float32(c),
                fwhm = Float32(fwhm), amp = Float32(amp))
        end
    end
    return img
end

@testset "internal fitting functions for RANSAC" begin
    using Astroalign: _fit_minimal_rigid, _fit_minimal_similarity
    using CoordinateTransformations: AffineMap
    using LinearAlgebra: norm

    # Two perfectly-known correspondences: rotation of 30° + translation (5, -3)
    θ = deg2rad(30)
    c, s = cos(θ), sin(θ)
    R = [c -s; s c]
    t = [5.0, -3.0]
    p1 = [10.0, 20.0]
    p2 = [50.0, 80.0]
    q1 = R * p1 + t
    q2 = R * p2 + t

    x = [p1[1] p2[1]; p1[2] p2[2]; q1[1] q2[1]; q1[2] q2[2]]

    # ── _fit_minimal_rigid ──────────────────────────────────────────────────
    @testset "rigid" begin
        result = _fit_minimal_rigid(x)
        @test length(result) == 1
        M = only(result)
        @test M isa AffineMap
        @test M.linear ≈ R     atol=1e-10
        @test M.translation ≈ t atol=1e-10

        # Degenerate: both from-points coincide
        x_deg = copy(x); x_deg[1:2, 2] .= x_deg[1:2, 1]
        @test isempty(_fit_minimal_rigid(x_deg))
    end

    # ── _fit_minimal_similarity ─────────────────────────────────────────────
    @testset "similarity with scale" begin
        # Scale of 1.5, rotation 20°, translation (10, -7)
        s_factor = 1.5
        θ2 = deg2rad(20)
        c2, s2 = cos(θ2), sin(θ2)
        M_true = [s_factor*c2  -s_factor*s2; s_factor*s2  s_factor*c2]
        t2 = [10.0, -7.0]
        q1s = M_true * p1 + t2
        q2s = M_true * p2 + t2
        xs = [p1[1] p2[1]; p1[2] p2[2]; q1s[1] q2s[1]; q1s[2] q2s[2]]

        result = _fit_minimal_similarity(xs)
        @test length(result) == 1
        M = only(result)
        @test M.linear ≈ M_true atol=1e-10
        @test M.translation ≈ t2 atol=1e-10
    end
end

@testset "rigid alignment on synthetic images" begin
    # All coordinates are in Astroalign's native (xcenter=row, ycenter=col) convention.
    # Stars are rendered with PSFModels.gaussian(row_eval, col_eval; x=row_star, y=col_star).
    # The forward transform T_fwd maps img_to star (row, col) positions to img_from positions.
    # align_frame recovers this as params.tfm (backward: img_to → img_from), so:
    #   params.tfm.linear     ≈ R_fwd
    #   params.tfm.translation ≈ t_fwd
    using Astroalign: align_frame
    using PSFModels: gaussian
    using CoordinateTransformations: AffineMap
    using LinearAlgebra: I, norm, dot
    using Random: MersenneTwister
    rng = MersenneTwister(1234)

    img_size  = (500, 500)
    fwhm      = 4.0
    θ         = deg2rad(25)

    # Rotation + translation in (row, col) = (xcenter, ycenter) space
    c_θ, s_θ  = cos(θ), sin(θ)
    R_fwd     = [c_θ -s_θ; s_θ c_θ]          # forward rotation in row-col space
    center_rc = [250.0, 250.0]               # center of rotation (row, col)
    δ_rc      = [12.0, -10.0]                # translation (Δrow, Δcol)
    t_fwd     = ((I - R_fwd) * center_rc) + δ_rc
    T_fwd     = AffineMap(R_fwd, t_fwd)

    nstars = 20
    # Stars randomly inside [100, 400] × [100, 400] in (row, col)
    stars_to_rc = [(rand(rng) * 300 + 100, rand(rng) * 300 + 100, rand(rng) * 0.5 + 0.5)
                   for _ in 1:nstars]

    # Forward-transform star positions for img_from
    stars_from_rc = map(stars_to_rc) do (r, c, amp)
        pf = T_fwd([r, c])
        (pf[1], pf[2], amp)
    end

    # All from-stars stay within image bounds (validated by the transform geometry)
    @test all(stars_from_rc) do (r, c, _)
        1 ≤ round(Int, r) ≤ 500 && 1 ≤ round(Int, c) ≤ 500
    end

    img_to   = render_stars(stars_to_rc,   img_size, fwhm)
    img_from = render_stars(stars_from_rc, img_size, fwhm)

    img_aligned, params = align_frame(img_to, img_from;
        scale            = false,
        min_fwhm         = (0.5, 0.5),
        N_max            = nstars,
        ransac_threshold = 2.0,
    )

    # Recovered backward transform (img_to → img_from) should match T_fwd
    @test params.tfm.linear ≈ R_fwd  atol=0.01
    @test params.tfm.translation ≈ t_fwd  atol=1.0

    # RANSAC should utilise most of the stars as inliers
    @test length(params.inlier_idxs) ≥ nstars

    # Warped image should correlate strongly with img_to
    a = vec(Float64.(img_aligned))
    b = vec(Float64.(img_to))
    mask = .!(isnan.(a) .| iszero.(a) .| iszero.(b))
    μa  = sum(a[mask]) / count(mask)
    μb  = sum(b[mask]) / count(mask)
    da  = a[mask] .- μa
    db  = b[mask] .- μb
    corr = dot(da, db) / (norm(da) * norm(db))
    @test corr > 0.98
end

@testset "similarity alignment (scale=true) on synthetic images" begin
    using Astroalign: align_frame
    using PSFModels: gaussian
    using CoordinateTransformations: AffineMap
    using LinearAlgebra: I, norm
    using Random: MersenneTwister

    rng = MersenneTwister(5678)
    img_size  = (500, 500)
    fwhm      = 4.0
    θ         = deg2rad(15)
    scale_fac = 0.9                    # slight zoom-out in (row, col) space
    c_θ, s_θ  = cos(θ), sin(θ)
    M_fwd     = scale_fac * [c_θ -s_θ; s_θ c_θ]
    center_rc = [250.0, 250.0]
    δ_rc      = [8.0, -5.0]
    t_fwd     = center_rc - M_fwd * center_rc + δ_rc
    T_fwd     = AffineMap(M_fwd, t_fwd)

    nstars = 20
    stars_to_rc = [(rand(rng) * 200 + 150, rand(rng) * 200 + 150, rand(rng) * 0.5 + 0.5)
                   for _ in 1:nstars]
    stars_from_rc = map(stars_to_rc) do (r, c, amp)
        pf = T_fwd([r, c])
        (pf[1], pf[2], amp)
    end

    @test all(stars_from_rc) do (r, c, _)
        1 ≤ round(Int, r) ≤ 500 && 1 ≤ round(Int, c) ≤ 500
    end

    img_to   = render_stars(stars_to_rc,   img_size, fwhm)
    img_from = render_stars(stars_from_rc, img_size, fwhm)

    img_aligned, params = align_frame(img_to, img_from;
        scale            = true,
        min_fwhm         = (0.5, 0.5),
        N_max            = nstars,
        ransac_threshold = 2.0,
    )

    @test params.tfm.linear ≈ M_fwd  atol=0.02
    @test params.tfm.translation ≈ t_fwd  atol=1.0
    @test length(params.inlier_idxs) ≥ nstars
end

@testset "RANSAC: two sub-images from a 2000 × 2000 master field" begin
    # Two 512 × 512 sub-images are extracted from a large synthetic field via
    # ImageTransformations.warp with known backward transforms.  align_frame
    # must recover the relative rotation to better than ≈1° and produce a
    # strongly-correlated aligned image.
    #
    # Geometry:
    #   img_to  – axis-aligned crop covering master rows/cols [745:1256]
    #   img_from – same *centre* (master 1000.5, 1000.5), rotated 22° CCW
    #
    # All 50 stars are place in master [820:1180, 820:1180] so they lie
    # inside both sub-images regardless of the 22° rotation.
    using Astroalign: align_frame
    using PSFModels: gaussian
    using CoordinateTransformations: AffineMap, LinearMap, Translation
    using ImageTransformations: warp
    using StaticArrays: SVector
    using LinearAlgebra: norm, dot
    using Random: MersenneTwister

    function render_master(star_r, star_c, amps, sz, fwhm)
        img = zeros(Float64, sz)
        fwhm_int = ceil(Int, 3.0 * fwhm)
        for (r, c, amp) in zip(star_r, star_c, amps)
            r0, c0 = round(Int, r), round(Int, c)
            for ir in max(1, r0 - fwhm_int):min(sz[1], r0 + fwhm_int),
                    ic in max(1, c0 - fwhm_int):min(sz[2], c0 + fwhm_int)
                img[ir, ic] += gaussian(Float64(ir), Float64(ic);
                    x = Float64(r), y = Float64(c),
                    fwhm = Float64(fwhm), amp = Float64(amp))
            end
        end
        return img
    end

    rng = MersenneTwister(9999)

    master_size = (2000, 2000)
    fwhm = 5.0
    nstars = 50

    # Stars concentrated in [820:1180, 820:1180] – entirely within img_to AND
    # within 256*cos(11°) ≈ 251 pixels of master centre, so within img_from.
    star_r = rand(rng, 820:1180, nstars)
    star_c = rand(rng, 820:1180, nstars)
    star_a = rand(rng, nstars) .* 0.5 .+ 0.5

    master = render_master(star_r, star_c, star_a, master_size, fwhm)

    # img_to: axis-aligned crop, master offset (744, 744)
    off1      = SVector(744.0, 744.0)
    tfm1_back = Translation(off1)
    img_to    = warp(master, tfm1_back, (1:512, 1:512))

    # img_from: same master centre (1000.5, 1000.5), rotated 22° CCW
    θ          = deg2rad(22)
    c_θ, s_θ   = cos(θ), sin(θ)
    R_rot      = [c_θ -s_θ; s_θ c_θ]           # rotation in (row, col) space
    sub_ctr    = SVector(256.5, 256.5)
    master_ctr = SVector(1000.5, 1000.5)
    tfm2_back  = Translation(master_ctr) ∘ LinearMap(R_rot) ∘ Translation(-sub_ctr)
    img_from   = warp(master, tfm2_back, (1:512, 1:512))

    img_aligned, params = align_frame(img_to, img_from;
        scale            = false,
        min_fwhm         = (1.0, 1.0),
        N_max            = 20,
        ransac_threshold = 5.0,
        k_nearest        = 8,
    )

    # params.tfm maps img_to → img_from in (row, col) space.
    # tfm = inv(tfm2_back) ∘ tfm1_back → linear part ≈ R_rot' = R(-22°)
    @test params.tfm.linear ≈ R_rot'  atol=0.02

    @test length(params.inlier_idxs) ≥ 6

    a = vec(img_aligned)
    b = vec(Float64.(img_to))
    mask = .!(isnan.(a) .| iszero.(a) .| iszero.(b))
    @test count(mask) > 0
    a_m, b_m = a[mask], b[mask]
    μa, μb   = sum(a_m) / length(a_m), sum(b_m) / length(b_m)
    da, db   = a_m .- μa, b_m .- μb
    corr     = dot(da, db) / (norm(da) * norm(db))
    @test corr > 0.95
end

