@testset "triangle_invariants" begin
    using Astroalign: _triangle_invariants

    points = Data.points_to
    combinations = Data.combinations_to
    invariants = Data.invariants

    C, ℳ = _triangle_invariants(points)

    @test length(C) == 1
    @test collect(C) == combinations
    @test ℳ == invariants
end

@testset "get_sources" begin
    using Astroalign: _get_sources

    img = [
        1 0
        0 0
    ]

    sources, subt, errs = _get_sources(img; box_size = 1, nsigma = 1, N_max = 10)

    @test first(Tuple(sources)) == (x = 1, y = 1, value = 1.0)
    @test subt == img
    @test errs == zero(subt)
end

@testset "align_frames" begin
    using Astroalign: align_frames

    img_to = Data.img_to
    img_from = Data.img_from

    img_aligned = align_frames(img_from, img_to; use_fitpos = false)

    @test img_aligned ≈ img_to
end

@testset "align_frames (vector input reuses phot_to)" begin
    using Astroalign: align_frames

    img_to = Data.img_to
    img_from = Data.img_from
    opts = (; use_fitpos = false)

    expected = align_frames(img_from, img_to; opts...)
    aligned = align_frames([img_from, img_from, img_from], img_to; opts...)

    @test length(aligned) == 3
    @test all(a -> a ≈ expected, aligned)
    @test all(a -> a ≈ img_to, aligned)
end

@testset "align_frames (single-arg convenience: first is reference)" begin
    using Astroalign: align_frames

    img_to = Data.img_to
    img_from = Data.img_from
    opts = (; use_fitpos = false)

    expected = align_frames([img_from, img_from], img_to; opts...)
    aligned = align_frames([img_to, img_from, img_from]; opts...)

    @test length(aligned) == 2
    @test aligned[1] ≈ expected[1]
    @test aligned[2] ≈ expected[2]
end

@testset "find_transform" begin
    using Astroalign: find_transform

    img_to = Data.img_to
    img_from = [(9, 9), (5, 6), (9, 6)]
    opts = (; use_fitpos = false)

    tfm, params = find_transform(img_from, img_to; opts...)

    @test params.point_map == [
        [9.0, 9.0] => [6.0, 9.0],
        [5.0, 6.0] => [2.0, 6.0],
        [9.0, 6.0] => [6.0, 6.0]
    ]
    @test tfm.linear ≈ [1 0; 0 1]
    @test tfm.translation ≈ [-3.0, 0.0]
end

@testset "find_transform reuses precomputed phot" begin
    using Astroalign: find_transform

    img_to = Data.img_to
    img_from = Data.img_from
    opts = (; use_fitpos = false)

    tfm_ref, params_ref = find_transform(img_from, img_to; opts...)

    # Passing the precomputed phot_to Table in place of img_to must yield the
    # same transform without rerunning photometry on img_to.
    tfm_reuse, params_reuse = find_transform(img_from, params_ref.phot_to; opts...)
    @test tfm_reuse.linear ≈ tfm_ref.linear
    @test tfm_reuse.translation ≈ tfm_ref.translation
    @test params_reuse.phot_to === params_ref.phot_to
    @test params_reuse.phot_to_params == ()

    # Symmetric: precomputed phot_from also works.
    tfm_both, params_both = find_transform(params_ref.phot_from, params_ref.phot_to; opts...)
    @test tfm_both.linear ≈ tfm_ref.linear
    @test tfm_both.translation ≈ tfm_ref.translation
    @test params_both.phot_from_params == ()
end

@testset "photometry" begin
    using Astroalign: _photometry, com_psf

    img_to = Data.img_to

    phot_to, _ = _photometry(img_to;
        box_size = 5,
        ap_radius = 2,
        min_fwhm = 0.1,
        nsigma = 1,
        f = com_psf,
        N_max = 10,
        use_fitpos = false,
    )
    @test typeof(phot_to.xcenter) <: Vector{Int64}
    @test typeof(phot_to.ycenter) <: Vector{Int64}

    phot_to, _ = _photometry(img_to;
        box_size = 5,
        ap_radius = 2,
        min_fwhm = 0.1,
        nsigma = 1,
        f = com_psf,
        N_max = 10,
        use_fitpos = true,
    )
    @test typeof(phot_to.xcenter) <: Vector{Float64}
    @test typeof(phot_to.ycenter) <: Vector{Float64}
end

@testset "api"  begin
    using Astroalign: find_transform, apply_transform

    img_to = Data.img_to
    img_from = Data.img_from

    tfm, p = find_transform(img_to, img_from; use_fitpos = false)
    img_aligned = apply_transform(tfm, img_from, img_to)

    @test img_aligned isa AbstractMatrix
    @test propertynames(p) == (
        :point_map,
        :correspondences,
        :inlier_idxs,
        :C_from,
        :ℳ_from,
        :C_to,
        :ℳ_to,
        :phot_from,
        :phot_to,
        :phot_from_params,
        :phot_to_params,
    )
end
