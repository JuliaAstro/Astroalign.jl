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

@testset "align_frame" begin
    using Astroalign: align_frame

    img_to = Data.img_to
    img_from = Data.img_from

    img_aligned = align_frame(img_from, img_to; box_size = 1, ap_radius = 1, min_fwhm = (0.1, 0.1))

    @test img_aligned ≈ img_to
end

@testset "find_transform" begin
    using Astroalign: find_transform

    img_to = Data.img_to
    img_from = [(9, 9), (5, 6), (9, 6)]

    tfm, params = find_transform(img_from, img_to; box_size = 1, ap_radius = 1, min_fwhm = (0.1, 0.1))

    @test params.point_map == [
        [9.0, 9.0] => [6.0, 9.0],
        [5.0, 6.0] => [2.0, 6.0],
        [9.0, 6.0] => [6.0, 6.0]
    ]
    @test tfm.linear ≈ [1 0; 0 1]
    @test tfm.translation ≈ [-3.0, 0.0]
end

@testset "photometry" begin
    using Astroalign: _photometry, PSF

    img_to = Data.img_to

    phot_to, _ = _photometry(img_to;
        box_size = 5,
        ap_radius = 2,
        min_fwhm = 0.1,
        nsigma = 1,
        f = PSF(),
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
        f = PSF(),
        N_max = 10,
        use_fitpos = true,
    )
    @test typeof(phot_to.xcenter) <: Vector{Float64}
    @test typeof(phot_to.ycenter) <: Vector{Float64}
end

@testset "api"  begin
    using Astroalign: PSF, find_transform, apply_transform

    img_to = Data.img_to
    img_from = Data.img_from

    tfm, p = find_transform(img_to, img_from;
        box_size = 1,
        ap_radius = 1,
        min_fwhm = 0.1,
        f = PSF(params = (x = 6, y = 6, fwhm = 0.2))
    )
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
