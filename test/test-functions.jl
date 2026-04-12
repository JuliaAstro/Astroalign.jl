@testset "triangle_invariants" begin
    using Astroalign: triangle_invariants

    points = Data.points_to
    combinations = Data.combinations_to
    invariants = Data.invariants

    C, ℳ = triangle_invariants(points)

    @test length(C) == 1
    @test collect(C) == combinations
    @test ℳ == invariants
end

@testset "get_sources" begin
    using Astroalign: get_sources

    img = [
        1 0
        0 0
    ]

    sources, subt, errs = get_sources(img)

    @test first(Tuple(sources)) == (x = 1, y = 1, value = 1.0)
    @test subt == img
    @test errs == zero(subt)
end

@testset "align_frame" begin
    using Astroalign: align_frame

    img_to = Data.img_to
    img_from = Data.img_from

    img_aligned, params = align_frame(img_from, img_to; min_fwhm = (0.1, 0.1))

    @test img_aligned ≈ img_to
    @test params.point_map == [
        [9.0, 9.0] => [6.0, 9.0], 
        [5.0, 6.0] => [2.0, 6.0], 
        [9.0, 6.0] => [6.0, 6.0]
    ]
    @test params.tfm.linear ≈ [1 0; 0 1]
    @test params.tfm.translation ≈ [-3.0, 0.0]
end

@testset "_photometry" begin
    using Astroalign: _photometry, PSF

    img_to = Data.img_to

    phot_to = _photometry(
        img_to,
        5, # box_size
        2, # ap_radius
        0.1, # min_fwhm
        1, # nsigma
        PSF();
        use_fitpos = false,
    )

    @test typeof(phot_to.xcenter) <: Vector{Int64}
    @test typeof(phot_to.ycenter) <: Vector{Int64}

    phot_to = _photometry(
        img_to,
        5, # box_size
        2, # ap_radius
        0.1, # min_fwhm
        1, # nsigma
        PSF();
        use_fitpos = true,
    )

    @test typeof(phot_to.xcenter) <: Vector{Float64}
    @test typeof(phot_to.ycenter) <: Vector{Float64}
end

@testset "api"  begin
    import Astroalign

    img_to = Data.img_to
    img_from = Data.img_from

    img_aligned, p = Astroalign.align_frame(img_to, img_from;
        f = Astroalign.PSF(params = (x = 6, y = 6, fwhm = 3))
    )

    @test img_aligned isa AbstractMatrix
    @test propertynames(p) == (
        :point_map,
        :tfm,
        :correspondences,
        :inlier_idxs,
        :C_from,
        :ℳ_from,
        :C_to,
        :ℳ_to,
    )
end