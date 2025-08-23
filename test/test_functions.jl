@testmodule Data begin
    # 3-4-5 triangle in 1st and 4th quadrant
    const img_to = [
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 1 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
   ]

    const points_to = (
        (xcenter = 0, ycenter = 0),
        (xcenter = 0, ycenter = 4),
        (xcenter = 0, ycenter = -4),
        (xcenter = 3, ycenter = 0),
    )

    # Asterism shifted vertically one unit
    const img_from = [
        0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 1 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
    ]

    const points_from = (
        (xcenter = 0, ycenter =  1),
        (xcenter = 0, ycenter =  5),
        (xcenter = 0, ycenter = -3),
        (xcenter = 3, ycenter =  1),
    )

    const combinations_to = [
        [(xcenter = 0, ycenter = 0), (xcenter = 0, ycenter =  4), (xcenter = 0, ycenter = -4)],
        [(xcenter = 0, ycenter = 0), (xcenter = 0, ycenter =  4), (xcenter = 3, ycenter =  0)],
        [(xcenter = 0, ycenter = 0), (xcenter = 0, ycenter = -4), (xcenter = 3, ycenter =  0)],
        [(xcenter = 0, ycenter = 4), (xcenter = 0, ycenter = -4), (xcenter = 3, ycenter =  0)],
    ]

    const combinations_from = [
        [(xcenter = 0, ycenter = 1), (xcenter = 0, ycenter =  5), (xcenter = 0, ycenter = -3)],
        [(xcenter = 0, ycenter = 1), (xcenter = 0, ycenter =  5), (xcenter = 3, ycenter =  1)],
        [(xcenter = 0, ycenter = 1), (xcenter = 0, ycenter = -3), (xcenter = 3, ycenter =  1)],
        [(xcenter = 0, ycenter = 5), (xcenter = 0, ycenter = -3), (xcenter = 3, ycenter =  1)],
    ]

    const invariants = [
        8/4 5/4 5/4 8/5
        4/4 4/3 4/3 5/5
    ]

    const matched_triangle_to = [
        (xcenter = 0, ycenter =  0),
        (xcenter = 0, ycenter =  4),
        (xcenter = 0, ycenter = -4),
    ]

    const matched_triangle_from = [
        (xcenter = 0, ycenter = 1),
        (xcenter = 0, ycenter = 5),
        (xcenter = 0, ycenter = -3),
    ]
end

@testitem "triangle_invariants" setup=[Data] begin
    using Astroalign: triangle_invariants

    points = Data.points_to
    combinations = Data.combinations_to
    invariants = Data.invariants

    C, ℳ = triangle_invariants(points)

    @test length(C) == 4
    @test collect(C) == combinations
    @test ℳ == invariants
end

@testitem "find_nearest" setup=[Data] begin
    using Astroalign: find_nearest

    C_to, ℳ_to = Data.combinations_to, Data.invariants
    C_from, ℳ_from = Data.combinations_from, Data.invariants

    sol_to, sol_from = find_nearest(C_to, ℳ_to, C_from, ℳ_from)

    @test sol_to == Data.matched_triangle_to
    @test sol_from == Data.matched_triangle_from
end

@testitem "get_sources" begin
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

@testitem "align_frame" setup = [Data] begin
    using Astroalign: align_frame

    img_to = Data.img_to
    img_from = Data.img_from

    img_aligned, params = align_frame(img_to, img_from)

    @test img_aligned == img_to
    @test params.point_map == [
        [1.0, 6.0] => [2.0, 6.0],
        [5.0, 6.0] => [6.0, 6.0],
        [9.0, 6.0] => [10.0, 6.0],
    ]
    @test params.tfm.linear == [1 0; 0 1]
    @test params.tfm.translation == [-1.0, 0.0] # AstroImages.jl orientation convention
end

@testitem "api" setup = [Data] begin
    import Astroalign

    img_to = Data.img_to
    img_from = Data.img_from

    img_aligned, p = Astroalign.align_frame(img_to, img_from;
        f = Astroalign.PSF(params = (x = 6, y = 6, fwhm = 1))
    )

    @test img_aligned isa AbstractMatrix
    @test propertynames(p) == (:point_map, :tfm, :C_to, :ℳ_to, :C_from, :ℳ_from)
end
