@testmodule Data begin
    # 3-4-5 triangle in 1st and 4th quadrant
    const points_to = (
        (xcenter = 0, ycenter = 0),
        (xcenter = 0, ycenter = 4),
        (xcenter = 0, ycenter = -4),
        (xcenter = 3, ycenter = 0),
    )

    # Asterism shifted vertically one unit
    points_from = (
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
