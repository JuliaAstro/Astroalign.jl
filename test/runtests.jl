using Test
using TestItemRunner

@testitem "triangle_invariants" begin
    using Astroalign: triangle_invariants

    points = (
        (xcenter = 0, ycenter = 0),
        (xcenter = 0, ycenter = 4),
        (xcenter = 0, ycenter = -4),
        (xcenter = 3, ycenter = 0),
    )

    C, ℳ = triangle_invariants(points)

    combinations = [
        [(xcenter = 0, ycenter = 0), (xcenter = 0, ycenter =  4), (xcenter = 0, ycenter = -4)],
        [(xcenter = 0, ycenter = 0), (xcenter = 0, ycenter =  4), (xcenter = 3, ycenter =  0)],
        [(xcenter = 0, ycenter = 0), (xcenter = 0, ycenter = -4), (xcenter = 3, ycenter =  0)],
        [(xcenter = 0, ycenter = 4), (xcenter = 0, ycenter = -4), (xcenter = 3, ycenter =  0)],
    ]

    @test length(C) == 4
    @test collect(C) == combinations
end
