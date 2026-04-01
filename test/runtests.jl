using ParallelTestRunner: runtests, find_tests, parse_args
using Astroalign
using Test

const init_code = quote
    const Data = (
        # 3-4-5 triangle in 1st quadrant
        img_to = [
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 0 0 1 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
        ],

        points_to = (
            (xcenter = 0, ycenter = 0),
            (xcenter = 0, ycenter = 4),
            (xcenter = 3, ycenter = 0),
        ),

        # Asterism shifted vertically three units
        img_from = [
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 0 0 1 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
        ],

        points_from = (
            (xcenter = 0, ycenter = -3),
            (xcenter = 0, ycenter =  1),
            (xcenter = 3, ycenter = -3),
        ),

        combinations_to = [
            [(xcenter = 0, ycenter = 0), (xcenter = 0, ycenter =  4), (xcenter = 3, ycenter =  0)],
        ],

        combinations_from = [
            [(xcenter = 0, ycenter = -3), (xcenter = 0, ycenter =  1), (xcenter = 3, ycenter = -3)],
        ],

       invariants = [
           5/4
           4/3;;
       ],

        matched_triangle_to = [
            (xcenter = 0, ycenter =  0),
            (xcenter = 0, ycenter =  4),
            (xcenter = 3, ycenter =  0),
        ],

        matched_triangle_from = [
            (xcenter = 0, ycenter = -3),
            (xcenter = 0, ycenter =  1),
            (xcenter = 3, ycenter = -3),
        ],
    )
end

args = parse_args(Base.ARGS)
testsuite = find_tests(@__DIR__)

runtests(Astroalign, args; testsuite, init_code)
