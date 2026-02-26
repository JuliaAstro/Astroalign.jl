using ParallelTestRunner: runtests, find_tests, parse_args
using Astroalign
using Test

const init_code = quote
    const Data = (
    # 3-4-5 triangle in 1st quadrant + an extra source in the 4th quadrant
    img_to = [
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 1 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
   ],

   points_to = (
        # 3-4-5
        (xcenter = 0, ycenter = 0),
        (xcenter = 0, ycenter = 4),
        (xcenter = 3, ycenter = 0),
       # Extra source 
       (xcenter = 1, ycenter = -2),
    ),

    # Asterism shifted vertically one unit
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
        0 0 0 0 0 0 1 0 0 0 0
    ],

    points_from = (
        # 3-4-5
        (xcenter = 0, ycenter = -3),
        (xcenter = 0, ycenter =  1),
        (xcenter = 3, ycenter = -3),
        # Extra source
        (xcenter = 1, ycenter = -5),
    ),

    combinations_to = [
        [(xcenter=0, ycenter=0), (xcenter=0, ycenter= 4), (xcenter=3, ycenter= 0)],
        [(xcenter=0, ycenter=0), (xcenter=0, ycenter= 4), (xcenter=1, ycenter=-2)],
        [(xcenter=0, ycenter=0), (xcenter=3, ycenter= 0), (xcenter=1, ycenter=-2)],
        [(xcenter=0, ycenter=4), (xcenter=3, ycenter= 0), (xcenter=1, ycenter=-2)],
    ],

    combinations_from = [
        [(xcenter=0, ycenter=-3), (xcenter=0, ycenter= 1), (xcenter=3, ycenter=-3)],
        [(xcenter=0, ycenter=-3), (xcenter=0, ycenter= 1), (xcenter=1, ycenter=-5)],
        [(xcenter=0, ycenter=-3), (xcenter=3, ycenter=-3), (xcenter=1, ycenter=-5)],
        [(xcenter=0, ycenter= 1), (xcenter=3, ycenter=-3), (xcenter=1, ycenter=-5)],
    ],

    invariants = [
        5/4 √37/4    3/√8 √37/5
        4/3   4/√5 √8/√5   5/√8
    ],

    matched_triangle_to = [
        (xcenter = 0, ycenter = 0),
        (xcenter = 0, ycenter = 4),
        (xcenter = 3, ycenter = 0),
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
