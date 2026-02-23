module Astroalign

using Combinatorics: combinations
using CoordinateTransformations: kabsch, SVector
using Distances: euclidean
using ImageTransformations: warp, AffineMap, Translation, compose
using NearestNeighbors: nn, KDTree
using PSFModels: gaussian, fit
using Photometry: estimate_background,
                  extract_sources,
                  photometry,
                  sigma_clip,
                  CircularAperture,
                  PeakMesh

export get_sources, find_nearest, triangle_invariants
export align_frame, stack_many, stack_many_drizzle

include("utils.jl")
include("findpeaks.jl")
include("register.jl")
include("warp.jl")
include("stacker.jl")


end # module
