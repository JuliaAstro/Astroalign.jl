module Astroalign

using Combinatorics: combinations
using CoordinateTransformations: kabsch
using Distances: euclidean
using ImageTransformations: compose, warp, AffineMap, Translation
using NearestNeighbors: nn, KDTree
using PSFModels: gaussian, fit
using Photometry: estimate_background, extract_sources, photometry, sigma_clip, CircularAperture, PeakMesh
using StaticArrays: SVector

export align_frame, find_nearest, get_sources, stack_many, stack_many_drizzle, triangle_invariants

include("utils.jl")
include("findpeaks.jl")
include("register.jl")
include("warp.jl")
include("stacker.jl")

end # module
