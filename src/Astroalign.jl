module Astroalign

using Combinatorics: combinations
using CoordinateTransformations: kabsch
using Distances: euclidean
using StaticArrays: SVector
using ImageTransformations: warp, AffineMap, Translation, compose
using NearestNeighbors: nn, KDTree
using PSFModels: gaussian, fit
using Photometry: estimate_background,
                  extract_sources,
                  photometry,
                  sigma_clip,
                  CircularAperture,
                  PeakMesh
using NDTools: select_region, select_region_view, reorient

export get_sources, find_nearest, triangle_invariants
export align_frame, stack_many, stack_many_drizzle
export correct_dark_flat, bin_rgb, bin_mono

include("utils.jl")
include("findpeaks.jl")
include("register.jl")
include("warp.jl")
include("stacker.jl")


end # module
