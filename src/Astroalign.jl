module Astroalign

using Combinatorics: combinations, permutations
using CoordinateTransformations: kabsch
using Distances: euclidean
using ImageTransformations: compose, warp, AffineMap, Translation
using NearestNeighbors: nn, KDTree
using PSFModels: gaussian, fit
using Photometry: estimate_background, extract_sources, photometry, sigma_clip, CircularAperture, PeakMesh
using StaticArrays: SVector
using LinearAlgebra: norm, triu
using Random: randperm
using Statistics: median

export align_frame, find_nearest, get_sources, stack_many, triangle_invariants
export com_psf, collect_info

include("utils.jl")
include("findpeaks.jl")
include("register.jl")
include("warp.jl")
include("stacker.jl")

end # module
