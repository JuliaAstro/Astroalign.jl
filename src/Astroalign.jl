module Astroalign

using Combinatorics: combinations
using CoordinateTransformations: kabsch
using Distances: euclidean
using ImageTransformations: warp
using NearestNeighbors: nn, KDTree
using PSFModels: gaussian, fit
using Photometry:
    estimate_background,
    extract_sources,
    photometry,
    sigma_clip,
    CircularAperture,
    PeakMesh
using TypedTables: Table

export align_frame, get_sources, find_nearest, triangle_invariants

include("utils.jl")
include("findpeaks.jl")
include("register.jl")
include("psf.jl")
include("warp.jl")

end # module
