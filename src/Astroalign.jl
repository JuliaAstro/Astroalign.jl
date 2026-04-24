module Astroalign

using Combinatorics: combinations
using ConsensusFitting: ransac
using CoordinateTransformations: kabsch, AffineMap
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

export align_frame, find_transform

include("psf.jl")
include("findpeaks.jl")
include("register.jl")
include("warp.jl")

end # module
