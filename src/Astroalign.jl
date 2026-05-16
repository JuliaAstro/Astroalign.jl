module Astroalign

using ConsensusFitting: ransac
using CoordinateTransformations: kabsch, AffineMap
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

export align_frames, find_transform, apply_transform

include("findpeaks.jl")
include("register.jl")
include("warp.jl")

end # module
