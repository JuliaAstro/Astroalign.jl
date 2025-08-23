using Test
using TestItemRunner

@run_package_tests filter = ti -> occursin("Astroalign.jl/test", ti.filename) verbose = true
