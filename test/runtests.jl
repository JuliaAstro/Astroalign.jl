using Test
using TestItemRunner

@run_package_tests filter = ti -> occursin("Astroalign/test", ti.filename) verbose = true
