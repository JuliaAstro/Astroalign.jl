using Test
using TestItemRunner

@run_package_tests filter = ti -> endswith(ti.filename, "test_functions.jl") verbose = true
