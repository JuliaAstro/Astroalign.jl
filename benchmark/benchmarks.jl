using Astroalign: _triangle_invariants, _build_correspondences, _photometry, com_psf
using BenchmarkTools
using TypedTables: Table
using PrettyTables: pretty_table

const SUITE = BenchmarkGroup()
SUITE["core"] = BenchmarkGroup()

SUITE["core"]["_triangle_invariants"] = @benchmarkable _triangle_invariants(phot) setup=(phot = Table(xcenter = rand(Float64, 100), ycenter = rand(Float64, 100)))
SUITE["core"]["_build_correspondences"] = @benchmarkable _build_correspondences(C_from, ℳ_from, phot_from, C_to, ℳ_to, phot_to) setup=begin
    phot = Table(xcenter = rand(Float64, 100), ycenter = rand(Float64, 100))
    C_from, ℳ_from = _triangle_invariants(phot)
    C_to, ℳ_to = _triangle_invariants(phot)
    phot_from = phot
    phot_to = phot
end
SUITE["core"]["photometry_com"] = @benchmarkable _photometry(img; opts...) setup = begin
    img = [
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 1 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
    ]
    opts = (;
        box_size = 1,
        ap_radius = 3,
        min_fwhm = 0.1,
        nsigma = 1,
        f = com_psf,
        N_max = 10,
        use_fitpos = false,
   )
end

# If not on CI, we'll show a nice table
if get(ENV, "CI", "false") == "false"
    # Run the benchmarks
    results = run(SUITE, verbose=true)

    # Collect results
    sorted  = sort(collect(results["core"]), by=first)
    names   = [k for (k,_) in sorted]
    trials  = [v for (_,v) in sorted]

    # Pack into matrix
    data = hcat(
        names,
        [BenchmarkTools.prettytime(median(t).time) for t in trials],
        [BenchmarkTools.prettymemory(median(t).memory) for t in trials],
        [median(t).allocs for t in trials]
    )

    # Make pretty table
    pretty_table(data;
        column_labels = ["Benchmark", "Median Time", "Memory", "Allocs"],
        alignment     = [:l, :r, :r, :r]
    )
end
