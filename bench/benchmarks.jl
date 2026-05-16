using Astroalign: _triangle_invariants
using BenchmarkTools
using TypedTables: Table

const SUITE = BenchmarkGroup()
SUITE["core"] = BenchmarkGroup()

SUITE["core"]["_triangle_invariants"] = @benchmarkable _triangle_invariants(phot) setup=(phot = Table(xcenter = rand(Float64, 100), ycenter = rand(Float64, 100)))


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

    using PrettyTables: pretty_table
    # Make pretty table
    pretty_table(data;
        column_labels = ["Benchmark", "Median Time", "Memory", "Allocs"],
        alignment     = [:l, :r, :r, :r]
    )

end