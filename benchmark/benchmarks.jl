using Astroalign: _triangle_invariants, _build_correspondences, com_psf
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

function gaussian(T, px, py; x, y, fwhm, amp)
    σ = fwhm / (2 * sqrt(2 * log(2)))
    return convert(T, amp * exp(-((px - x)^2 + (py - y)^2) / (2 * σ^2)))
end
model(T, x, y, amp) = gaussian(T, 4, 4; x, y, fwhm = 3, amp)
const T = Float32
const x = 1:20
const y = 1:20
SUITE["core"]["com_psf"] = @benchmarkable com_psf(data; rel_thresh) setup = begin
    data = model.(T, x, y', 10) .+ T(0.1) * randn(T, length(x), length(y))
    rel_thresh = 0.1f0
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
