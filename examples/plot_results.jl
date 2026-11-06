# Plot the output of examples/density_plateaus.jl (requires Plots.jl).
#   julia examples/plot_results.jl [results_dir]
# One figure per ε, one row per τ that has been computed. Files named
#   ref_eps<ε>_tau<τ>.dat  (columns x rho u T)  in the same directory are
# overlaid as a reference (e.g. results of another solver).
using Plots, Printf

function readcols(file)
    rows = [parse.(Float64, split(l)) for l in eachline(file)
            if !isempty(strip(l)) && !startswith(strip(l), "#")]
    return permutedims(reduce(hcat, rows))
end

dir = isempty(ARGS) ? joinpath(@__DIR__, "..", "results") : ARGS[1]
for ε in (1.0, 0.1, 0.05)
    taus = [τ for τ in (0.1, 0.01)
            if isfile(joinpath(dir, @sprintf("bgk_eps%g_tau%g_uniform.dat", ε, τ)))]
    isempty(taus) && continue
    plt = plot(layout = (length(taus), 3), size = (1200, 325 * length(taus) + 25))
    for (row, τ) in enumerate(taus)
        tag = @sprintf("eps%g_tau%g", ε, τ)
        B   = readcols(joinpath(dir, "bgk_" * tag * "_uniform.dat"))
        fr  = joinpath(dir, "ref_" * tag * ".dat")
        R   = isfile(fr) ? readcols(fr) : nothing
        for (col, name) in enumerate(("ρ", "u", "T"))
            sp = (row - 1) * 3 + col
            plot!(plt[sp], B[:, 1], B[:, col + 1], label = "BGK (meshfree ALE)",
                  title = "$name,  ε=$ε, τ=$τ", xlabel = "x", legend = :best)
            R === nothing || plot!(plt[sp], R[:, 1], R[:, col + 1],
                                   ls = :dash, label = "reference")
        end
    end
    out = joinpath(dir, @sprintf("fig_eps%g.png", ε))
    savefig(plt, out); println("wrote ", out)
end
