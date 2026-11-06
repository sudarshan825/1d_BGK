# Periodic density-perturbation test on [0, 1]:
#   ρ(0,x) = 1 + ε² on (0.2,0.3],  1 - ε² on (0.7,0.8],  1 elsewhere;  u = T = 0.
# Runs ε ∈ {1, 0.1, 0.05} × τ ∈ {0.1, 0.01} up to t = 0.1.
#
#   julia -t auto examples/density_plateaus.jl                 # all six cases
#   julia -t auto examples/density_plateaus.jl 0.1 0.01        # one case: ε τ
#   julia -t auto examples/density_plateaus.jl 0.1 0.01 1000 200   # ε τ Nx Nv
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)
using ALEBGK1D

outdir = joinpath(@__DIR__, "..", "results")

if isempty(ARGS)
    for ε in (1.0, 0.1, 0.05), τ in (0.1, 0.01)
        run_case(Params(eps = ε, tau = τ, tfinal = 0.1, outdir = outdir))
    end
else
    ε  = parse(Float64, ARGS[1])
    τ  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.01
    Nx = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1000
    Nv = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 200
    run_case(Params(eps = ε, tau = τ, Nx = Nx, Nv = Nv, tfinal = 0.1, outdir = outdir))
end
