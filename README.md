# ALEBGK1D.jl

A meshfree **Arbitrary Lagrangian–Eulerian (ALE)** solver for the one-dimensional
BGK model of the Boltzmann equation in the low-Mach / small-Knudsen scaling,
written in pure Julia (standard library only).

The method follows

> S. Tiwari, A. Klar, G. Russo, *A meshfree arbitrary Lagrangian–Eulerian method
> for the BGK model of the Boltzmann equation with moving boundaries*,
> J. Comput. Phys. **458** (2022) 111088.

## Model

On a periodic interval the code solves the scaled BGK equation

$$
\partial_t F + \frac{1}{\varepsilon}\, v\,\partial_x F = \frac{1}{\varepsilon^2\tau}\,(M_F - F),
\qquad
M_F = \frac{\rho_F}{\sqrt{2\pi T_F}}\exp\!\Big(-\frac{(v-u_F)^2}{2T_F}\Big),
$$

where $\varepsilon$ is the scaling parameter (Knudsen ∼ Mach number) and $\tau$ the
relaxation time. Macroscopic quantities are the moments
$\rho_F=\int F\,dv$, $\rho_F u_F=\int vF\,dv$, $\rho_F u_F^2+\rho_F T_F=\int v^2F\,dv$.

Initial data and results are expressed as fluctuations around the global
equilibrium $(\rho_F,u_F,T_F)=(1,0,1)$:

$$
\rho_F = 1+\varepsilon\rho,\qquad \rho_F u_F = \varepsilon u,\qquad
T = \frac{1}{\varepsilon}\Big(\int v^2F\,dv-1\Big)-\rho .
$$

## Method

| Ingredient | Choice |
|---|---|
| Frame | Points move with the gas: $\dot x_i = u_F(x_i)/\varepsilon$; along the paths $\dot F = -\frac{v-u_F}{\varepsilon}\partial_xF + \frac{1}{\varepsilon^2\tau}(M_F-F)$ |
| Derivatives | Weighted least squares (GFDM), Gaussian weight $e^{-\alpha d^2/h^2}$ with $\alpha=6$, $\Delta x = 0.35\,h$, quadratic Taylor fit |
| Upwinding | WENO-type combination of left, central and right least-squares stencils, linear weights $(\tfrac12,\tfrac12,0)$ for $v>u_F$ and $(0,\tfrac12,\tfrac12)$ otherwise |
| Time | First-order splitting or second-order ARS(2,2,1)-type splitting; the relaxation step is solved exactly-implicitly, so $\Delta t$ does not depend on $\tau$ |
| Point management | Pairs closer than `gap_min·Δx₀` are merged, gaps larger than `gap_max·Δx₀` are filled; new values by moving least squares |
| Velocity space | Uniform grid of `Nv+1` points on `[-vmax, vmax]`, trapezoidal quadrature |

Loops over particles are multithreaded.

## Installation

```bash
git clone https://github.com/sudarshan825/1d_BGK.git
cd 1d_BGK
julia --project -e 'using Pkg; Pkg.instantiate()'
```

Only the Julia standard library is needed for the solver; the plotting example
uses [Plots.jl](https://github.com/JuliaPlots/Plots.jl).

## Quick start

Start Julia in the project folder with `julia --project -t auto`, then

```julia
using ALEBGK1D
res = run_case(Params(eps = 0.1, tau = 0.01, tfinal = 0.1))
```

`res` is a named tuple with the particle positions `x`, the fluctuations
`rho, u, T`, the kinetic fields `rhoF, uF, TF`, the same fields resampled to a
uniform grid (`xu, rho_u, u_u, T_u, ...`), and the distribution `f` on the
velocity grid `v`.

### Your own initial data

Pass a function `(x, ε) -> (ρ, u, T)` returning the initial fluctuations:

```julia
wave(x, ε) = (0.0, sin(2π * x), 0.0)
run_case(Params(eps = 0.05, tau = 0.01, init = wave))
```

### Example: periodic density plateaus

`examples/density_plateaus.jl` starts from
$\rho = 1 \pm \varepsilon^2$ on $(0.2,0.3]$ and $(0.7,0.8]$, $\rho = 1$ elsewhere,
$u = T = 0$, and runs $\varepsilon\in\{1,0.1,0.05\}$, $\tau\in\{0.1,0.01\}$ up to $t=0.1$:

```bash
julia -t auto examples/density_plateaus.jl              # all six cases
julia -t auto examples/density_plateaus.jl 0.1 0.01     # one case: ε τ [Nx Nv]
julia examples/plot_results.jl                          # figures in results/
```

To overlay results from another solver, put files `ref_eps<ε>_tau<τ>.dat`
(columns `x rho u T`) in `results/`.

## Parameters

| Field | Default | Meaning |
|---|---|---|
| `eps` | `0.1` | scaling parameter ε |
| `tau` | `0.01` | relaxation time τ |
| `L` | `1.0` | length of the periodic domain |
| `Nx` | `1000` | initial number of particles |
| `Nv` | `200` | velocity intervals (`Nv+1` points) |
| `vmax` | `10.0` | velocity cut-off |
| `cfl` | `0.5` | time step `Δt = cfl·Δx·ε/vmax` |
| `dt` | `0.0` | fixed time step if `> 0` |
| `tfinal` | `0.1` | final time |
| `order` | `2` | `1`: first-order splitting, `2`: ARS(2,2,1) |
| `ale` | `true` | `false` keeps the points fixed (Eulerian) |
| `hratio` | `0.35` | `Δx₀ = hratio·h` |
| `alpha` | `6.0` | Gaussian weight parameter |
| `min_side` | `3` | minimum neighbours on each side |
| `weno_eps` | `1e-6` | regularisation in the smoothness indicators |
| `gap_min`, `gap_max` | `0.4`, `1.6` | merge / insert thresholds (× `Δx₀`) |
| `init` | `step_profile` | initial fluctuations `(x, ε) -> (ρ, u, T)` |
| `outdir` | `"results"` | output folder (created if missing) |
| `write_output` | `true` | write `.dat` files |
| `nprint` | `0` | progress interval in steps (`0`: about 20 messages) |

## Output

For each run two text files are written to `outdir`:

- `bgk_eps<ε>_tau<τ>_particles.dat` – values at the particles,
- `bgk_eps<ε>_tau<τ>_uniform.dat` – values resampled to `Nx` cell centres,

with columns `x  rho  u  T  rho_F  u_F  T_F`.

## Cost

The explicit transport step requires $\Delta t \propto \varepsilon\,\Delta x / v_{\max}$,
so the run time grows like $1/\varepsilon$ and linearly with `Nv`. Start Julia with
`-t auto` to use all cores; for quick tests use e.g. `Nx = 200, Nv = 40`.

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## Citation

If you use this code, please cite

```bibtex
@article{TiwariKlarRusso2022,
  author  = {Tiwari, Sudarshan and Klar, Axel and Russo, Giovanni},
  title   = {A meshfree arbitrary {L}agrangian--{E}ulerian method for the {BGK}
             model of the {B}oltzmann equation with moving boundaries},
  journal = {Journal of Computational Physics},
  volume  = {458},
  pages   = {111088},
  year    = {2022}
}
```

## License

MIT – see [LICENSE](LICENSE).
