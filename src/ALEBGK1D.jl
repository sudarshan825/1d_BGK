# =============================================================================
#  ALEBGK1D.jl
#
#  Meshfree Arbitrary Lagrangian-Eulerian (ALE) solver for the 1D BGK model
#  in the low-Mach / small-Knudsen scaling
#
#        ∂t F + (1/ε) v ∂x F = 1/(ε² τ) (M_F - F),
#        M_F = ρ_F / sqrt(2π T_F) exp(-(v-u_F)²/(2 T_F)),
#
#  on a periodic interval, following the meshfree ALE method of
#     S. Tiwari, A. Klar, G. Russo, J. Comput. Phys. 458 (2022) 111088.
#
#  ALE form:  dx_i/dt = u_F(x_i)/ε,
#             dF/dt   = -(v - u_F)/ε ∂x F + 1/(ε² τ) (M_F - F).
#
#  Space : weighted least squares (GFDM), Gaussian weight exp(-α d²/h²),
#          quadratic Taylor fit, WENO-type combination of left / central /
#          right stencils for upwinding.
#  Time  : first order or ARS(2,2,1)-type splitting; the BGK relaxation is
#          solved exactly-implicitly.
#  Points: merged when too close, inserted where the spacing is too large;
#          new values by moving least squares (MLS).
# =============================================================================
module ALEBGK1D

using LinearAlgebra, Printf
using Base.Threads

export Params, run_case, step_profile, write_results

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
Base.@kwdef mutable struct Params
    eps::Float64     = 0.1      # scaling parameter ε
    tau::Float64     = 0.01     # relaxation time τ
    L::Float64       = 1.0      # periodic domain [0, L]
    Nx::Int          = 1000     # initial number of particles
    Nv::Int          = 200      # velocity intervals (Nv+1 velocity points)
    vmax::Float64    = 10.0     # velocity cut-off
    cfl::Float64     = 0.5      # Δt = cfl Δx ε / vmax
    dt::Float64      = 0.0      # > 0 overrides the CFL time step
    tfinal::Float64  = 0.1
    order::Int       = 2        # 1: first order splitting, 2: ARS(2,2,1)
    ale::Bool        = true     # false -> Eulerian (fixed points)
    hratio::Float64  = 0.35     # Δx0 = hratio * h
    alpha::Float64   = 6.0      # Gaussian weight parameter
    min_side::Int    = 3        # minimum neighbours on each side
    weno_eps::Float64 = 1e-6    # ε in the smoothness indicators
    gap_max::Float64 = 1.6      # insert particles if spacing > gap_max*Δx0
    gap_min::Float64 = 0.4      # merge particles  if spacing < gap_min*Δx0
    init::Function   = step_profile # (x, ε) -> (ρ, u, T) initial fluctuations
    outdir::String   = "results" # created if it does not exist
    nprint::Int      = 0        # progress every nprint steps (0: ~20 messages)
    write_output::Bool = true
end

# -----------------------------------------------------------------------------
# Velocity grid  (uniform, Nv+1 points, trapezoidal weights)
# -----------------------------------------------------------------------------
struct VGrid
    v::Vector{Float64}
    w::Vector{Float64}
end
function VGrid(Nv::Int, vmax::Float64)
    v  = collect(range(-vmax, vmax; length = Nv + 1))
    dv = 2vmax / Nv
    w  = fill(dv, Nv + 1); w[1] = w[end] = dv / 2
    return VGrid(v, w)
end

# -----------------------------------------------------------------------------
# Moments and Maxwellian
# -----------------------------------------------------------------------------
function moments(f::Matrix{Float64}, vg::VGrid)
    N = size(f, 2)
    ρ = zeros(N); U = zeros(N); T = zeros(N); E2 = zeros(N)
    v, w = vg.v, vg.w
    @threads for i in 1:N
        m0 = 0.0; m1 = 0.0; m2 = 0.0
        @inbounds @simd for k in eachindex(v)
            a = w[k] * f[k, i]
            m0 += a; m1 += a * v[k]; m2 += a * v[k]^2
        end
        ρ[i] = m0; U[i] = m1 / m0; T[i] = m2 / m0 - (m1 / m0)^2; E2[i] = m2
    end
    return ρ, U, T, E2
end

@inline maxwell(ρ, u, T, v) = ρ / sqrt(2π * T) * exp(-(v - u)^2 / (2T))

# f ← (f̃ + a M[f̃]) / (1 + a),   a = Δt_relax / (ε² τ)
function relax!(f::Matrix{Float64}, vg::VGrid, a::Float64)
    ρ, U, T, _ = moments(f, vg)
    c = 1 / (1 + a)
    @threads for i in axes(f, 2)
        @inbounds for k in axes(f, 1)
            f[k, i] = c * (f[k, i] + a * maxwell(ρ[i], U[i], T[i], vg.v[k]))
        end
    end
    return f
end

# -----------------------------------------------------------------------------
# Periodic helpers
# -----------------------------------------------------------------------------
@inline pdist(d, L) = d - L * round(d / L)

"Sort particles by position in [0,L) (cyclic order is preserved by the flow)."
function wrap_and_sort(x::Vector{Float64}, f::Matrix{Float64}, L)
    xw = mod.(x, L)
    pm = sortperm(xw)
    return xw[pm], f[:, pm]
end

# -----------------------------------------------------------------------------
# Weighted least squares: f(x_j) - f(x_i) ≈ d f_x + d²/2 f_xx
#   returns coefficients so that f_x = Σ cx_j (f_j - f_i), f_xx = Σ cxx_j (...)
# -----------------------------------------------------------------------------
function wls_coeffs!(cx, cxx, d, w, m)
    a11 = 0.0; a12 = 0.0; a22 = 0.0
    @inbounds for k in 1:m
        q = 0.5 * d[k]^2
        a11 += w[k] * d[k]^2; a12 += w[k] * d[k] * q; a22 += w[k] * q^2
    end
    det = a11 * a22 - a12^2
    if m >= 2 && det > 1e-12 * a11 * a22
        @inbounds for k in 1:m
            q = 0.5 * d[k]^2
            cx[k]  = ( a22 * w[k] * d[k] - a12 * w[k] * q) / det
            cxx[k] = (-a12 * w[k] * d[k] + a11 * w[k] * q) / det
        end
    elseif m >= 1                       # linear fall-back
        @inbounds for k in 1:m
            cx[k] = w[k] * d[k] / a11; cxx[k] = 0.0
        end
    end
    return nothing
end

# Fixed-width stencil storage: column i holds the neighbours of particle i
struct Stencil
    cnt::Vector{Int}
    idx::Matrix{Int}
    cx::Matrix{Float64}
    cxx::Matrix{Float64}
end
Stencil(K, N) = Stencil(zeros(Int, N), zeros(Int, K, N), zeros(K, N), zeros(K, N))

struct Stencils
    L::Stencil
    C::Stencil
    R::Stencil
end

"""
Neighbour search + WLS coefficients.  `x` must be cyclically ordered.
Neighbours: all particles within h on each side, at least `min_side`
(searching further if needed), at most K per side.
"""
function build_stencils(x::Vector{Float64}, h::Float64, p::Params, dx0::Float64)
    N  = length(x); L = p.L
    # (single assignment: a reassigned variable captured by @threads gets boxed)
    K  = min(ceil(Int, h / (0.25 * p.gap_min * dx0)) + p.min_side + 2, N - 1)
    S  = Stencils(Stencil(K, N), Stencil(2K, N), Stencil(K, N))
    chunks = collect(Iterators.partition(1:N, cld(N, nthreads())))
    @threads for ch in chunks
        dl = zeros(K); dr = zeros(K); dc = zeros(2K)
        wl = zeros(K); wr = zeros(K); wc = zeros(2K)
        il = zeros(Int, K); ir = zeros(Int, K)
        cx = zeros(2K); cxx = zeros(2K)
        for i in ch
            # --- left neighbours
            nl = 0; j = i
            while nl < K
                j = j == 1 ? N : j - 1
                j == i && break
                d = pdist(x[j] - x[i], L)
                d >= 0 && break
                (-d > h && nl >= p.min_side) && break
                nl += 1; il[nl] = j; dl[nl] = d
            end
            # --- right neighbours
            nr = 0; j = i
            while nr < K
                j = j == N ? 1 : j + 1
                j == i && break
                d = pdist(x[j] - x[i], L)
                d <= 0 && break
                (d > h && nr >= p.min_side) && break
                nr += 1; ir[nr] = j; dr[nr] = d
            end
            # --- local smoothing length (enlarged if stencil had to grow)
            dmax = 0.0
            for k in 1:nl; dmax = max(dmax, -dl[k]); end
            for k in 1:nr; dmax = max(dmax,  dr[k]); end
            hi = max(h, 1.0001 * dmax)
            for k in 1:nl; wl[k] = exp(-p.alpha * (dl[k] / hi)^2); end
            for k in 1:nr; wr[k] = exp(-p.alpha * (dr[k] / hi)^2); end
            # --- left stencil
            wls_coeffs!(cx, cxx, dl, wl, nl)
            S.L.cnt[i] = nl
            for k in 1:nl
                S.L.idx[k, i] = il[k]; S.L.cx[k, i] = cx[k]; S.L.cxx[k, i] = cxx[k]
            end
            # --- right stencil
            wls_coeffs!(cx, cxx, dr, wr, nr)
            S.R.cnt[i] = nr
            for k in 1:nr
                S.R.idx[k, i] = ir[k]; S.R.cx[k, i] = cx[k]; S.R.cxx[k, i] = cxx[k]
            end
            # --- central stencil  P_C = P_L ∪ P_R
            nc = nl + nr
            for k in 1:nl; dc[k] = dl[k]; wc[k] = wl[k]; end
            for k in 1:nr; dc[nl+k] = dr[k]; wc[nl+k] = wr[k]; end
            wls_coeffs!(cx, cxx, dc, wc, nc)
            S.C.cnt[i] = nc
            for k in 1:nl
                S.C.idx[k, i] = il[k]; S.C.cx[k, i] = cx[k]; S.C.cxx[k, i] = cxx[k]
            end
            for k in 1:nr
                S.C.idx[nl+k, i] = ir[k]; S.C.cx[nl+k, i] = cx[nl+k]; S.C.cxx[nl+k, i] = cxx[nl+k]
            end
        end
    end
    return S
end

@inline function accumulate!(fx, fxx, S::Stencil, i, f)
    @inbounds for m in 1:S.cnt[i]
        j = S.idx[m, i]; a = S.cx[m, i]; b = S.cxx[m, i]
        @simd for k in axes(f, 1)
            d = f[k, j] - f[k, i]
            fx[k] += a * d; fxx[k] += b * d
        end
    end
end

"""
E[k,i] = -(v_k - s_i)/ε ∂x f(x_i, v_k)   with s_i = u_F(x_i) (ALE) or 0 (Eulerian)
∂x f from the WENO-type least-squares combination of Tiwari-Klar-Russo.
"""
function transport!(E, f, U, S::Stencils, vg::VGrid, p::Params, dx0)
    N  = size(f, 2); nv = size(f, 1)
    dx2 = dx0^2; dx4 = dx0^4; we = p.weno_eps; ε = p.eps
    chunks = collect(Iterators.partition(1:N, cld(N, nthreads())))
    @threads for ch in chunks
        fxL = zeros(nv); fxxL = zeros(nv)
        fxC = zeros(nv); fxxC = zeros(nv)
        fxR = zeros(nv); fxxR = zeros(nv)
        for i in ch
            fill!(fxL, 0); fill!(fxxL, 0); fill!(fxC, 0)
            fill!(fxxC, 0); fill!(fxR, 0); fill!(fxxR, 0)
            accumulate!(fxL, fxxL, S.L, i, f)
            accumulate!(fxC, fxxC, S.C, i, f)
            accumulate!(fxR, fxxR, S.R, i, f)
            s = p.ale ? U[i] : 0.0
            @inbounds for k in 1:nv
                a  = (vg.v[k] - s) / ε
                bC = 0.5 / (fxC[k]^2 * dx2 + fxxC[k]^2 * dx4 + we)^2
                if a > 0
                    bL = 0.5 / (fxL[k]^2 * dx2 + fxxL[k]^2 * dx4 + we)^2
                    fx = (bL * fxL[k] + bC * fxC[k]) / (bL + bC)
                else
                    bR = 0.5 / (fxR[k]^2 * dx2 + fxxR[k]^2 * dx4 + we)^2
                    fx = (bR * fxR[k] + bC * fxC[k]) / (bR + bC)
                end
                E[k, i] = -a * fx
            end
        end
    end
    return E
end

# -----------------------------------------------------------------------------
# Moving least squares interpolation at xq (quadratic basis, Gaussian weight)
# -----------------------------------------------------------------------------
function near_indices(x::Vector{Float64}, xq, h, L, minside)
    N  = length(x)
    i0 = searchsortedfirst(x, mod(xq, L))         # first particle right of xq
    idx = Int[]; ds = Float64[]
    # right side
    j = i0 > N ? 1 : i0; c = 0
    while c < N
        d = pdist(x[j] - xq, L)
        (d < 0 || (d > h && c >= minside)) && break
        push!(idx, j); push!(ds, d); c += 1
        j = j == N ? 1 : j + 1
    end
    # left side
    j = i0 - 1 < 1 ? N : i0 - 1; c2 = 0
    while c2 < N - c
        d = pdist(x[j] - xq, L)
        d >= 0 && break
        (j in idx) && break
        (-d > h && c2 >= minside) && break
        push!(idx, j); push!(ds, d); c2 += 1
        j = j == 1 ? N : j - 1
    end
    return idx, ds
end

"MLS values of all columns of `g` (nv × N) at position xq."
function mls_interp(x::Vector{Float64}, g::AbstractMatrix{Float64}, xq, h, p::Params)
    idx, ds = near_indices(x, xq, h, p.L, p.min_side)
    m  = length(idx)
    hi = max(h, 1.0001 * maximum(abs, ds))
    w  = [exp(-p.alpha * (d / hi)^2) for d in ds]
    nb = m >= 3 ? 3 : (m == 2 ? 2 : 1)
    P  = zeros(m, nb)
    for k in 1:m
        P[k, 1] = 1.0
        nb >= 2 && (P[k, 2] = ds[k])
        nb >= 3 && (P[k, 3] = 0.5 * ds[k]^2)
    end
    A  = P' * Diagonal(w) * P
    c  = (A \ (P' * Diagonal(w)))[1, :]         # weights of the constant term
    out = zeros(size(g, 1))
    for k in 1:m
        @views out .+= c[k] .* g[:, idx[k]]
    end
    return out
end

# -----------------------------------------------------------------------------
# Particle management (merging of clusters, filling of holes)
# -----------------------------------------------------------------------------
function manage_particles(x::Vector{Float64}, f::Matrix{Float64}, p::Params, dx0, h)
    L = p.L
    x, f = wrap_and_sort(x, f, L)
    N = length(x)
    # ---- merge pairs that are too close
    used = falses(N); newx = Float64[]; newf = Vector{Vector{Float64}}()
    nmerge = 0
    for i in 1:N
        j = i == N ? 1 : i + 1
        gap = mod(x[j] - x[i], L)
        if gap < p.gap_min * dx0 && !used[i] && !used[j] && N > 4
            xm = mod(x[i] + gap / 2, L)
            push!(newx, xm); push!(newf, mls_interp(x, f, xm, h, p))
            used[i] = used[j] = true; nmerge += 1
        end
    end
    if nmerge > 0
        keep = findall(!, used)
        x = vcat(x[keep], newx)
        f = hcat(f[:, keep], reduce(hcat, newf))
        x, f = wrap_and_sort(x, f, L)
        N = length(x)
    end
    # ---- fill holes
    newx = Float64[]; newf = Vector{Vector{Float64}}()
    for i in 1:N
        j = i == N ? 1 : i + 1
        gap = mod(x[j] - x[i], L)
        if gap > p.gap_max * dx0
            nins = max(1, round(Int, gap / dx0) - 1)
            for q in 1:nins
                xq = mod(x[i] + q * gap / (nins + 1), L)
                push!(newx, xq); push!(newf, mls_interp(x, f, xq, h, p))
            end
        end
    end
    if !isempty(newx)
        x = vcat(x, newx)
        f = hcat(f, reduce(hcat, newf))
        x, f = wrap_and_sort(x, f, L)
    end
    return x, f, nmerge, length(newx)
end

# -----------------------------------------------------------------------------
# Time stepping
# -----------------------------------------------------------------------------
"First order splitting: explicit transport, exact implicit relaxation."
function step1(x, f, dt, p, vg, dx0, h)
    _, U, _, _ = moments(f, vg)
    S  = build_stencils(x, h, p, dx0)
    E  = similar(f); transport!(E, f, U, S, vg, p, dx0)
    ft = f .+ dt .* E
    xn = p.ale ? x .+ dt .* U ./ p.eps : copy(x)
    relax!(ft, vg, dt / (p.eps^2 * p.tau))
    return xn, ft
end

"Second order ARS(2,2,1)-type splitting with mid-point transport."
function step2(x, f, dt, p, vg, dx0, h)
    a_ε = p.eps^2 * p.tau
    # stage 1: half step
    _, U, _, _ = moments(f, vg)
    S  = build_stencils(x, h, p, dx0)
    E  = similar(f); transport!(E, f, U, S, vg, p, dx0)
    fh = f .+ (dt / 2) .* E
    xh = p.ale ? x .+ (dt / 2) .* U ./ p.eps : copy(x)
    relax!(fh, vg, dt / (2a_ε))
    # stage 2: full step with mid-point transport
    _, Uh, _, _ = moments(fh, vg)
    S  = build_stencils(xh, h, p, dx0)
    transport!(E, fh, Uh, S, vg, p, dx0)
    fn = f .+ dt .* E
    xn = p.ale ? x .+ dt .* Uh ./ p.eps : copy(x)
    relax!(fn, vg, dt / a_ε)
    return xn, fn
end

# -----------------------------------------------------------------------------
# Initial data
# -----------------------------------------------------------------------------
"""
    step_profile(x, ε) -> (ρ, u, T)

Default initial fluctuations: density plateaus 1 ± ε² on (0.2,0.3] and
(0.7,0.8] of the unit interval, 1 elsewhere; u = T = 0.
"""
function step_profile(x, ε)
    ρ = if x <= 0.2;     1.0
        elseif x <= 0.3; 1.0 + ε^2
        elseif x <= 0.7; 1.0
        elseif x <= 0.8; 1.0 - ε^2
        else             1.0
        end
    return ρ, 0.0, 0.0
end

"""
Maxwellian initial data from the fluctuations (ρ, u, T) returned by `p.init`:
ρ_F = 1 + ερ,  ρ_F u_F = εu,  T_F = 1 + εT/ρ_F - ε²u²/ρ_F².
"""
function init_f(x, vg, p)
    ε = p.eps
    f = zeros(length(vg.v), length(x))
    for i in eachindex(x)
        ρ, u, T = p.init(x[i], ε)
        ρF = 1 + ε * ρ
        uF = ε * u / ρF
        TF = 1 + ε * T / ρF - ε^2 * u^2 / ρF^2
        TF > 0 || error("initial temperature T_F <= 0 at x = $(x[i])")
        for k in eachindex(vg.v)
            f[k, i] = maxwell(ρF, uF, TF, vg.v[k])
        end
    end
    return f
end

"""
Fluctuations around the global equilibrium (ρ_F, u_F, T_F) = (1, 0, 1):
ρ = (ρ_F-1)/ε,  u = ρ_F u_F/ε,  T = (∫v²F dv - 1)/ε - ρ.
"""
function fluctuations(f, vg, ε)
    ρF, uF, TF, E2 = moments(f, vg)
    ρ = (ρF .- 1) ./ ε
    u = ρF .* uF ./ ε
    T = (E2 .- 1) ./ ε .- ρ
    return ρ, u, T, ρF, uF, TF
end

"Total mass ∫ρ_F dx with the periodic trapezoidal rule on the particles."
function total_mass(x, ρF, L)
    N = length(x); m = 0.0
    for i in 1:N
        j = i == N ? 1 : i + 1
        m += 0.5 * (ρF[i] + ρF[j]) * mod(x[j] - x[i], L)
    end
    return m
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
function run_case(p::Params; verbose = true)
    vg  = VGrid(p.Nv, p.vmax)
    dx0 = p.L / p.Nx
    h   = dx0 / p.hratio
    x   = [(i - 0.5) * dx0 for i in 1:p.Nx]
    f   = init_f(x, vg, p)
    dt  = p.dt > 0 ? p.dt : p.cfl * dx0 * p.eps / p.vmax
    nst = ceil(Int, p.tfinal / dt - 1e-9); dt = p.tfinal / nst
    np  = p.nprint > 0 ? p.nprint : max(1, nst ÷ 20)

    m0 = total_mass(x, moments(f, vg)[1], p.L)
    verbose && @printf("ε=%g τ=%g  Nx=%d Nv=%d  dt=%.3e  steps=%d  threads=%d  %s\n",
                       p.eps, p.tau, p.Nx, p.Nv, dt, nst, nthreads(),
                       p.ale ? "ALE" : "Eulerian")
    t = 0.0; nm = 0; ni = 0; tstart = time()
    for n in 1:nst
        x, f = p.order == 1 ? step1(x, f, dt, p, vg, dx0, h) :
                              step2(x, f, dt, p, vg, dx0, h)
        t += dt
        if p.ale
            x, f, a, b = manage_particles(x, f, p, dx0, h)
            nm += a; ni += b
        end
        if verbose && (n % np == 0 || n == nst)
            ρ, u, T, ρF, _, _ = fluctuations(f, vg, p.eps)
            @printf("  step %7d  t=%.4f  N=%5d  max|u|=%.3e  max|T|=%.3e  mass err=%.2e  (%.1fs)\n",
                    n, t, length(x), maximum(abs, u), maximum(abs, T),
                    total_mass(sort(mod.(x, p.L)), ρF[sortperm(mod.(x, p.L))], p.L) - m0,
                    time() - tstart)
        end
    end
    x, f = wrap_and_sort(x, f, p.L)
    ρ, u, T, ρF, uF, TF = fluctuations(f, vg, p.eps)
    verbose && @printf("  merged %d, inserted %d particles\n", nm, ni)

    # resample macroscopic fields to a uniform grid (for plotting/comparison)
    xu = [(i - 0.5) * dx0 for i in 1:p.Nx]
    G  = permutedims(hcat(ρ, u, T, ρF, uF, TF))          # 6 × N
    Gu = reduce(hcat, [mls_interp(x, G, xq, h, p) for xq in xu])

    res = (x = x, rho = ρ, u = u, T = T, rhoF = ρF, uF = uF, TF = TF,
           xu = xu, rho_u = Gu[1, :], u_u = Gu[2, :], T_u = Gu[3, :],
           rhoF_u = Gu[4, :], uF_u = Gu[5, :], TF_u = Gu[6, :], f = f, v = vg.v)

    p.write_output && write_results(res, p; verbose = verbose)
    return res
end

"""
    write_results(res, p; verbose = true)

Write `bgk_eps<ε>_tau<τ>_particles.dat` (values at the particles) and
`bgk_eps<ε>_tau<τ>_uniform.dat` (MLS-resampled to Nx cell centres) to
`p.outdir`. Columns: x  ρ  u  T  ρ_F  u_F  T_F.
"""
function write_results(res, p::Params; verbose = true)
    mkpath(p.outdir)
    tag = @sprintf("bgk_eps%g_tau%g", p.eps, p.tau)
    hdr = "# x rho u T rho_F u_F T_F   (eps=$(p.eps), tau=$(p.tau), t=$(p.tfinal))"
    function dump(file, cols)
        open(joinpath(p.outdir, file), "w") do io
            println(io, hdr)
            for i in eachindex(cols[1])
                println(io, join((@sprintf("%.15e", c[i]) for c in cols), " "))
            end
        end
    end
    dump(tag * "_particles.dat", (res.x, res.rho, res.u, res.T, res.rhoF, res.uF, res.TF))
    dump(tag * "_uniform.dat",   (res.xu, res.rho_u, res.u_u, res.T_u,
                                  res.rhoF_u, res.uF_u, res.TF_u))
    verbose && println("  wrote ", joinpath(p.outdir, tag * "_*.dat"))
    return nothing
end

end # module
