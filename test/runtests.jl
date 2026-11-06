using ALEBGK1D, Test

# small, fast configurations
small(; kw...) = Params(; Nx = 100, Nv = 30, tfinal = 0.02, write_output = false, kw...)

@testset "ALEBGK1D" begin
    @testset "global equilibrium stays at rest" begin
        res = run_case(small(eps = 1.0, tau = 0.1, init = (x, ε) -> (0.0, 0.0, 0.0));
                       verbose = false)
        @test maximum(abs, res.rho) < 1e-8
        @test maximum(abs, res.u)   < 1e-8
    end

    @testset "low Mach (ε = 0.1): symmetry and mass" begin
        res = run_case(small(eps = 0.1, tau = 0.01); verbose = false)
        ρ, u = res.rho_u, res.u_u
        @test all(isfinite, res.f)
        # initial data antisymmetric about x = 0.5 → ρ-1 antisymmetric, u symmetric
        @test maximum(abs, (ρ .- 1) .+ (reverse(ρ) .- 1)) < 1e-3 * (maximum(ρ) - minimum(ρ))
        @test maximum(abs, u .- reverse(u)) < 1e-2 * maximum(abs, u)
        # mass ∫ρ_F dx = 1 + ε
        @test abs(sum(res.rhoF_u) / length(ρ) - 1.1) < 1e-6
    end

    @testset "ALE agrees with Eulerian (ε = 1)" begin
        a = run_case(small(eps = 1.0, tau = 0.1, ale = true);  verbose = false)
        e = run_case(small(eps = 1.0, tau = 0.1, ale = false); verbose = false)
        @test all(isfinite, a.f)
        @test maximum(abs, a.u_u .- e.u_u) < 0.05 * maximum(abs, e.u_u)
    end
end
