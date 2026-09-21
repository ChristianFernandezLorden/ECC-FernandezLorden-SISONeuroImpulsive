# =====================================================================
#  Minimum sampling frequency for a sampled static output feedback loop
# =====================================================================
#
#  Plant (continuous time):      ẋ = A x + B u,   y = C x
#  Controller (sampled + ZOH):   u(t) = sgn * K * y(t_k),  t ∈ [t_k, t_k + h)
#                                t_k  = k*h
#
#  Between two samples the plant runs in continuous time with a frozen
#  input, so the exact state map over one period is
#
#       x(t_{k+1}) = M(h) x(t_k),
#       M(h) = Φ(h) + sgn * Γ(h) K C,
#       Φ(h) = exp(A h),   Γ(h) = ∫₀ʰ exp(A s) ds ⋅ B
#
#  The loop is (exponentially) stable iff ρ(M(h)) < 1.  Because Φ and Γ
#  are the *exact* flow over [0,h], this is not an approximation: it is
#  the continuous-time system, sampled.  Inter-sample behaviour adds no
#  extra condition — over a finite h the transition is bounded, so
#  stability of the sampled sequence implies stability of x(t) for all t.
#
#  This file finds h_crit = the first h > 0 at which ρ(M(h)) reaches 1,
#  and reports f_min = 1/h_crit.
#
#  Only LinearAlgebra + Printf are required (Plots is optional).
# =====================================================================

using LinearAlgebra
using Printf

# ---------------------------------------------------------------------
# Exact ZOH discretisation
# ---------------------------------------------------------------------

"""
    zoh_matrices(A, B, h) -> (Φ, Γ)

Exact zero-order-hold matrices over one period `h`:

    Φ = exp(A h),    Γ = ∫₀ʰ exp(A s) ds ⋅ B

Uses the Van Loan block-matrix trick

    exp( [A B; 0 0] h ) = [Φ Γ; 0 I]

so it stays valid when `A` is singular (no `A⁻¹(Φ-I)B` formula needed).
"""
function zoh_matrices(A::AbstractMatrix, B::AbstractMatrix, h::Real)
    n, m = size(A, 1), size(B, 2)
    size(A, 2) == n || throw(DimensionMismatch("A must be square"))
    size(B, 1) == n || throw(DimensionMismatch("B must have size(A,1) rows"))
    T = promote_type(float(eltype(A)), float(eltype(B)), typeof(float(h)))
    M = zeros(T, n + m, n + m)
    M[1:n, 1:n] .= A
    M[1:n, (n+1):end] .= B
    E = exp(M .* T(h))
    return E[1:n, 1:n], E[1:n, (n+1):end]
end

"""
    closed_loop_map(A, B, C, K, h; sgn = -1) -> M(h)

One-period transition matrix of the sampled-data loop,
`M(h) = Φ(h) + sgn * Γ(h) K C`.

`sgn = -1` for the negative-feedback convention `u = -K y` (default),
`sgn = +1` for `u = +K y`.
"""
function closed_loop_map(A, B, C, K, h::Real; sgn::Int=-1)
    Φ, Γ = zoh_matrices(A, B, h)
    return Φ + sgn * Γ * K * C
end

spectral_radius(M) = maximum(abs, eigvals(Matrix(M)))

"""
    rho(A, B, C, K, h; sgn = -1)

Spectral radius of the one-period map. `< 1` ⟺ the sampled loop is stable
for that sampling period.
"""
rho(A, B, C, K, h::Real; sgn::Int=-1) =
    spectral_radius(closed_loop_map(A, B, C, K, h; sgn=sgn))

# ---------------------------------------------------------------------
# Search for the critical sampling period
# ---------------------------------------------------------------------

"""
    continuous_margin(A, B, C, K; sgn = -1) -> (max Re λ, λ)

Spectral abscissa of the ideal (continuous-time) closed loop `A + sgn*B*K*C`.
Must be negative for any sampling period to work.
"""
function continuous_margin(A, B, C, K; sgn::Int=-1)
    λ = eigvals(Matrix(A + sgn * B * K * C))
    return maximum(real, λ), λ
end

"""
    default_horizon(A, B, C, K; sgn = -1, factor = 50)

Heuristic upper bound for the search in `h`, scaled by the fastest mode of
the open and closed loop. Override with `hmax` if the search reports that it
found no instability.
"""
function default_horizon(A, B, C, K; sgn::Int=-1, factor::Real=50)
    λ = vcat(eigvals(Matrix(A)), eigvals(Matrix(A + sgn * B * K * C)))
    s = maximum(abs, λ)
    (isfinite(s) && s > 0) || (s = one(float(s)))
    return factor / s
end

"""
    max_sampling_period(A, B, C, K; sgn=-1, hmax=nothing, ngrid=2000,
                        margin=0.0, rtol=1e-12) -> h_crit

Largest sampling period such that the sampled-data loop is stable for
**every** `h` in `(0, h_crit)`, i.e. the first zero of `ρ(M(h)) - (1-margin)`.

The search brackets the crossing on a uniform grid of `ngrid` points over
`(0, hmax]`, then bisects. Set `margin > 0` (e.g. `0.02`) to require a
guaranteed decay rate `ρ ≤ 1 - margin` instead of marginal stability.

Warning: `ρ(M(h))` is not monotone in `h`. A grid that is too coarse can step
over a narrow unstable pocket. Increase `ngrid`, or call `stability_windows`
to see the whole picture.
"""
function max_sampling_period(A, B, C, K; sgn::Int=-1, hmax=nothing,
    ngrid::Int=2000, margin::Real=0.0,
    rtol::Real=1e-12)

    σ, _ = continuous_margin(A, B, C, K; sgn=sgn)
    σ < 0 || error("A + sgn*B*K*C is not Hurwitz (max Re λ = $σ). " *
                   "K does not stabilise the continuous loop, so no sampling " *
                   "period can work. Check K and the sign convention `sgn`.")

    H = hmax === nothing ? default_horizon(A, B, C, K; sgn=sgn) : float(hmax)
    ρ_target = 1 - margin
    f(h) = rho(A, B, C, K, h; sgn=sgn) - ρ_target

    # Find a small h that is certainly stable (ρ → 1⁻ as h → 0⁺).
    h0 = H / ngrid
    for _ in 1:200
        f(h0) < 0 && break
        h0 /= 2
    end
    f(h0) < 0 || error("Could not find any stable sampling period. " *
                       "With margin=$margin the requirement may be infeasible.")

    hlo = h0
    for h in range(h0, H; length=ngrid)
        if f(h) ≥ 0
            a, b = hlo, h                      # f(a) < 0 ≤ f(b)
            while b - a > rtol * max(1.0, b)
                m = 0.5 * (a + b)
                f(m) < 0 ? (a = m) : (b = m)
            end
            return 0.5 * (a + b)
        end
        hlo = h
    end

    @warn "No loss of stability found for h ≤ $H. Increase `hmax`."
    return Inf
end

"""
    min_sampling_frequency(A, B, C, K; verbose=true, kwargs...)

Convenience wrapper. Returns a named tuple

    (h_max, f_min, ω_min, rho_at_h_max)

with `f_min = 1/h_max` [Hz] and `ω_min = 2π/h_max` [rad/s].
All keyword arguments of `max_sampling_period` are forwarded.
"""
function min_sampling_frequency(A, B, C, K; sgn::Int=-1, verbose::Bool=true,
    kwargs...)
    h = max_sampling_period(A, B, C, K; sgn=sgn, kwargs...)
    f = 1 / h
    ω = 2π / h
    if verbose
        σ, λc = continuous_margin(A, B, C, K; sgn=sgn)
        @printf("continuous closed loop: max Re λ = %.6g\n", σ)
        @printf("  eigenvalues: %s\n", string(round.(λc, digits=6)))
        @printf("critical sampling period  h_max = %.10g s\n", h)
        @printf("minimum sampling frequency f_min = %.10g Hz  (%.10g rad/s)\n", f, ω)
        @printf("check: ρ(M(0.99 h_max)) = %.6f,  ρ(M(1.01 h_max)) = %.6f\n",
            rho(A, B, C, K, 0.99h; sgn=sgn),
            rho(A, B, C, K, 1.01h; sgn=sgn))
    end
    return (h_max=h, f_min=f, ω_min=ω,
        rho_at_h_max=rho(A, B, C, K, h; sgn=sgn))
end

# ---------------------------------------------------------------------
# Full picture: every stable window, and the ρ(h) curve for plotting
# ---------------------------------------------------------------------

"""
    rho_curve(A, B, C, K; sgn=-1, hmax=nothing, ngrid=2000) -> (hs, ρs)

Sampled `ρ(M(h))` over `(0, hmax]`. Handy for plotting:

    hs, rs = rho_curve(A, B, C, K)
    using Plots
    plot(hs, rs, xlabel="h [s]", ylabel="ρ(M(h))", yscale=:log10, label="ρ")
    hline!([1.0], ls=:dash, label="stability limit")
"""
function rho_curve(A, B, C, K; sgn::Int=-1, hmax=nothing, ngrid::Int=2000)
    H = hmax === nothing ? default_horizon(A, B, C, K; sgn=sgn) : float(hmax)
    hs = collect(range(H / ngrid, H; length=ngrid))
    rs = [rho(A, B, C, K, h; sgn=sgn) for h in hs]
    return hs, rs
end

"""
    stability_windows(A, B, C, K; sgn=-1, hmax=nothing, ngrid=4000, rtol=1e-12)

All intervals of `h` in `(0, hmax]` where `ρ(M(h)) < 1`, with the endpoints
refined by bisection. Returns a vector of `(hlow, hhigh)` tuples.

Useful because sampled-data loops can regain stability at larger `h`: only the
first window matters for a "minimum sampling frequency", but the rest tells you
whether the answer is a hard cliff or a narrow ledge.
"""
function stability_windows(A, B, C, K; sgn::Int=-1, hmax=nothing,
    ngrid::Int=4000, rtol::Real=1e-12)
    H = hmax === nothing ? default_horizon(A, B, C, K; sgn=sgn) : float(hmax)
    hs = collect(range(H / ngrid, H; length=ngrid))
    f(h) = rho(A, B, C, K, h; sgn=sgn) - 1
    stable = [f(h) < 0 for h in hs]

    refine(a, b) = begin                       # sign change bracketed on [a,b]
        fa = f(a)
        while b - a > rtol * max(1.0, b)
            m = 0.5 * (a + b)
            (f(m) < 0) == (fa < 0) ? (a = m) : (b = m)
        end
        0.5 * (a + b)
    end

    windows = Tuple{Float64,Float64}[]
    k = 1
    while k ≤ length(hs)
        if stable[k]
            i = k
            while k ≤ length(hs) && stable[k]
                ;
                k += 1;
            end
            lo = i == 1 ? 0.0 : refine(hs[i-1], hs[i])
            hi = k > length(hs) ? hs[end] : refine(hs[k-1], hs[k])
            push!(windows, (lo, hi))
        else
            k += 1
        end
    end
    return windows
end

# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

function demo()
    # Unstable second-order plant, single measured output y = x₁ + x₂
    A = [0.0 1.0;
        1.0 0.0]
    B = reshape([0.0, 1.0], 2, 1)
    C = [1.0 1.0]
    K = reshape([2.0], 1, 1)          # u = -K y ⇒ A-BKC has a double pole at -1

    println("── demo: ẋ = Ax + Bu, y = Cx, u = -K y (ZOH) ──")
    res = min_sampling_frequency(A, B, C, K; sgn=-1)
    println("\nstable windows in h: ", stability_windows(A, B, C, K; sgn=-1))
    @printf("\n(closed form for this example: h_max = ln 3 = %.10g)\n", log(3))
    return res
end

# Uncomment to run:
# demo()

function simulate_zoh_from_starting_point(A::Matrix{Float64}, B::Matrix{Float64}, C::Matrix{Float64}, K::Float64, freq::Float64; tspan=(0.0, 100.0), solver=DE.Tsit5(), problemargs=(), solverargs=())
    """
        Simulate the ideal linear system without the IF from the initial state x0 using the optimized parameters.
    """

    # Define the ODE problem
    function dynamics!(du, u, p, t)
        du[1:end] = (system.A - system.B * system.K * system.C) * u
    end

    prob = DE.ODEProblem(dynamics!, x0, tspan, nothing; problemargs...)
    sol = DE.solve(prob, solver; solverargs...)

    return sol
end