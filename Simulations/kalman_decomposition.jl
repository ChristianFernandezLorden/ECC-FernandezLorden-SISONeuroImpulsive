# =============================================================================
#  Linear Systems – Kalman Structural Decompositions
# =============================================================================
#
#  Implements the three classical state-space decompositions for LTI systems:
#
#    ẋ = A x + B u
#    y = C x + D u
#
#  1. Controllability Decomposition  (separates controllable/uncontrollable)
#  2. Observability  Decomposition   (separates observable/unobservable)
#  3. Full Kalman Decomposition      (four-way canonical partition)
#
# =============================================================================

using LinearAlgebra
using Random


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

"""
    controllability_matrix(A, B) → Wc

Builds the controllability matrix:
    Wc = [B  AB  A²B  ⋯  Aⁿ⁻¹B]   ∈ ℝⁿˣⁿᵐ
"""
function controllability_matrix(A::AbstractMatrix, B::AbstractMatrix)
    n  = size(A, 1)
    Wc = copy(B)
    Ak = copy(B)
    for _ in 2:n
        Ak = A * Ak
        Wc = hcat(Wc, Ak)
    end
    return Wc
end

"""
    observability_matrix(A, C) → Wo

Builds the observability matrix (stacked as rows):
    Wo = [C; CA; CA²; ⋯; CAⁿ⁻¹]   ∈ ℝⁿᵖˣⁿ
"""
function observability_matrix(A::AbstractMatrix, C::AbstractMatrix)
    n  = size(A, 1)
    Wo = copy(C)
    Ck = copy(C)
    for _ in 2:n
        Ck = Ck * A
        Wo = vcat(Wo, Ck)
    end
    return Wo
end

"""
    numerical_rank(M; tol) → r

Numerical rank via SVD, using a relative tolerance on singular values.
"""
function numerical_rank(M::AbstractMatrix; tol=1e-10)
    isempty(M) && return 0
    sv = svdvals(M)
    isempty(sv) && return 0
    return sum(sv .> tol * max(sv[1], 1e-14))
end

"""
    extend_to_basis(V, n) → T

Given an n×k matrix V with orthonormal columns, extends it to a full
orthonormal n×n basis [V | V⊥] by appending the orthogonal complement.
"""
function extend_to_basis(V::AbstractMatrix, n::Int; tol=1e-10)
    k = size(V, 2)
    k == n && return Matrix{Float64}(V)
    k == 0 && return Matrix{Float64}(I, n, n)
    return hcat(Matrix{Float64}(V), nullspace(V'; atol=tol))
end

"""Zero out entries below a scaled threshold (in-place)."""
function threshold!(M::AbstractMatrix, scale::Real; tol=1e-10)
    ε = tol * max(scale, 1.0)
    @. M = abs(M) < ε ? zero(eltype(M)) : M
    return M
end

"""Print a labelled matrix (or '[empty]' for size-0 arrays)."""
function print_matrix(label::String, M::AbstractMatrix; digits=4)
    println("  ", label, ":")
    if isempty(M)
        println("    [empty — dimension 0]")
        return
    end
    for i in axes(M, 1)
        row = [round(M[i,j]; digits=digits) for j in axes(M, 2)]
        println("    ", row)
    end
end


# ─────────────────────────────────────────────────────────────────────────────
#  1.  CONTROLLABILITY DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

"""
    controllability_decomposition(A, B, C, D; tol=1e-10) → NamedTuple

Computes the **controllability decomposition** of the LTI system (A,B,C,D).

Finds a state transformation T such that, with z = T⁻¹x:

         ⎡ Ac   A₁₂ ⎤          ⎡ Bc ⎤
    Ā  = ⎣  0   Auc ⎦   B̄  =  ⎣  0 ⎦   C̄ = [Cc  Cuc]

  • (Ac, Bc)  controllable subsystem,  dimension rc
  • (Auc)     uncontrollable part,     dimension n−rc

Returns a NamedTuple with fields:
    T, Tinv, Ā, B̄, C̄, D    — transformed system
    rc, ruc                  — dimensions
    Ac, A12, Auc, Bc, Cc, Cuc — block matrices
"""
function controllability_decomposition(A, B, C, D; tol=1e-10)
    n   = size(A, 1)
    Wc  = controllability_matrix(A, B)
    rc  = numerical_rank(Wc; tol=tol)
    ruc = n - rc

    # Left singular vectors of Wc span the controllable subspace
    U  = svd(Wc).U
    Tc = rc > 0 ? U[:, 1:rc] : zeros(n, 0)
    T  = extend_to_basis(Tc, n; tol=tol)

    Tinv = inv(T)
    sc   = norm(A) + norm(B) + norm(C) + 1          # scale for thresholding
    Ā    = threshold!(Tinv * A * T, sc; tol=tol)
    B̄    = threshold!(Tinv * B,     sc; tol=tol)
    C̄    = threshold!(C * T,         sc; tol=tol)

    return (
        T    = T,     Tinv = Tinv,
        Ā    = Ā,     B̄    = B̄,    C̄    = C̄,   D = D,
        rc   = rc,    ruc  = ruc,
        # ── block matrices ──────────────────────────────────────
        Ac   = rc>0          ? Ā[1:rc,     1:rc]       : zeros(0,0),
        A12  = rc>0 && ruc>0 ? Ā[1:rc,     rc+1:end]   : zeros(rc, ruc),
        Auc  = ruc>0         ? Ā[rc+1:end, rc+1:end]   : zeros(0,0),
        Bc   = rc>0          ? B̄[1:rc,     :]           : zeros(0, size(B,2)),
        Cc   = rc>0          ? C̄[:,         1:rc]        : zeros(size(C,1), 0),
        Cuc  = ruc>0         ? C̄[:,         rc+1:end]   : zeros(size(C,1), 0),
    )
end

"""Pretty-print the result of `controllability_decomposition`."""
function display_controllability(r)
    n = r.rc + r.ruc
    println("\n", "═"^64)
    println(" 1.  CONTROLLABILITY DECOMPOSITION")
    println("═"^64)
    @printf("  n = %d   |   rc (controllable) = %d   |   ruc (uncontrollable) = %d\n",
            n, r.rc, r.ruc)
    println()
    println("  Structure after transformation  z = T⁻¹x:")
    println()
    println("    Ā = T⁻¹AT  =  ⎡ Ac   A₁₂ ⎤    (Ac,Bc) is controllable")
    println("                   ⎣  0   Auc ⎦")
    println("    B̄ = T⁻¹B   =  ⎡ Bc ⎤")
    println("                   ⎣  0 ⎦")
    println("    C̄ = CT     =  [ Cc   Cuc ]")
    println()
    print_matrix("Ā = T⁻¹AT", r.Ā)
    print_matrix("B̄ = T⁻¹B",  r.B̄)
    print_matrix("C̄ = CT",     r.C̄)
    println()
    println("  ┌─ Controllable block ─────────────────────────────────────┐")
    print_matrix("Ac  (dynamics)", r.Ac)
    print_matrix("Bc  (inputs)",   r.Bc)
    print_matrix("Cc  (outputs)",  r.Cc)
    println("  └──────────────────────────────────────────────────────────┘")
    println("  ┌─ Uncontrollable block ───────────────────────────────────┐")
    print_matrix("Auc (dynamics)", r.Auc)
    print_matrix("Cuc (outputs)",  r.Cuc)
    println("  └──────────────────────────────────────────────────────────┘")
    println()
    # Verification
    Bbot = r.ruc > 0 ? norm(r.B̄[r.rc+1:end, :]) : 0.0
    @printf("  ✔  Verification  ‖B̄[rc+1:end, :]‖ = %.2e  (should ≈ 0)\n", Bbot)
end


# ─────────────────────────────────────────────────────────────────────────────
#  2.  OBSERVABILITY DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

"""
    observability_decomposition(A, B, C, D; tol=1e-10) → NamedTuple

Computes the **observability decomposition** of the LTI system (A,B,C,D).

Finds a state transformation T such that, with z = T⁻¹x:

         ⎡ Ao   0  ⎤          ⎡ Bo  ⎤
    Ā  = ⎣ A₂₁ Auo ⎦   B̄  =  ⎣ Buo ⎦   C̄ = [Co  0]

  • (Ao, Co)  observable subsystem,  dimension ro
  • (Auo)     unobservable part,     dimension n−ro

Returns a NamedTuple with fields:
    T, Tinv, Ā, B̄, C̄, D    — transformed system
    ro, ruo                  — dimensions
    Ao, A21, Auo, Bo, Buo, Co — block matrices
"""
function observability_decomposition(A, B, C, D; tol=1e-10)
    n   = size(A, 1)
    Wo  = observability_matrix(A, C)
    ro  = numerical_rank(Wo; tol=tol)
    ruo = n - ro

    # Right singular vectors of Wo span the observable subspace
    V  = svd(Wo).V
    To = ro > 0 ? V[:, 1:ro] : zeros(n, 0)
    T  = extend_to_basis(To, n; tol=tol)

    Tinv = inv(T)
    sc   = norm(A) + norm(B) + norm(C) + 1
    Ā    = threshold!(Tinv * A * T, sc; tol=tol)
    B̄    = threshold!(Tinv * B,     sc; tol=tol)
    C̄    = threshold!(C * T,         sc; tol=tol)

    return (
        T    = T,    Tinv = Tinv,
        Ā    = Ā,    B̄    = B̄,    C̄    = C̄,   D = D,
        ro   = ro,   ruo  = ruo,
        # ── block matrices ──────────────────────────────────────
        Ao   = ro>0          ? Ā[1:ro,     1:ro]      : zeros(0,0),
        A21  = ro>0 && ruo>0 ? Ā[ro+1:end, 1:ro]      : zeros(ruo, ro),
        Auo  = ruo>0         ? Ā[ro+1:end, ro+1:end]  : zeros(0,0),
        Bo   = ro>0          ? B̄[1:ro,     :]          : zeros(0, size(B,2)),
        Buo  = ruo>0         ? B̄[ro+1:end, :]          : zeros(0, size(B,2)),
        Co   = ro>0          ? C̄[:,         1:ro]       : zeros(size(C,1), 0),
    )
end

"""Pretty-print the result of `observability_decomposition`."""
function display_observability(r)
    n = r.ro + r.ruo
    println("\n", "═"^64)
    println(" 2.  OBSERVABILITY DECOMPOSITION")
    println("═"^64)
    @printf("  n = %d   |   ro (observable) = %d   |   ruo (unobservable) = %d\n",
            n, r.ro, r.ruo)
    println()
    println("  Structure after transformation  z = T⁻¹x:")
    println()
    println("    Ā = T⁻¹AT  =  ⎡ Ao    0  ⎤    (Ao,Co) is observable")
    println("                   ⎣ A₂₁  Auo ⎦")
    println("    B̄ = T⁻¹B   =  ⎡ Bo  ⎤")
    println("                   ⎣ Buo ⎦")
    println("    C̄ = CT     =  [ Co   0 ]")
    println()
    print_matrix("Ā = T⁻¹AT", r.Ā)
    print_matrix("B̄ = T⁻¹B",  r.B̄)
    print_matrix("C̄ = CT",     r.C̄)
    println()
    println("  ┌─ Observable block ───────────────────────────────────────┐")
    print_matrix("Ao  (dynamics)", r.Ao)
    print_matrix("Bo  (inputs)",   r.Bo)
    print_matrix("Co  (outputs)",  r.Co)
    println("  └──────────────────────────────────────────────────────────┘")
    println("  ┌─ Unobservable block ─────────────────────────────────────┐")
    print_matrix("Auo (dynamics)", r.Auo)
    print_matrix("Buo (inputs)",   r.Buo)
    println("  └──────────────────────────────────────────────────────────┘")
    println()
    # Verification
    Cright = r.ruo > 0 ? norm(r.C̄[:, r.ro+1:end]) : 0.0
    @printf("  ✔  Verification  ‖C̄[:, ro+1:end]‖ = %.2e  (should ≈ 0)\n", Cright)
end


# ─────────────────────────────────────────────────────────────────────────────
#  3.  KALMAN DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

"""
    kalman_decomposition(A, B, C, D; tol=1e-10) → NamedTuple

Computes the **full Kalman decomposition** of the LTI system (A,B,C,D).

Partitions the n-dimensional state space into four subspaces:

    co   – Controllable  AND Observable      (dimension n_co)
    cno  – Controllable  AND Unobservable    (dimension n_cno)
    nco  – Uncontrollable AND Observable     (dimension n_nco)
    ncno – Uncontrollable AND Unobservable   (dimension n_ncno)

With state ordering  z = T⁻¹x = [z_co; z_cno; z_nco; z_ncno]:

         ⎡ A_co    0      *      0    ⎤
    Ā  = ⎢ A₂₁  A_cno    *      0    ⎥
         ⎢  0      0    A_nco    0    ⎥
         ⎣  0      0     A₄₃  A_ncno ⎦

         ⎡ B_co  ⎤
    B̄  = ⎢ B_cno ⎥     C̄ = [C_co  0  C_nco  0]
         ⎢  0    ⎥
         ⎣  0    ⎦

⚡ The transfer function equals:  H(s) = C_co (sI − A_co)⁻¹ B_co + D

Returns a NamedTuple with fields:
    T, Tinv, Ā, B̄, C̄, D          — full transformed system
    dims                            — NamedTuple (co, cno, nco, ncno)
    Aco, Acno, Anco, Ancno         — diagonal blocks
    Bco, Bcno                      — input blocks (only controllable)
    Cco, Cnco                      — output blocks (only observable)
"""
function kalman_decomposition(A, B, C, D; tol=1e-10)
    n = size(A, 1)
    m = size(B, 2)
    p = size(C, 1)

    # ── Step 1: Controllability decomposition of the full system ──────────
    cd  = controllability_decomposition(A, B, C, D; tol=tol)
    rc  = cd.rc
    ruc = cd.ruc

    # ── Step 2a: Observability decomp within the controllable subspace ────
    if rc > 0
        od_c  = observability_decomposition(cd.Ac, cd.Bc, cd.Cc, D; tol=tol)
        n_co  = od_c.ro
        n_cno = od_c.ruo
    else
        od_c  = nothing
        n_co  = 0;  n_cno = 0
    end

    # ── Step 2b: Observability decomp within the uncontrollable subspace ──
    if ruc > 0
        Buc   = zeros(ruc, m)   # uncontrollable inputs are structurally zero
        od_uc = observability_decomposition(cd.Auc, Buc, cd.Cuc, D; tol=tol)
        n_nco  = od_uc.ro
        n_ncno = od_uc.ruo
    else
        od_uc  = nothing
        n_nco  = 0;  n_ncno = 0
    end

    # ── Step 3: Build the global transformation matrix ────────────────────
    #  T_cd splits X into [X_c | X_uc]; then within each block we refine.
    Tc  = rc  > 0 ? cd.T[:, 1:rc]   : zeros(n, 0)
    Tuc = ruc > 0 ? cd.T[:, rc+1:end] : zeros(n, 0)

    T_co   = rc>0  && n_co>0   ? Tc  * od_c.T[:, 1:n_co]            : zeros(n, 0)
    T_cno  = rc>0  && n_cno>0  ? Tc  * od_c.T[:, n_co+1:end]        : zeros(n, 0)
    T_nco  = ruc>0 && n_nco>0  ? Tuc * od_uc.T[:, 1:n_nco]          : zeros(n, 0)
    T_ncno = ruc>0 && n_ncno>0 ? Tuc * od_uc.T[:, n_nco+1:end]      : zeros(n, 0)

    T    = hcat(T_co, T_cno, T_nco, T_ncno)
    Tinv = inv(T)
    sc   = norm(A) + norm(B) + norm(C) + 1
    Ā    = threshold!(Tinv * A * T, sc; tol=tol)
    B̄    = threshold!(Tinv * B,     sc; tol=tol)
    C̄    = threshold!(C * T,         sc; tol=tol)

    # ── Index boundaries ──────────────────────────────────────────────────
    i1 = n_co
    i2 = n_co + n_cno
    i3 = n_co + n_cno + n_nco

    return (
        T     = T,    Tinv  = Tinv,
        Ā     = Ā,    B̄     = B̄,    C̄     = C̄,    D = D,
        dims  = (co=n_co, cno=n_cno, nco=n_nco, ncno=n_ncno),
        # ── diagonal A blocks ─────────────────────────────────────────────
        Aco   = i1>0    ? Ā[1:i1,     1:i1]      : zeros(0,0),
        Acno  = n_cno>0 ? Ā[i1+1:i2,  i1+1:i2]  : zeros(0,0),
        Anco  = n_nco>0 ? Ā[i2+1:i3,  i2+1:i3]  : zeros(0,0),
        Ancno = n_ncno>0? Ā[i3+1:end, i3+1:end]  : zeros(0,0),
        # ── B blocks (nonzero only for controllable states) ───────────────
        Bco   = i1>0    ? B̄[1:i1,    :]           : zeros(0, m),
        Bcno  = n_cno>0 ? B̄[i1+1:i2, :]           : zeros(0, m),
        # ── C blocks (nonzero only for observable states) ─────────────────
        Cco   = i1>0    ? C̄[:,        1:i1]        : zeros(p, 0),
        Cnco  = n_nco>0 ? C̄[:,        i2+1:i3]     : zeros(p, 0),
    )
end

"""Pretty-print the result of `kalman_decomposition`."""
function display_kalman(r)
    d  = r.dims
    n  = d.co + d.cno + d.nco + d.ncno
    println("\n", "═"^64)
    println(" 3.  KALMAN DECOMPOSITION")
    println("═"^64)
    @printf("  n = %d   state dimension\n", n)
    println()
    println("  ┌──────────────────────┬──────────────────────┐")
    println("  │                      │                      │")
    @printf("  │  Controllable        │  Controllable        │\n")
    @printf("  │  & Observable        │  & Unobservable      │\n")
    @printf("  │  n_co  = %3d         │  n_cno = %3d         │\n", d.co, d.cno)
    println("  ├──────────────────────┼──────────────────────┤")
    @printf("  │  Uncontrollable      │  Uncontrollable      │\n")
    @printf("  │  & Observable        │  & Unobservable      │\n")
    @printf("  │  n_nco = %3d         │  n_ncno= %3d         │\n", d.nco, d.ncno)
    println("  └──────────────────────┴──────────────────────┘")
    println()
    println("  Kalman canonical structure  z = T⁻¹x = [z_co; z_cno; z_nco; z_ncno]:")
    println()
    println("    Ā  =  ⎡ A_co    0      *      0    ⎤")
    println("          ⎢ A₂₁  A_cno    *      0    ⎥")
    println("          ⎢  0      0    A_nco    0    ⎥")
    println("          ⎣  0      0     A₄₃  A_ncno ⎦")
    println()
    println("    B̄  =  ⎡ B_co  ⎤     C̄ = [C_co  0  C_nco  0]")
    println("          ⎢ B_cno ⎥")
    println("          ⎢  0    ⎥")
    println("          ⎣  0    ⎦")
    println()
    println("  ⚡  Transfer function  H(s) = C_co (sI − A_co)⁻¹ B_co + D")
    println()
    print_matrix("Ā = T⁻¹AT (full, rounded)", r.Ā)
    print_matrix("B̄ = T⁻¹B",                   r.B̄)
    print_matrix("C̄ = CT",                      r.C̄)
    println()
    println("  ┌─ Diagonal A blocks ──────────────────────────────────────┐")
    print_matrix("A_co   (controllable & observable)",   r.Aco)
    print_matrix("A_cno  (controllable & unobservable)", r.Acno)
    print_matrix("A_nco  (uncontrollable & observable)", r.Anco)
    print_matrix("A_ncno (uncontrollable & unobservable)", r.Ancno)
    println("  └──────────────────────────────────────────────────────────┘")
    println("  ┌─ B blocks ───────────────────────────────────────────────┐")
    print_matrix("B_co  (inputs to co states)",  r.Bco)
    print_matrix("B_cno (inputs to cno states)", r.Bcno)
    println("  └──────────────────────────────────────────────────────────┘")
    println("  ┌─ C blocks ───────────────────────────────────────────────┐")
    print_matrix("C_co  (outputs from co states)",  r.Cco)
    print_matrix("C_nco (outputs from nco states)", r.Cnco)
    println("  └──────────────────────────────────────────────────────────┘")
    println()
    # Verification
    i1 = d.co; i2 = d.co + d.cno; i3 = d.co + d.cno + d.nco
    Bnco_check  = (d.nco + d.ncno) > 0 ? norm(r.B̄[i2+1:end, :])     : 0.0
    Ccno_check  = d.cno > 0            ? norm(r.C̄[:, i1+1:i2])       : 0.0
    Cncno_check = d.ncno > 0           ? norm(r.C̄[:, i3+1:end])      : 0.0
    @printf("  ✔  Verification:\n")
    @printf("      ‖B̄ rows for uncontrollable states‖ = %.2e  (should ≈ 0)\n", Bnco_check)
    @printf("      ‖C̄ cols for cno states‖            = %.2e  (should ≈ 0)\n", Ccno_check)
    @printf("      ‖C̄ cols for ncno states‖           = %.2e  (should ≈ 0)\n", Cncno_check)
end


# ─────────────────────────────────────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────────────────────────────────────
# We build a 4-state system whose Kalman structure is known:
#
#   State 1 – controllable & observable      (co)
#   State 2 – controllable & unobservable    (cno)
#   State 3 – uncontrollable & observable    (nco)
#   State 4 – uncontrollable & unobservable  (ncno)
#
# The system is written in Kalman canonical form, then SCRAMBLED by a random
# invertible transformation T₀ to make the structure non-obvious.
# The three decompositions then recover it.
# ─────────────────────────────────────────────────────────────────────────────

using Printf

println("\n", "█"^64)
println(" DEMO: Kalman Structural Decompositions")
println("█"^64)

# ── Ground-truth system in Kalman canonical form ─────────────────────────────
#
#  A block structure:
#    A_co  = -1,  A_cno = -2,  A_nco = -3,  A_ncno = -4
#  Off-diagonal coupling (*) between the subspaces:
#    A[1,3] = 1   (co ← nco coupling — allowed by controllability structure)
#    A[2,3] = 0.5 (cno ← nco coupling)
#    A[4,3] = 1   (ncno ← nco coupling)
#
A_k = [-1.0   0.0   1.0   0.0;
        1.0  -2.0   0.5   0.0;
        0.0   0.0  -3.0   0.0;
        0.0   0.0   1.0  -4.0]

B_k = reshape([1.0, 1.0, 0.0, 0.0], 4, 1)  # only co and cno are driven

C_k = [1.0  0.0  1.0  0.0]                  # only co and nco are observed

D_k = reshape([0.5], 1, 1)

# ── Scramble with a random invertible transformation ─────────────────────────
Random.seed!(7)
T₀ = randn(4, 4)
# Condition the scramble so it's well-posed
T₀ = T₀ / norm(T₀) * 4.0

A = T₀ * A_k * inv(T₀)
B = T₀ * B_k
C = C_k * inv(T₀)
D = D_k

println("\n  Original (scrambled) system matrices:")
print_matrix("A", A)
print_matrix("B", B)
print_matrix("C", C)

# ── Run the three decompositions ─────────────────────────────────────────────

cd_result = controllability_decomposition(A, B, C, D)
display_controllability(cd_result)

od_result = observability_decomposition(A, B, C, D)
display_observability(od_result)

kd_result = kalman_decomposition(A, B, C, D)
display_kalman(kd_result)

# ── Transfer function comparison ─────────────────────────────────────────────
println("\n", "═"^64)
println(" TRANSFER FUNCTION VERIFICATION")
println("═"^64)
println("  H(s) at s = 1 (via full system vs. Kalman co block):\n")

s = 1.0 + 0im
n = size(A, 1)
H_full = C * ((s * I(n) - A) \ B) + D

A_co = kd_result.Aco
B_co = kd_result.Bco
C_co = kd_result.Cco
nc   = kd_result.dims.co
H_co = (nc > 0) ? (C_co * ((s * I(nc) - A_co) \ B_co) + D) : D

@printf("  H_full(1) = %+.6f%+.6fi\n", real(H_full[1]), imag(H_full[1]))
@printf("  H_co(1)   = %+.6f%+.6fi   (from co block only)\n", real(H_co[1]), imag(H_co[1]))
@printf("  Difference = %.2e\n", abs(H_full[1] - H_co[1]))
println()
println("  ✔  Transfer functions match — only (A_co, B_co, C_co, D)")
println("     determines the input-output behaviour of the system.")
println()