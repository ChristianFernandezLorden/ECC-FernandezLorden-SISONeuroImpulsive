using LinearAlgebra
import Convex as Cv
using SCS
using MathOptInterface
using MatrixEquations
using LambertW

import DifferentialEquations as DE


mutable struct SpikingSystem
    # Parameters for the linear system
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Matrix{Float64}
    K::Float64
    P::Matrix{Float64}
    Q::Matrix{Float64}
    lambda_0::Float64
    # Parameters for the spiking system
    kappa::Float64
    alpha::Float64
    Delta::Float64
    I0::Float64
    # Functions for norms
    PVecNorm::Function
    PMatNorm::Function
    # Constructor
    SpikingSystem(A, B, C, K) = new(A, B, C, K, zeros(size(A)), zeros(size(A)), -1.0, -1.0, -1.0, -1.0, -1.0, x -> sqrt(x' * x), X -> opnorm(X))
end


function optimize_lambda_0!(system::SpikingSystem; tol=1e-6, lambda_opt_min=1.0, silent=true)
    """
        Optimize P and Q to maximize lambda_0 = λ_min(Q) / λ_max(P)
        subject to the Lyapunov equation: (A - K*B*C)'*P + P*(A - K*B*C) + Q = 0
    """

    A = system.A
    B = system.B
    C = system.C
    K = system.K

    A_cl = A - K*B*C  # Closed-loop system matrix

    if maximum(real(eigvals(A_cl))) >= 0
        error("The closed-loop system is not stable. Please choose K such that A - K*B*C is stable.")
    end
    
    # Decision variables
    n = size(A, 1)

    P = Cv.Semidefinite(n)
    Q = Cv.Semidefinite(n)
    lambda_min_Q = lambda_opt_min
    lambda_max_P = Cv.Variable()

    Idn = Matrix(I, n, n)
    
    # Constraints
    constraints = [
        # Lyapunov equation constraint
        A_cl' * P + P * A_cl + Q == 0,
        # lambda_min(Q) constraint: λ_min(Q) >= lambda_min_Q 
        Cv.eigmin(Q) >= lambda_min_Q,
        # lambda_max(P) constraint: λ_max(P) <= lambda_max_P
        Cv.eigmax(P) <= lambda_max_P,
        # Ensure P is positive definite
        Cv.eigmin(P) >= 1e-6,
        # Q ⪰ 1e-6 * Idn,
        #lambda_min_Q >= 1e-6,
        #lambda_max_P >= 1e-6
    ]
    
    # Objective: maximize lambda_0 = λ_min(Q) / λ_max(P)
    # Equivalent to minimizing λ_max(P) / λ_min(Q)
    objective = Cv.minimize(lambda_max_P / lambda_min_Q, constraints)
    
    # Solve the problem
    s = Cv.solve!(objective, SCS.Optimizer; silent=silent)

    if s.status != MathOptInterface.OPTIMAL
        error("Optimization failed: $(s.status)")
    end

    Qsol = Cv.evaluate(Q)
    Psol = lyapc(A_cl', Qsol)
    Psolsqrt = sqrt(Psol)
    lambda_0 = minimum(real(eigvals(Qsol))) / maximum(real(eigvals(Psol)))
    
    # Update the system with the optimized P, Q, and lambda_0, and define the associated norms
    system.P = Psol
    system.Q = Qsol
    system.lambda_0 = lambda_0
    system.PVecNorm = x -> sqrt(x' * Psol * x)
    system.PMatNorm = X -> opnorm(Psolsqrt * X * inv(Psolsqrt))

    return system
end

function optimize_for_starting_point!(system::SpikingSystem, x0::Vector{Float64}; Delta=nothing)
    """
        Optimize the parameters alpha, Delta, and I0 for the spiking system based on the initial state x0.
    """

    x0_norm = system.PVecNorm(x0)
    if x0_norm == 0
        error("The initial state x0 cannot be the zero vector.")
    end

    kappa_max = system.lambda_0 / (2 * system.PMatNorm(system.B * system.C))

    # Choose a Delta value that makes the Lambert W bound easier if it is not provided
    if Delta === nothing
        Delta = x0_norm * system.PVecNorm(system.C[1, :]) * logarithmic_norm(system.P, system.A)
    end
    system.Delta = Delta

    # Declare bounds necessary for approximation theorem 
    min_bound = min(1/system.PMatNorm(system.A), 1/system.PMatNorm(system.A - system.K*system.B*system.C))

    # Find the optimal kappa that maximizes the minimum of the two bounds and the min_bound, which is the best bound we can get from the approximation theorem
    kappa_values = range(kappa_max*0.99, stop=kappa_max*0.01, length=99)
    best_kappa = -1.0
    best_bound = -Inf
    for kappa in kappa_values
        bound_lambert = I0_lambert_bound(system, kappa, x0)
        bound_kappa = I0_kappa_bound(system, kappa)
        current_bound = min(bound_lambert, bound_kappa, min_bound)
        if current_bound > best_bound
            best_bound = current_bound
            best_kappa = kappa
        end
    end
    system.kappa = best_kappa

    # Compute a good I0  based on the best bounds
    I0 = 1.5 * Delta / best_bound
    system.I0 = I0

    # Compute the best alpha based on the best kappa and the bounds on tk, which is the best we can do for the spiking system to have a spike at time tk
    max_spike_time = tk_max_bound(system, x0)
    min_spike_time = tk_min_bound(system, x0)

    min_gain = system.K - best_kappa
    max_gain = system.K + best_kappa

    alpha = ((min_gain*max_spike_time) + (max_gain*min_spike_time)) / 2

    # Update the system parameters
    system.alpha = alpha

    return system
end


function simulate_from_starting_point(system::SpikingSystem, x0::Vector{Float64}; tspan=(0.0, 100.0), solver=DE.Tsit5(), problemargs=(), callbackargs=(), solverargs=())
    """
        Simulate the spiking system from the initial state x0 using the optimized parameters.
    """

    # Define the ODE problem
    function spiking_dynamics!(du, u, p, t)
        du[1] = abs(system.C[1, :]' * u[2:end]) + system.I0 
        du[2:end] = system.A * u[2:end]
    end

    # Define the conditions for the IF 
    function if_condition(u, t, integrator)
        return u[1] - system.Delta
    end

    function if_up_affect!(integrator)
        integrator.u[1] = 0.0
        y = system.C[1, :]' * integrator.u[2:end]
        integrator.u[2:end] = integrator.u[2:end] - system.alpha * system.B * y# sqrt(abs(y)) * sign(y)
    end

    # Set up the ODE problem with the IF
    condition = DE.ContinuousCallback(if_condition, if_up_affect!, nothing; callbackargs...)
    prob = DE.ODEProblem(spiking_dynamics!, [0.0; x0], tspan, nothing; callback=condition, problemargs...)
    sol = DE.solve(prob, solver; solverargs...)

    return sol
end


function simulate_ideal_from_starting_point(system::SpikingSystem, x0::Vector{Float64}; tspan=(0.0, 100.0), solver=DE.Tsit5(), problemargs=(), solverargs=())
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


function I0_kappa_bound(system::SpikingSystem, kappa::Float64)
    """
        Compute the I0 bound based on kappa using the approximation theorem.
    """

    PMatNorm = system.PMatNorm
    A = system.A
    B = system.B
    C = system.C
    K = system.K
    lambda_0 = system.lambda_0

    eul = Base.MathConstants.e
    C1 = (PMatNorm(A)^3 + PMatNorm(A - K * B * C)^3) * ((2 * eul - 5) / 2) + (abs(K) + abs(kappa)) * PMatNorm(B * C) * (PMatNorm(A)^2) * (2 * eul - 2)
    C2 = PMatNorm(B * C) * ((abs(K) + abs(kappa)) * PMatNorm(A) + (abs(K)^2 / 2) * abs((C*B)[1, 1]))
    C3 = abs(kappa) * PMatNorm(B * C)

    return ( -(C2 + (lambda_0^2)/8) + sqrt((C2 + (lambda_0^2)/8)^2 + 4*C1*((lambda_0/2) - C3)) ) / (2*C1)
end

function I0_lambert_bound(system::SpikingSystem, kappa::Float64, x0::Vector{Float64})
    """
        Compute the I0 bound based on the Lambert W function using the approximation theorem.
    """

    PMatNorm = system.PMatNorm
    PVecNorm = system.PVecNorm
    A = system.A
    B = system.B
    C = system.C
    Cvec = C[1, :]
    K = system.K
    lambda_0 = system.lambda_0
    Delta = system.Delta

    A_log_norm = logarithmic_norm(system.P, A)
    kappa_ratio = (K - kappa)/(K + kappa)
    delta_log_ratio = (Delta * A_log_norm) / (PVecNorm(Cvec) * PVecNorm(x0))
    delta_log_delta_ratio = (Delta * A_log_norm) / (PVecNorm(Cvec) * PVecNorm(x0) + (1 - kappa_ratio)*Delta * A_log_norm)

    return log( (kappa_ratio * delta_log_ratio) / lambertw(kappa_ratio * delta_log_ratio * exp(kappa_ratio * delta_log_delta_ratio)) ) / (kappa_ratio * A_log_norm)
end

function tk_max_bound(system::SpikingSystem, x0::Vector{Float64})
    """
        Compute the maximum tk bound for the spiking system to have a spike at time tk based on the approximation theorem.
    """

    I0 = system.I0
    Delta = system.Delta

    return Delta/I0
end


function tk_min_bound(system::SpikingSystem, x0::Vector{Float64})
    """
        Compute the minimum tk bound for the spiking system to have a spike at time tk based on the approximation theorem.
    """

    PMatNorm = system.PMatNorm
    PVecNorm = system.PVecNorm
    A = system.A
    B = system.B
    C = system.C

    I0 = system.I0
    Delta = system.Delta

    A_log_norm = logarithmic_norm(system.P, A)

    norm_I0_ratio = (PVecNorm(C[1, :]) * PVecNorm(x0)) / I0
    lognorm_I0_ratio = (A_log_norm * Delta) / I0


    return log(lambertw(norm_I0_ratio * exp(norm_I0_ratio) * exp(lognorm_I0_ratio)) / (norm_I0_ratio)) / A_log_norm
end


function minimum_convergence(system::SpikingSystem, x0::Vector{Float64})
    """
        Compute the minimum convergence bound for the spiking system.
    """

    PMatNorm = system.PMatNorm
    A = system.A
    B = system.B
    C = system.C
    K = system.K
    lambda_0 = system.lambda_0
    kappa = system.kappa

    eul = Base.MathConstants.e
    C1 = (PMatNorm(A)^3 + PMatNorm(A - K * B * C)^3) * ((2 * eul - 5) / 2) + (abs(K) + abs(kappa)) * PMatNorm(B * C) * (PMatNorm(A)^2) * (2 * eul - 2)
    C2 = PMatNorm(B * C) * ((abs(K) + abs(kappa)) * PMatNorm(A) + (abs(K)^2 / 2) * abs((C*B)[1, 1]))
    C3 = abs(kappa) * PMatNorm(B * C)

    max_spike_time = tk_max_bound(system, x0)
    min_spike_time = tk_min_bound(system, x0)

    gamma(t) = exp(-t*lambda_0/2) + C1*t^3 + C2*t^2 + C3*t

    return max(gamma(min_spike_time), gamma(max_spike_time))
end

function logarithmic_norm(P::Matrix{Float64}, M::Matrix{Float64})
    """
        Compute the logarithmic norm of matrix M with respect to the norm defined by P.
    """    

    n = size(M, 1)
    if size(M) != (n, n)
        error("M must be a square matrix.")
    end
    if size(P) != (n, n)
        error("P must be an n×n matrix where n is the size of M.")
    end

    sqrtP = sqrt(P)
    invSqrtP = inv(sqrtP)
    H = Symmetric((sqrtP * M * invSqrtP + invSqrtP' * M' * sqrtP') / 2)
    lambdas = eigvals(H)
    #Psym = Symmetric(sqrt(P))      # enforce exact symmetry
    #H = Symmetric((Psym * M + M' * Psym) / 2)
    #lambdas = eigvals(H, Psym)             # LAPACK dsygv, stable
    return maximum(real(lambdas))
end