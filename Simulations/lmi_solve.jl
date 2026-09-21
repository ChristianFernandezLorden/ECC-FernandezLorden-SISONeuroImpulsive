using LinearAlgebra
using Convex
using SCS
using MathOptInterface

function solve_lmi(A, B, C, alpha, tk, tol=1e-6)
    n = size(A, 1)

    # Define the decision variable
    P = Semidefinite(n)

    # Define Schur complement for the LMI
    M = [P P*(I-alpha*B*C)*exp(A*tk); (P*(I-alpha*B*C)*exp(A*tk))' P]

    # Define the LMI constraints
    constraint = M >= tol

    # Define the optimization problem
    problem = satisfy(constraint)

    # Solve the problem using SCS
    s = solve!(problem, SCS.Optimizer; silent=true)

    # Output feasibility
    if s.status == MathOptInterface.OPTIMAL
        return true
    else
        return false
    end
end

#=
tk_best = Variable(1, Positive());
M_best = [P P*(I - alpha_best*B*C)*exp(A * tk_best); (P * (I - alpha_best * B * C) * exp(A * tk_best))' P];
constraint = M_best >= tol
#obj = (alpha_best + 1/tk_best);
#problem = minimize(obj, constraint)
problem = minimize(tk_best, constraint)
s = solve!(problem, SCS.Optimizer; silent=true)
if s.status == MathOptInterface.OPTIMAL
    return (true, evaluate(alpha_best), evaluate(tk_best))
else
    return (false, nothing, nothing)
end
=#