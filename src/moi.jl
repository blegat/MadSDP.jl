# SolverCore / MOI plumbing.  We don't implement an MOI optimiser from scratch
# — we reuse `LowRankOpt.Optimizer`, which already handles MOI vectorised PSD
# constraints, nonnegative cones, low-rank constraints, and the dual /
# objective bookkeeping.  We just need to:
#
#   * wrap our IPM in a `SolverCore.AbstractOptimizationSolver` exposing
#     `.stats` of type `SolverCore.GenericExecutionStats`;
#   * implement `SolverCore.solve!` so the LRO optimiser can drive us;
#   * implement `MOI.get` for `LRO.Solution` (primal X / x) and `MOI.SolverName`.
#
# Then `MadSDP.Optimizer()` returns a pre-configured `LRO.Optimizer`.

import SolverCore
import MathOptInterface as MOI

const MOI_STATUS = Dict(
    MadNLP.SOLVE_SUCCEEDED => :first_order,
    MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL => :acceptable,
    MadNLP.MAXIMUM_ITERATIONS_EXCEEDED => :max_iter,
    MadNLP.MAXIMUM_WALLTIME_EXCEEDED => :max_time,
    MadNLP.INFEASIBLE_PROBLEM_DETECTED => :infeasible,
    MadNLP.ERROR_IN_STEP_COMPUTATION => :exception,
    MadNLP.USER_REQUESTED_STOP => :user,
    MadNLP.INITIAL => :unknown,
    MadNLP.REGULAR => :unknown,
)

_solvercore_status(s::MadNLP.Status) = get(MOI_STATUS, s, :unknown)

"""
    Solver(model::LRO.Model; kws...)

Thin `SolverCore.AbstractOptimizationSolver` wrapper around `MadSDPSolver` so
that `LowRankOpt.Optimizer` can drive the IPM and expose the result through
MathOptInterface.
"""
struct Solver{T,IS} <: SolverCore.AbstractOptimizationSolver
    inner::IS
    stats::SolverCore.GenericExecutionStats{T,Vector{T},Vector{T},Any}
end

function Solver(model::LRO.AbstractModel{T}; kws...) where {T}
    inner = MadSDPSolver(model; kws...)
    stats = SolverCore.GenericExecutionStats(model)
    return Solver{T,typeof(inner)}(inner, stats)
end

function SolverCore.solve!(
    s::Solver{T},
    ::NLPModels.AbstractNLPModel;   # the LRO model — already cached in `s.inner.model`
    verbose::Int = 1,
    kws...,
) where {T}
    solve!(s.inner; verbose = verbose > 0)
    # Populate the stats fields LRO.Optimizer reads from.
    stats = s.stats
    SolverCore.set_status!(stats, _solvercore_status(s.inner.status))
    SolverCore.set_objective!(stats, primal_objective(s.inner))
    SolverCore.set_solver_specific!(stats, :dual_objective, dual_objective(s.inner))
    # `multipliers` holds the LRO `y` (== MOI variable primals after dualisation).
    SolverCore.set_constraint_multipliers!(stats, copy(s.inner.y))
    SolverCore.set_residuals!(stats, s.inner.inf_pr, s.inner.inf_du)
    SolverCore.set_iter!(stats, s.inner.iter)
    SolverCore.set_time!(stats, time() - s.inner.start_time)
    return stats
end

# Primal `(x, X)` exposed as an LRO `ShapedSolution` so the MOI wrapper can
# index it with `sol[ScalarIndex]` / `sol[MatrixIndex(i)]`.
function MOI.get(s::Solver, ::LRO.Solution)
    return LRO.ShapedSolution(copy(s.inner.x), [copy(b.X) for b in s.inner.blocks])
end

MOI.get(::Solver, ::MOI.SolverName) = "MadSDP"

"""
    Optimizer()

Returns an `MOI.AbstractOptimizer` that solves SDPs (and mixed SDP + LP) by
combining `LowRankOpt.Optimizer`'s MOI plumbing with the MadSDP primal-dual NT
interior-point solver.
"""
function Optimizer()
    opt = LRO.Optimizer{Float64}()
    MOI.set(opt, MOI.RawOptimizerAttribute("solver"), Solver)
    return opt
end
