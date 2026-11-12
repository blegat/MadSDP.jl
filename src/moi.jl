# SolverCore / MOI plumbing.  We don't implement an MOI optimiser from scratch
# — we reuse `LowRankOpt.Optimizer`, which already handles MOI vectorised PSD
# constraints, nonnegative cones, low-rank constraints, and the dual /
# objective bookkeeping.  We just need to:
#
#   * wrap our IPM in a `SolverCore.AbstractOptimizationSolver` exposing
#     `.stats` of type `SolverCore.GenericExecutionStats`;
#   * implement `SolverCore.solve!` so the LRO optimiser can drive us;
#   * implement `MOI.get` for `LRO.DualSolution` (primal X / x) and `MOI.SolverName`.
#
# Then `MadSDP.Optimizer()` returns a pre-configured `LRO.Optimizer`.

import SolverCore
import MathOptInterface as MOI

const MOI_STATUS = Dict(
    MadNLP.SOLVE_SUCCEEDED => :first_order,
    MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL => :acceptable,
    MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL => :small_step,
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

`LowRankOpt.Optimizer` only gives its solver options to `SolverCore.solve!`
so the `MadSDPSolver` is created there. This allows to choose the components
from JuMP, e.g., with `set_attribute(model, "direction", MadSDP.HKM())`.
"""
mutable struct Solver{T,M} <: SolverCore.AbstractOptimizationSolver
    model::M
    inner::Any
    stats::SolverCore.GenericExecutionStats{T,Vector{T},Vector{T},Any}
end

function Solver(src::Union{LRO.Model{T},LRO.AbstractDualModel{T}}; kws...) where {T}
    model = LRO.primal_model(src)
    stats = SolverCore.GenericExecutionStats(model)
    return Solver{T,typeof(model)}(model, nothing, stats)
end

function SolverCore.solve!(
    s::Solver{T},
    ::NLPModels.AbstractNLPModel;   # the LRO model — already cached in `s.model`
    verbose::Int = 1,
    kws...,
) where {T}
    if iszero(s.model.meta.nvar)
        return _constant_problem!(s)
    end
    s.inner = MadSDPSolver(s.model; kws...)
    solve!(s.inner; verbose = verbose > 0)
    # Populate the stats fields LRO.Optimizer reads from.
    stats = s.stats
    SolverCore.set_status!(stats, _solvercore_status(s.inner.status))
    SolverCore.set_objective!(stats, NLPModels.obj(s.model, s.inner.y))
    SolverCore.set_solver_specific!(stats, :dual_objective, -primal_objective(s.inner))
    SolverCore.set_solution!(stats, s.inner.y)
    multipliers = LRO.VectorizedConeVector(stats.multipliers, s.model.dim)
    if s.inner.schur isa Union{AugmentedKKTWorkspace,RootKKTWorkspace}
        ws = equality_block(s.inner.schur)
        fill!(multipliers, zero(T))
        _, _, rows, signs = scalar_block_bounds(s.model)
        for k in eachindex(rows)
            multipliers[LRO.ScalarIndex][rows[k]] += signs[k] * s.inner.x[k]
        end
        multipliers[LRO.ScalarIndex][ws.equalities] .= ws.eta
        for (i, block) in enumerate(s.inner.blocks)
            multipliers[LRO.MatrixIndex(i)] .= -block.X
        end
    else
        cone = LRO.ShapedConeVector(s.inner.x, [b.X for b in s.inner.blocks])
        copyto!(multipliers, -LRO.cone_multipliers(s.model, cone))
    end
    SolverCore.set_constraint_multipliers!(stats, stats.multipliers)
    SolverCore.set_residuals!(stats, s.inner.inf_du, s.inner.inf_pr)
    SolverCore.set_iter!(stats, s.inner.iter)
    SolverCore.set_time!(stats, time() - s.inner.start_time)
    return stats
end

# Cone multipliers exposed as an LRO `ShapedConeVector` so the MOI wrapper can
# index it with `sol[ScalarIndex]` / `sol[MatrixIndex(i)]`.
function MOI.get(s::Solver, ::LRO.DualSolution)
    return LRO.VectorizedConeVector(-s.stats.multipliers, s.model.dim)
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

function _constant_problem!(s::Solver{T}) where {T}
    model, stats = s.model, s.stats
    feasible = all(model.meta.lcon .<= model.constant .<= model.meta.ucon)
    cone = LRO.VectorizedConeVector(model.constant, model.dim)
    for idx in LRO.matrix_indices(model)
        feasible &=
            LinearAlgebra.eigmin(LinearAlgebra.Symmetric(cone[idx])) >= -sqrt(eps(T))
    end
    SolverCore.set_status!(stats, feasible ? :first_order : :infeasible)
    SolverCore.set_solution!(stats, T[])
    SolverCore.set_objective!(stats, model.objective_constant)
    SolverCore.set_solver_specific!(stats, :dual_objective, zero(T))
    SolverCore.set_constraint_multipliers!(stats, zeros(T, model.meta.ncon))
    SolverCore.set_residuals!(stats, feasible ? zero(T) : T(Inf), zero(T))
    SolverCore.set_iter!(stats, 0)
    SolverCore.set_time!(stats, 0.0)
    return stats
end
