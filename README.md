# MadSDP.jl

[![Build Status](https://github.com/blegat/MadSDP.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/blegat/MadSDP.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/blegat/MadSDP.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/blegat/MadSDP.jl)

[MadSDP.jl](https://github.com/blegat/MadSDP.jl) is a semidefinite programming solver written in Julia. It solves semidefinite programs, including problems with both positive semidefinite matrix variables and nonnegative scalar variables, using interior-point methods.

MadSDP is a **malleable solver**, thanks to [MadNLP](https://github.com/MadNLP/MadNLP.jl): it uses MadNLP's linear solver interface and infrastructure, and exposes interchangeable algorithmic components. You can choose the interior-point algorithm, search direction, step-length rule, centering strategy, initial point, and Schur complement solver, or implement new components in Julia.

MadSDP provides an interface to [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) through [LowRankOpt](https://github.com/blegat/LowRankOpt.jl), so it can be used with [JuMP](https://jump.dev).

## Getting help

If you need help, please ask a question on the [JuMP community forum](https://jump.dev/forum).

If you have a reproducible example of a bug, please [open a GitHub issue](https://github.com/blegat/MadSDP.jl/issues/new).

## Installation

MadSDP requires Julia 1.10 or later. Install it from GitHub using the Julia package manager:

```julia
import Pkg
Pkg.add(url = "https://github.com/blegat/MadSDP.jl")
Pkg.add("JuMP")
```

## Use with JuMP

To use MadSDP with JuMP, use `MadSDP.Optimizer`. This example minimizes `2X[1, 1] + X[2, 2]` over positive semidefinite matrices with trace one:

```julia
using JuMP
import MadSDP

model = Model(MadSDP.Optimizer)
@variable(model, X[1:2, 1:2], PSD)
@constraint(model, X[1, 1] + X[2, 2] == 1)
@objective(model, Min, 2X[1, 1] + X[2, 2])
optimize!(model)

assert_is_solved_and_feasible(model)
objective_value(model)  # approximately 1.0
value.(X)              # approximately [0.0 0.0; 0.0 1.0]
```

## Options

Set numerical options using JuMP's `set_attribute`:

```julia
set_attribute(model, "tol", 1e-8)
set_attribute(model, "max_iter", 200)
set_attribute(model, "max_wall_time", 60.0)
set_silent(model)
```

Common options are:

| Name | Default | Description |
| --- | --- | --- |
| `tol` | `1e-7` | Convergence tolerance. |
| `max_iter` | `100` | Maximum number of iterations. |
| `max_wall_time` | `3600.0` | Time limit in seconds. |
| `mu_min` | `1e-14` | Minimum barrier parameter for `PrimalDual`; reaching it without convergence stops with a small-step status. |
| `regularize_schur` | `1e-12` | Initial regularization for the normal-matrix Schur solvers; unused by `QRSchur`. |
| `max_schur_reg` | `8` | Maximum number of regularization attempts. |
| `datasparsity` | `8` | Constraint matrix nonzero threshold for switching to dense products. |

For `PrimalDual`, success requires primal feasibility, dual feasibility, and the relative objective gap to satisfy `tol`. If the barrier parameter drops below `mu_min` before these criteria are met, the solver returns `MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL` rather than reporting success.

See [src/options.jl](src/options.jl) for all numerical options.

## Algorithmic components

Algorithmic components can also be selected using `set_attribute`. For example, to use the Helmberg–Kojima–Monteiro direction, Cholesky backtracking, and MadNLP's linear solver interface:

```julia
set_attribute(model, "direction", MadSDP.HKM())
set_attribute(model, "step", MadSDP.CholeskyBacktrack())
set_attribute(model, "schur", MadSDP.MadNLPSchur())
optimize!(model)
```

The available components are:

| Option | Default | Alternatives |
| --- | --- | --- |
| `algorithm` | `MadSDP.PrimalDual()` | `MadSDP.DualCentering()` |
| `direction` | `MadSDP.NT()` (Nesterov–Todd) | `MadSDP.HKM()` |
| `step` | `MadSDP.EigenStep()` | `MadSDP.CholeskyBacktrack()` |
| `centering` | `MadSDP.Mehrotra()` | `MadSDP.AdaptiveMu()` |
| `initial_point` | `MadSDP.ScaledIdentity()` for `PrimalDual` | `MadSDP.ScaledIdentity(scale)`, `MadSDP.DualShift()`, `MadSDP.InitialPoint(X, y; x)` |
| `schur` | `MadSDP.DenseCholesky()` | `MadSDP.MadNLPSchur(linear_solver)`, `MadSDP.QRSchur()` |

The `direction`, `step`, and `centering` options configure the default `PrimalDual` algorithm. If you set `algorithm` explicitly, configure these components in `MadSDP.PrimalDual(; direction, step, centering)` instead.

`DualCentering` uses `DualShift()` as its default initial point and currently supports only matrix blocks. `AdaptiveMu` is intended for a feasible starting point; `Mehrotra` is more robust for an infeasible start.

`ScaledIdentity()` chooses an initial scale from the problem data. Use `ScaledIdentity(scale)` to choose it explicitly: PSD primal and dual slack blocks start at `scale * I`, nonnegative scalar variables and their dual slacks start at `scale`, and equality multipliers start at zero. The scale must be finite and strictly positive. This controls the starting point, not the scaling of the problem itself.

`MadNLPSchur()` defaults to `MadNLP.LapackCPUSolver`. Pass a compatible `MadNLP.AbstractLinearSolver` type to select a different factorization backend.

`QRSchur()` factors the NT-scaled constraint matrix with sparse QR instead of forming its normal matrix. It recovers primal search directions by orthogonal projection to avoid cancellation in the normal-equation recovery formula. It requires `PrimalDual` with the `NT()` direction, `Float64` data, and linearly independent constraints. It allocates during sparse matrix assembly and factorization and does not use `regularize_schur` or `max_schur_reg`. Compare its runtime and numerical behavior with the normal-matrix solvers on your problem.

For example, to switch the model above from HKM to NT with sparse QR and an explicit initial scale:

```julia
set_attribute(model, "direction", MadSDP.NT())
set_attribute(model, "schur", MadSDP.QRSchur())
set_attribute(model, "initial_point", MadSDP.ScaledIdentity(1.0))
optimize!(model)
```

See [src/components.jl](src/components.jl) for the component interfaces and their extension methods.

## Custom matrices and model callbacks

MadSDP separates the interior-point algorithm from the computations that describe the SDP. [LowRankOpt](https://github.com/blegat/LowRankOpt.jl) translates JuMP/MathOptInterface models into an extension of the [NLPModels](https://jso.dev/NLPModels.jl/) interface for PSD matrix blocks. The solver uses this interface for constraint and adjoint products, objective data, and Schur complement assembly.

You can supply your own coefficient matrices `Aᵢ` with custom Julia types to preserve structure and provide efficient matrix operations. [`LowRankOpt.LinearCombinationInSet`](https://github.com/blegat/LowRankOpt.jl#low-rank-constraints) carries these representations through the modeling interface instead of expanding them into generic affine coefficients. For ordinary MOI PSD constraints, LowRankOpt automatically detects some useful structure, including rank-one and one-hot matrices. The `"detect_rank_one"` and `"detect_one_hot"` attributes control these detections; explicit matrix representations let you express structure beyond what automatic detection recognizes. Check `print_active_bridges(model)` to see whether the bridges preserve your representation.

You can also work directly at the model callback layer, without constructing a JuMP model. Implement or specialize the SDP-aware NLPModels callbacks used by MadSDP to compute constraint products, adjoint products, and Schur complements using your own formulas. Relevant operations include `LowRankOpt.add_jprod!`, `LowRankOpt.unsafe_jtprod`, and `LowRankOpt.schur_complement!`; see LowRankOpt's [model](https://github.com/blegat/LowRankOpt.jl/blob/main/src/model.jl) and [buffering](https://github.com/blegat/LowRankOpt.jl/blob/main/src/buffer.jl) interfaces. Custom models must satisfy the SDP block-access and buffering interfaces as well as the callbacks; implementing the scalar NLPModels interface alone is not sufficient.

Start with JuMP and automatic structure detection, preserve custom matrix types when useful, and specialize callbacks where profiling shows a bottleneck. These layers let application-specific computations share the same solver algorithm, with the formulation, matrix operations, and iteration logic remaining independently readable and reusable.

## MathOptInterface API

The optimizer uses `Float64` coefficients and supports affine objectives with either minimization or maximization sense.

Its native constraint types are:

 * `MOI.VectorAffineFunction{Float64}` in `MOI.Nonnegatives`
 * `MOI.VectorAffineFunction{Float64}` in `MOI.PositiveSemidefiniteConeTriangle`

Here, `MOI` is an alias for `MathOptInterface`. JuMP and MathOptInterface bridges reformulate scalar affine constraints, equalities, and PSD variables into these forms.

The optimizer supports `MOI.Silent()` and `MOI.RawOptimizerAttribute` for setting options. It does not currently provide infeasibility or unboundedness certificates.
