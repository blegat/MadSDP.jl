# MadSDP.jl

[![Build Status](https://github.com/blegat/MadSDP.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/blegat/MadSDP.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/blegat/MadSDP.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/blegat/MadSDP.jl)

[MadSDP.jl](https://github.com/blegat/MadSDP.jl) is a semidefinite programming
solver written in Julia. It solves semidefinite programs, including problems
with both positive semidefinite matrix variables and nonnegative scalar
variables, using interior-point methods.

MadSDP is a **malleable solver**, thanks to
[MadNLP](https://github.com/MadNLP/MadNLP.jl): it uses MadNLP's linear solver
interface and infrastructure, and exposes interchangeable algorithmic
components. You can choose the interior-point algorithm, search direction,
step-length rule, centering strategy, initial point, and Schur complement
solver, or implement new components in Julia.

MadSDP provides an interface to
[MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) through
[LowRankOpt](https://github.com/blegat/LowRankOpt.jl), so it can be used with
[JuMP](https://jump.dev).

## Getting help

If you need help, please ask a question on the
[JuMP community forum](https://jump.dev/forum).

If you have a reproducible example of a bug, please
[open a GitHub issue](https://github.com/blegat/MadSDP.jl/issues/new).

## Installation

MadSDP requires Julia 1.10 or later. Install it from GitHub using the Julia
package manager:

```julia
import Pkg
Pkg.add(url = "https://github.com/blegat/MadSDP.jl")
Pkg.add("JuMP")
```

## Use with JuMP

To use MadSDP with JuMP, use `MadSDP.Optimizer`. This example minimizes
`2X[1, 1] + X[2, 2]` over positive semidefinite matrices with trace one:

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
| `regularize_schur` | `1e-12` | Initial Schur complement regularization. |
| `max_schur_reg` | `8` | Maximum number of regularization attempts. |
| `datasparsity` | `8` | Constraint matrix nonzero threshold for switching to dense products. |

See [src/options.jl](src/options.jl) for all numerical options.

## Algorithmic components

Algorithmic components can also be selected using `set_attribute`. For example,
to use the Helmberg–Kojima–Monteiro direction, Cholesky backtracking, and
MadNLP's linear solver interface:

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
| `initial_point` | `MadSDP.ScaledIdentity()` for `PrimalDual` | `MadSDP.DualShift()`, `MadSDP.InitialPoint(X, y; x)` |
| `schur` | `MadSDP.DenseCholesky()` | `MadSDP.MadNLPSchur(linear_solver)` |

The `direction`, `step`, and `centering` options configure the default
`PrimalDual` algorithm. If you set `algorithm` explicitly, configure these
components in `MadSDP.PrimalDual(; direction, step, centering)` instead.

`DualCentering` uses `DualShift()` as its default initial point and currently
supports only matrix blocks. `AdaptiveMu` is intended for a feasible starting
point; `Mehrotra` is more robust for an infeasible start.

`MadNLPSchur()` defaults to `MadNLP.LapackCPUSolver`. Pass a compatible
`MadNLP.AbstractLinearSolver` type to select a different factorization backend.

See [src/components.jl](src/components.jl) for the component interfaces and
their extension methods.

## MathOptInterface API

The optimizer uses `Float64` coefficients and supports affine objectives with
either minimization or maximization sense.

Its native constraint types are:

 * `MOI.VectorAffineFunction{Float64}` in `MOI.Nonnegatives`
 * `MOI.VectorAffineFunction{Float64}` in `MOI.PositiveSemidefiniteConeTriangle`

Here, `MOI` is an alias for `MathOptInterface`. JuMP and MathOptInterface
bridges reformulate scalar affine constraints, equalities, and PSD variables
into these forms.

The optimizer supports `MOI.Silent()` and `MOI.RawOptimizerAttribute` for
setting options. It does not currently provide infeasibility or unboundedness
certificates.
