# Algorithmic components of the primal-dual interior-point method.
#
# `MadSDPSolver` is parametrized by one value of each of the following
# abstract types. The solver specializes on their types, so choosing a
# component has no runtime cost, and a new variant only needs a new
# subtype and the methods documented with its abstract type.

"""
    AbstractDirection

Search direction, i.e., how the complementarity `X S = μ I` is linearized.
With `K = μ S⁻¹`, every direction computes
```
ΔS = Rd - A*(Δy)
ΔX = K - X - E(ΔS) + corr
```
for a linear operator `E` and a second-order (Mehrotra) correction `corr`.
The Schur complement is then `H[j, k] = ⟨Aⱼ, L Aₖ R⟩` for the matrices
`(L, R)` such that `⟨A, E(B)⟩ = ⟨A, L B R⟩` for symmetric `A`, `B`.

A subtype `D` implements
* `direction_workspace(D, T, n)`,
* `prepare_direction!(workspace, block)` returning `false` on failure,
* `schur_scaling(workspace, block)` returning `(L, R)`,
* `apply_E!(out, workspace, block, M)`,
* `correction!(corr, workspace, block)` from `block.delX` and
  `block.delS`, which hold the affine-scaling direction.
"""
abstract type AbstractDirection end

"""
    NT()

Nesterov-Todd direction: `E(M) = W M W` where `W S W = X`. It needs an SVD
per PSD block and per iteration to compute `W`.
"""
struct NT <: AbstractDirection end

"""
    HKM()

Helmberg-Kojima-Monteiro direction, also known as the HRVW/KSH/M direction:
`E(M) = (X M S⁻¹ + S⁻¹ M X) / 2`. It only needs `S⁻¹`, so it is cheaper
than [`NT`](@ref) per iteration, and its Schur complement for one-hot
constraints is the Hadamard product `S⁻¹ ∘ X` used by `mc_psd` [HRVW96].

[HRVW96] Helmberg, Christoph, Franz Rendl, Robert J. Vanderbei, and Henry
Wolkowicz. "An interior-point method for semidefinite programming." SIAM
Journal on Optimization 6, no. 2 (1996): 342-361.
"""
struct HKM <: AbstractDirection end

"""
    AbstractStepLength

How the step lengths `α` (primal) and `β` (dual) are computed. A subtype `S`
implements
* `prepare_step!(S, block)` called once per iteration,
* `step_lengths(S, solver)` returning `(α, β)`.
"""
abstract type AbstractStepLength end

"""
    EigenStep(τ = 0.95)

Takes the fraction `τ` of the maximal step keeping `X` and `S` positive
definite, computed from the smallest eigenvalue of `L⁻¹ ΔX L⁻ᵀ` where
`X = L Lᵀ`, capped at `1`.
"""
struct EigenStep{T} <: AbstractStepLength
    τ::T
end
EigenStep() = EigenStep(0.95)

"""
    CholeskyBacktrack(shrink = 0.8, boundary = 0.95)

Starts from a step of `1` and multiplies it by `shrink` until `X + α ΔX`
has a Cholesky factorization. If the step was shortened, it is multiplied
by `boundary` to stay away from the boundary. This is the line search of
`mc_psd` [HRVW96]. It needs no eigenvalue decomposition, so it is cheaper
than [`EigenStep`](@ref) when few factorizations are needed.
"""
struct CholeskyBacktrack{T} <: AbstractStepLength
    shrink::T
    boundary::T
end
CholeskyBacktrack() = CholeskyBacktrack(0.8, 0.95)

"""
    AbstractCentering

How the barrier parameter is chosen and how many linear systems are solved
per iteration. A subtype `C` implements `iterate!(solver, C)` and
`initialize_centering!(solver, C)`.
"""
abstract type AbstractCentering end

"""
    Mehrotra(σ_exponent = 3)

Mehrotra predictor-corrector: an affine-scaling predictor gives the
centering parameter `σ = (μ_aff / μ)^σ_exponent` and a second-order
correction for the corrector. Two linear solves per iteration with the same
factorization.
"""
struct Mehrotra{T} <: AbstractCentering
    σ_exponent::T
end
Mehrotra() = Mehrotra(3.0)

"""
    AdaptiveMu(initial_ratio = 1/4, ratio = 1/2)

One linear solve per iteration targeting `μ`, which is updated after each
step from `ratio * ⟨X, S⟩ / N` and the step lengths. This is the strategy
of `mc_psd` [HRVW96], `N` being the sum of the sizes of the blocks. The
initial `μ` is `initial_ratio * ⟨X, S⟩ / N`.

It is designed for a feasible start such as the [`InitialPoint`](@ref) of
`mc_psd`. From an infeasible start, the step lengths may stay below `1` so
`μ` only decreases linearly, and the Schur complement can become too
ill-conditioned to reach a tight tolerance; [`Mehrotra`](@ref) is more
robust in that case.
"""
struct AdaptiveMu{T} <: AbstractCentering
    initial_ratio::T
    ratio::T
end
AdaptiveMu() = AdaptiveMu(0.25, 0.5)

"""
    AbstractInitialPoint

Starting point. A subtype `I` implements `initial_point!(solver, I)`.
"""
abstract type AbstractInitialPoint end

"""
    ScaledIdentity()

`X = S = β I` and `y = 0` with `β` scaled from the norms of the data.
"""
struct ScaledIdentity <: AbstractInitialPoint end

"""
    InitialPoint(X, y; x = nothing)

Starts from the PSD blocks `X`, the multipliers `y` and the dual slacks
`S = C - A*(y)`, so the dual residual is zero if `S` is positive definite.
The scalar variables `x` default to ones and `z = d - C_linᵀ y`.
For instance, `X = I` and `y = -(1 .+ vec(sum(abs, L, dims = 1)))` is a
feasible start for the max-cut relaxation `min ⟨-L, X⟩, diag(X) = 1`.
"""
struct InitialPoint{T,V} <: AbstractInitialPoint
    X::Vector{Matrix{T}}
    y::Vector{T}
    x::V
end
function InitialPoint(X::Vector{<:AbstractMatrix}, y; x = nothing)
    T = eltype(y)
    return InitialPoint(Matrix{T}.(X), collect(y), x)
end

"""
    AbstractSchurSolver

Factorization of the Schur complement. A subtype `S` implements
`schur_workspace(S, T, m)`, `factorize!(workspace) -> Bool` and
`solve!(workspace, rhs)`, the matrix being `workspace.H`.
"""
abstract type AbstractSchurSolver end

"""
    DenseCholesky()

LAPACK Cholesky factorization of the dense Schur complement.
"""
struct DenseCholesky <: AbstractSchurSolver end

"""
    MadNLPSchur(linear_solver = MadNLP.LapackCPUSolver)

Factorization with a `MadNLP.AbstractLinearSolver`, e.g., to use a GPU.
"""
struct MadNLPSchur <: AbstractSchurSolver
    linear_solver::Type
end
MadNLPSchur() = MadNLPSchur(MadNLP.LapackCPUSolver)
