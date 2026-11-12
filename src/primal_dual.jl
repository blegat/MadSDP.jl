# The primal-dual path-following algorithm `PrimalDual`, see also `step.jl`
# and `directions.jl`.

algorithm_state(::PrimalDual, ::Type, _) = nothing

default_initial_point(::PrimalDual) = ScaledIdentity()

schur_scaling(::PrimalDual, b::SDPBlock) = schur_scaling(b.work.dir, b)

function initialize_centering!(solver, ::Mehrotra)
    solver.mu = total_complementarity(solver) / max(_barrier_dim(solver), 1)
    return
end

function initialize_centering!(solver, c::AdaptiveMu)
    solver.mu =
        c.initial_ratio * total_complementarity(solver) / max(_barrier_dim(solver), 1)
    return
end

function initialize_algorithm!(solver, a::PrimalDual)
    update_residuals!(solver)
    initialize_centering!(solver, a.centering)
    return
end

# Computes everything that depends only on the current iterate: the
# factorizations of the blocks and the scalings of the Schur complement.
function _prepare!(solver::MadSDPSolver{T}, a::PrimalDual) where {T}
    for b in solver.blocks
        prepare_block!(b) || return false
        prepare_direction!(b.work.dir, b) || return false
        prepare_step!(a.step, b) || return false
    end
    solver.zinv .= inv.(solver.z)
    return true
end

# Solves the Schur system targeting `μ S⁻¹` and computes the directions.
function _solve_directions!(solver::MadSDPSolver{T}, μ::T, corrected::Bool) where {T}
    rhs!(solver.dely, solver, μ, corrected)
    schur_solve!(solver.schur, solver.dely)
    directions!(solver, solver.dely, μ, corrected)
    return
end

function _take_step!(solver::MadSDPSolver{T}, α::T, β::T) where {T}
    solver.alpha, solver.beta = α, β
    for b in solver.blocks
        @. b.X += α * b.work.delX
        _symmetrize!(b.X)
        @. b.S += β * b.work.delS
        _symmetrize!(b.S)
    end
    @. solver.y += β * solver.dely
    if solver.n_scalar > 0
        @. solver.x += α * solver.delx
        @. solver.z += β * solver.delz
    end
    solver.iter += 1
    update_residuals!(solver)
    return
end

function _start_iteration!(solver, a::PrimalDual)
    if !_prepare!(solver, a) || !_factorize_schur!(solver)
        solver.status = MadNLP.ERROR_IN_STEP_COMPUTATION
        return false
    end
    return true
end

iterate!(solver::MadSDPSolver, a::PrimalDual) = iterate!(solver, a, a.centering)

# One Mehrotra predictor-corrector iteration.
function iterate!(solver::MadSDPSolver{T}, a::PrimalDual, c::Mehrotra) where {T}
    _start_iteration!(solver, a) || return
    μ = total_complementarity(solver) / max(_barrier_dim(solver), 1)
    solver.mu = μ
    # Predictor
    _solve_directions!(solver, zero(T), false)
    α_aff, β_aff = step_lengths(a.step, solver)
    μ_aff = affine_complementarity(solver, α_aff, β_aff)
    σ = clamp((μ_aff / max(μ, eps(T)))^T(c.σ_exponent), T(1e-8), one(T))
    solver.sigma = σ
    correction!(solver)
    # Corrector
    _solve_directions!(solver, σ * μ, true)
    α, β = step_lengths(a.step, solver)
    _take_step!(solver, α, β)
    return
end

# One iteration of `mc_psd` [HRVW96].
function iterate!(solver::MadSDPSolver{T}, a::PrimalDual, c::AdaptiveMu) where {T}
    _start_iteration!(solver, a) || return
    _solve_directions!(solver, solver.mu, false)
    α, β = step_lengths(a.step, solver)
    _take_step!(solver, α, β)
    μ = c.ratio * total_complementarity(solver) / max(_barrier_dim(solver), 1)
    if min(α, β) < T(0.5)
        μ *= T(1.5)
    end
    if α + β > T(1.6)
        μ *= T(0.75)
    end
    if α + β > T(1.9)
        μ /= 1 + T(0.1) * solver.iter
    end
    solver.mu = μ
    return
end

function update_termination!(solver::MadSDPSolver{T}, ::PrimalDual) where {T}
    pobj = primal_objective(solver)
    dobj = dual_objective(solver)
    solver.obj_val = pobj
    _update_infeasibilities!(solver)
    solver.inf_compl = abs(pobj - dobj) / max(T(1), abs(pobj) + abs(dobj))
    converged = max(solver.inf_pr, solver.inf_du, solver.inf_compl) <= solver.opt.tol
    _update_status!(solver, converged)
    if solver.status == MadNLP.REGULAR && solver.mu < solver.opt.mu_min
        solver.status = MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL
    end
    return
end
