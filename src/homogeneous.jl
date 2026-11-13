# Homogeneous primal-dual embedding. The cone Newton matrix is unchanged:
# a scalar Schur complement adds tau and kappa using one extra solve.
mutable struct HomogeneousState{T}
    tau::T
    kappa::T
    delta_tau::T
    delta_kappa::T
    correction::T
    gap_residual::T
    denominator::T
    residual_factor::T
    coupling::Vector{T}
    coupling_solution::Vector{T}
end
function algorithm_state(::HomogeneousPrimalDual, ::Type{T}, n) where {T}
    return HomogeneousState(
        one(T),
        one(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        one(T),
        one(T),
        T[],
        T[],
    )
end
block_workspace(a::HomogeneousPrimalDual, T, n) = block_workspace(a.inner, T, n)
schur_scaling(a::HomogeneousPrimalDual, b) = schur_scaling(a.inner, b)
default_initial_point(::HomogeneousPrimalDual) = ScaledIdentity(1.0)
function check_schur(component::RootKKT, a::HomogeneousPrimalDual)
    if !(a.inner.centering isa Mehrotra)
        throw(ArgumentError("HomogeneousPrimalDual requires Mehrotra centering."))
    end
    return check_schur(component, a.inner)
end
_prepare!(solver, a::HomogeneousPrimalDual) = _prepare!(solver, a.inner)
cone_data_scale(state::HomogeneousState) = state.tau
homogeneous_tau_direction(state::HomogeneousState) = state.delta_tau
homogeneous_residual_factor(state::HomogeneousState) = state.residual_factor

function initialize_algorithm!(solver, a::HomogeneousPrimalDual)
    state=solver.state
    state.tau=state.kappa=one(eltype(solver.y))
    state.delta_tau=state.delta_kappa=state.correction=zero(state.tau)
    state.residual_factor=one(state.tau)
    resize!(state.coupling, length(solver.schur.rhs))
    resize!(state.coupling_solution, length(solver.schur.rhs))
    update_residuals!(solver)
    solver.mu=(total_complementarity(solver)+state.tau*state.kappa)/(_barrier_dim(solver)+1)
    return
end

function _homogeneous_coupling!(solver)
    ws=solver.schur
    state=solver.state
    T=eltype(solver.y)
    ws.rhs[1:solver.m] .= -LRO.cons_constant(solver.model)
    ws.rhs[(solver.m+1):ws.cone_offset] .= -ws.equality.target
    offset=ws.cone_offset
    for (i, b) in enumerate(solver.blocks)
        _copy_to_dense!(b.buf2, LRO.grad(solver.model, LRO.MatrixIndex(i)))
        if !ws.psd_cone_hessian[i]
            LinearAlgebra.mul!(b.buf, b.buf2, b.work.dir.G)
            LinearAlgebra.mul!(b.buf2, b.work.dir.G', b.buf)
        end
        for q = 1:b.n, p = 1:q
            offset+=1
            ws.rhs[offset]=-b.buf2[p, q]*(p==q ? one(T) : sqrt(T(2)))
        end
    end
    for (i, b) in enumerate(ws.soc)
        if ws.soc_cone_hessian[i]
            ws.rhs[(offset+1):(offset+length(b.X))] .= -b.constant
        else
            LinearAlgebra.mul!(b.work1, b.G, b.constant)
            ws.rhs[(offset+1):(offset+length(b.X))] .= -b.work1
        end
        offset+=length(b.X)
    end
    for k = 1:solver.n_scalar
        ws.rhs[offset+k]=-sqrt(solver.x[k]*solver.zinv[k])*_inner(solver.model).d_lin[k]
    end
    state.coupling .= -ws.rhs
    state.coupling[1:solver.m] .*= -1
    _base_root_linear_solve!(ws, solver)
    copyto!(state.coupling_solution, ws.rhs)
    state.denominator=state.kappa/state.tau+LinearAlgebra.dot(
        state.coupling,
        state.coupling_solution,
    )
    return isfinite(state.denominator) && state.denominator > 0
end

function _root_linear_solve!(ws, solver, state::HomogeneousState, mu, corrected)
    _base_root_linear_solve!(ws, solver)
    correction=corrected ? state.correction : zero(mu)
    state.delta_tau=(
        state.residual_factor*(primal_objective(solver)-dual_objective(solver))+(
            state.residual_factor-1
        )*state.kappa+(mu-correction)/state.tau+LinearAlgebra.dot(state.coupling, ws.rhs)
    )/state.denominator
    state.delta_kappa=(mu-state.tau*state.kappa-correction-state.kappa*state.delta_tau)/state.tau
    ws.rhs .-= state.delta_tau .* state.coupling_solution
    return
end

step_fraction(step::EigenStep) = step.τ
step_fraction(step::CholeskyBacktrack) = step.boundary
function _homogeneous_step(solver, step)
    alpha, beta=step_lengths(step, solver)
    state=solver.state
    if state.delta_tau < 0
        alpha=min(alpha, -step_fraction(step)*state.tau/state.delta_tau)
    end
    if state.delta_kappa < 0
        beta=min(beta, -step_fraction(step)*state.kappa/state.delta_kappa)
    end
    return min(alpha, beta)
end
function iterate!(solver::MadSDPSolver{T}, a::HomogeneousPrimalDual) where {T}
    _start_iteration!(solver, a.inner) || return
    if !_homogeneous_coupling!(solver)
        solver.status=MadNLP.ERROR_IN_STEP_COMPUTATION
        return
    end
    state=solver.state
    mu=(total_complementarity(solver)+state.tau*state.kappa)/(_barrier_dim(solver)+1)
    solver.mu=mu
    state.residual_factor=one(T)
    solve_directions!(solver.schur, solver, zero(T), false)
    if !isfinite(state.delta_tau) ||
       !isfinite(state.delta_kappa) ||
       !all(isfinite, solver.dely)
        solver.status=MadNLP.ERROR_IN_STEP_COMPUTATION
        return
    end
    alpha=_homogeneous_step(solver, a.inner.step)
    affine=affine_complementarity(solver, alpha, alpha)*_barrier_dim(solver) +
           (state.tau+alpha*state.delta_tau)*(state.kappa+alpha*state.delta_kappa)
    affine/=(_barrier_dim(solver)+1)
    sigma=clamp((affine/max(mu, eps(T)))^T(a.inner.centering.σ_exponent), T(1e-8), one(T))
    solver.sigma=sigma
    correction!(solver)
    state.correction=state.delta_tau*state.delta_kappa
    state.residual_factor=one(T)-sigma
    solve_directions!(solver.schur, solver, sigma*mu, true)
    if !isfinite(state.delta_tau) ||
       !isfinite(state.delta_kappa) ||
       !all(isfinite, solver.dely)
        solver.status=MadNLP.ERROR_IN_STEP_COMPUTATION
        return
    end
    alpha=_homogeneous_step(solver, a.inner.step)
    state.tau+=alpha*state.delta_tau
    state.kappa+=alpha*state.delta_kappa
    _take_step!(solver, alpha, alpha)
    return
end

function update_termination!(solver::MadSDPSolver{T}, ::HomogeneousPrimalDual) where {T}
    state=solver.state
    primal, dual=primal_objective(solver), dual_objective(solver)
    state.gap_residual=primal-dual+state.kappa
    _update_infeasibilities!(solver)
    solver.inf_pr/=state.tau
    solver.inf_du/=state.tau
    solver.inf_compl=abs(primal-dual)/max(state.tau, abs(primal)+abs(dual))
    solver.obj_val=primal/state.tau
    converged=max(solver.inf_pr, solver.inf_du, solver.inf_compl)<=solver.opt.tol
    _update_status!(solver, converged)
    if !converged && primal < 0
        data=LRO.cons_constant(solver.model)
        residual=maximum(
            (abs(state.tau*data[j]-solver.Rp[j]) for j in eachindex(data));
            init = zero(T),
        )
        if residual/(-primal)<=T(1e-8)
            solver.status=MadNLP.INFEASIBLE_PROBLEM_DETECTED
        end
    end
    if !converged && state.tau < sqrt(solver.opt.tol)*state.kappa
        scale=LRO.recession_certificate(solver.schur.equality.model, solver.y)
        if !isnothing(scale)
            solver.status=MadNLP.DIVERGING_ITERATES
        end
    end
    if solver.status==MadNLP.REGULAR &&
       min(solver.alpha, solver.beta)<=eps(T) &&
       solver.iter>0
        solver.status=MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL
    end
    return
end

function finalize_algorithm!(solver, ::HomogeneousPrimalDual)
    if solver.status!=MadNLP.SOLVE_SUCCEEDED
        return
    end
    state=solver.state
    scale=state.tau
    solver.y ./= scale
    solver.x ./= scale
    solver.z ./= scale
    solver.schur.equality.eta ./= scale
    for b in solver.blocks
        b.X ./= scale
        b.S ./= scale
    end
    for b in solver.schur.soc
        b.X ./= scale
        b.S ./= scale
    end
    state.kappa/=scale
    state.tau=one(scale)
    update_residuals!(solver)
    update_termination!(solver, solver.algorithm)
    return
end
