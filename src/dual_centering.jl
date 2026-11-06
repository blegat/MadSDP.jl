# Dual predictor-corrector method `DualCentering` after Nesterov (2025)
# "Asymmetric Long-Step Primal-Dual Interior-Point Methods with Dual
# Centering" — Algorithm (7.13).
#
# The method carries only `y` and the dual slacks `S_i = C_i - A*_i(y)`.
# Every outer step is either:
#   b) a damped Newton step on  ψ_t(y) = ζ(y) - t·⟨b,y⟩  with t fixed, or
#   c) a single predictor step that also advances t.
# ζ(y) = -Σ_i log det S_i is the dual log-det barrier; ν = Σ_i n_i.
# The Hessian ∇²ζ(y) is the Schur complement with `L = R = S⁻¹`.

mutable struct DualCenteringState{T}
    d::Vector{T}                 # B·g  damped-Newton direction
    g::Vector{T}                 # ∇ψ_t(y) = ∇ζ(y) - t·b
    grad_zeta::Vector{T}         # ∇ζ(y) = A(S^{-1})
    Δy::Vector{T}                # t·B·b  predictor direction
    yhat::Vector{T}              # ŷ = y + d
    ν::Int                       # barrier parameter Σ n_i
    A::T                         # proximity bound of the predictor
    t::T
    λ::T                         # Newton decrement
    n_predictor::Int
    n_newton::Int
end

function algorithm_state(::DualCentering, ::Type{T}, m) where {T}
    vec() = zeros(T, m)
    return DualCenteringState(
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        0,
        zero(T),
        zero(T),
        zero(T),
        0,
        0,
    )
end

# Workspace of `DualCentering` for one block:
#   Shat         slack at ŷ = y + d
#   D            A*(Δy)
#   eigvals      eigenvalues of Ŝ^{-1/2} D Ŝ^{-1/2}
#   eigen        LAPACK workspace to compute them
struct DualCenteringBlock{T,E}
    Shat::Matrix{T}
    D::Matrix{T}
    eigvals::Vector{T}
    eigen::E
end

function block_workspace(::DualCentering, ::Type{T}, n) where {T}
    return DualCenteringBlock(
        zeros(T, n, n),
        zeros(T, n, n),
        zeros(T, n),
        LRO.EigenWorkspace(zeros(T, n, n)),
    )
end

default_initial_point(::DualCentering) = DualShift()

schur_scaling(::DualCentering, b::SDPBlock) = (b.Si, b.Si)

function check_model(::DualCentering, model)
    if LRO.num_scalars(model) > 0
        error("MadSDP: `DualCentering` does not support scalar (LP) blocks yet.")
    end
    if iszero(LRO.num_matrices(model))
        error("MadSDP: `DualCentering` needs at least one PSD block.")
    end
    return
end

# ω_*(τ) = -τ - log(1-τ) for τ < 1.
_omega_star(τ) = -τ - log1p(-τ)

# Sets `S = C - A*(y)` and factorizes it. Returns `false` if `S` is not
# positive definite.
function _refresh_slacks!(solver::MadSDPSolver)
    _set_dual_slacks!(solver)
    return all(prepare_block!, solver.blocks)
end

# ∇ζ(y)_j = Σ_i ⟨A_{i,j}, S_i^{-1}⟩ ;  ∇ψ_t = ∇ζ - t b.
function _compute_gradient!(solver::MadSDPSolver{T}) where {T}
    st = solver.state
    fill!(st.grad_zeta, zero(T))
    for (i, b) in enumerate(solver.blocks)
        LRO.add_jprod!(solver.model, b.Si, st.grad_zeta, LRO.MatrixIndex(i))
    end
    bvec = LRO.cons_constant(solver.model)
    @. st.g = st.grad_zeta - st.t * bvec
    return
end

# Factorizes ∇²ζ(y) and computes the Newton direction `d = B g` and the
# Newton decrement `λ = √(g' d)`.
function _newton_direction!(solver::MadSDPSolver{T}) where {T}
    st = solver.state
    _compute_gradient!(solver)
    _factorize_schur!(solver) || return false
    copyto!(st.d, st.g)
    schur_solve!(solver.schur, st.d)
    st.λ = sqrt(max(LinearAlgebra.dot(st.g, st.d), zero(T)))
    return true
end

function initialize_algorithm!(solver::MadSDPSolver{T}, a::DualCentering) where {T}
    st = solver.state
    st.ν = total_dim(solver.blocks)
    st.A = isnothing(a.A) ? T(a.δ + _omega_star(2 * a.β / (1 - a.β)^2)) : T(a.A)
    st.n_predictor = st.n_newton = 0
    _refresh_slacks!(solver) ||
        error("MadSDP: the initial `y` does not give a dual slack `S ≻ 0`.")
    # Pick t₀ as the least-squares best fit minimising ‖∇ζ(y₀) - t·b‖²_{y₀}.
    # For F_y unbounded this is essential — pure ζ-centering would diverge.
    st.t = zero(T)
    _newton_direction!(solver) || error("MadSDP: failed to factorize ∇²ζ at y₀.")
    # Now `d = B ∇ζ` since `t = 0`; `Δy = B b`.
    bvec = LRO.cons_constant(solver.model)
    copyto!(st.Δy, bvec)
    schur_solve!(solver.schur, st.Δy)
    bBb = max(LinearAlgebra.dot(bvec, st.Δy), eps(T))
    bBg = LinearAlgebra.dot(bvec, st.d)
    st.t = max(bBg / bBb, T(a.β) / sqrt(bBb))
    # Damped Newton on ψ_{t₀} until λ ≤ β/2.
    for _ = 1:(a.centering_max_iter)
        _newton_direction!(solver) ||
            error("MadSDP: failed to factorize ∇²ζ during the initial centering.")
        st.λ ≤ a.β / 2 && break
        @. solver.y -= st.d / (1 + st.λ)
        solver.iter += 1
        st.n_newton += 1
        _refresh_slacks!(solver) ||
            error("MadSDP: damped Newton left the interior during the initial centering.")
    end
    solver.mu = one(T) / st.t
    return
end

# ξ(α) = Σ_i Σ_p [-log(1 - α μ_{i,p}) - log(1 + (α/(1-α)) μ_{i,p})]
function _xi_value(blocks, α::T) where {T}
    s = zero(T)
    β2 = α / (1 - α)
    for b in blocks
        @inbounds for μ in b.work.eigvals
            u = 1 - α * μ
            v = 1 + β2 * μ
            (u > 0 && v > 0) || return T(Inf)
            s -= log(u)
            s -= log(v)
        end
    end
    return s
end

# Largest α ∈ (0, 1) s.t. all log arguments remain positive.
function _xi_alpha_max(blocks, ::Type{T}) where {T}
    αmax = one(T)
    for b in blocks
        @inbounds for μ in b.work.eigvals
            if μ > zero(T)
                αmax = min(αmax, one(T) / μ)
            end
            if μ < zero(T)
                # 1 + (α/(1-α))μ > 0  ⇔  α < 1/(1 - μ)
                αmax = min(αmax, one(T) / (1 - μ))
            end
        end
    end
    return αmax
end

# Bisection on ξ(α) = A.  ξ is increasing in α on (0, αmax) (sum of - log(1- ·)
# terms, each monotone), and ξ(0)=0, ξ(αmax⁻) = +∞, so the root exists when A>0.
function _solve_alpha(blocks, A::T) where {T}
    αmax = _xi_alpha_max(blocks, T)
    hi = αmax * T(0.99999)
    ξhi = _xi_value(blocks, hi)
    # Degenerate case: ξ never reaches A on (0, αmax) — the proximity
    # constraint is slack along the predictor direction.  Take the
    # largest interior step we can.
    if !isfinite(ξhi) || ξhi < A
        return hi
    end
    lo = zero(T)
    for _ = 1:80
        mid = (lo + hi) / 2
        if _xi_value(blocks, mid) < A
            lo = mid
        else
            hi = mid
        end
        (hi - lo) ≤ eps(T) * max(one(T), hi) && break
    end
    return (lo + hi) / 2
end

# Eigenvalues of  Ŝ^{-1/2} D Ŝ^{-1/2}, computed as the eigenvalues of the
# symmetric matrix  L̂⁻¹ D L̂⁻ᵀ  where  Ŝ = L̂ L̂ᵀ.
function _proximity_eigvals!(b::SDPBlock)
    w = b.work
    _cholesky_factor!(b.buf, w.Shat) || return false
    L = LinearAlgebra.LowerTriangular(b.buf)
    copyto!(b.buf2, w.D)
    LinearAlgebra.ldiv!(L, b.buf2)
    LinearAlgebra.rdiv!(b.buf2, L')
    _symmetrize!(b.buf2)
    copyto!(w.eigvals, first(LRO.eigen!(w.eigen, b.buf2)))
    return true
end

# ‖ŝ‖²_{s_k} = Σ_i tr(S_i^{-1} Ŝ_i S_i^{-1} Ŝ_i) = Σ_i ‖L_s^{-1} Ŝ L_s^{-T}‖_F²
# where `Ls` is the Cholesky factor of `S` at `y_k`.
function _shat_norm2(blocks, ::Type{T}) where {T}
    s = zero(T)
    for b in blocks
        L = LinearAlgebra.LowerTriangular(b.Ls)
        copyto!(b.buf, b.work.Shat)
        LinearAlgebra.ldiv!(L, b.buf)
        LinearAlgebra.rdiv!(b.buf, L')
        s += sum(abs2, b.buf)
    end
    return s
end

# Predictor step from `y_k`, at which `d` is the centered direction.
function _predictor_step!(solver::MadSDPSolver{T}) where {T}
    st = solver.state
    model = solver.model
    # ŷ = y + d
    @. st.yhat = solver.y + st.d
    # Δy = t · B · b, with the Schur complement factorized at `y_k`.
    copyto!(st.Δy, LRO.cons_constant(model))
    schur_solve!(solver.schur, st.Δy)
    st.Δy .*= st.t
    for (i, b) in enumerate(solver.blocks)
        _slack!(b.work.Shat, model, st.yhat, i)
        fill!(b.work.D, zero(T))
        AT = LRO.unsafe_jtprod(model, st.Δy, LRO.MatrixIndex(i))
        _axpy_sparse_dense!(b.work.D, one(T), AT)
        _symmetrize!(b.work.D)
        _proximity_eigvals!(b) || return false
    end
    shat_n2 = _shat_norm2(solver.blocks, T)
    α = _solve_alpha(solver.blocks, st.A)
    solver.alpha = solver.beta = α
    # Primal estimate X = (1/t) S⁻¹ ((1-α) Ŝ + α D) S⁻¹ at `y_k`.
    for b in solver.blocks
        @. b.buf = (1 - α) * b.work.Shat + α * b.work.D
        LinearAlgebra.mul!(b.buf2, b.Si, b.buf)
        LinearAlgebra.mul!(b.X, b.buf2, b.Si, inv(st.t), false)
        _symmetrize!(b.X)
    end
    # y_{k+1} = ŷ + α · Δy ;  t_{k+1} = ν · t_k / ((1-α) · ‖ŝ‖²_{s_k}).
    @. solver.y = st.yhat + α * st.Δy
    st.t = st.ν * st.t / ((1 - α) * shat_n2)
    st.n_predictor += 1
    return true
end

function iterate!(solver::MadSDPSolver{T}, a::DualCentering) where {T}
    st = solver.state
    if !_newton_direction!(solver)
        solver.status = MadNLP.ERROR_IN_STEP_COMPUTATION
        return
    end
    if st.λ > a.β
        # b) damped Newton, t unchanged.
        @. solver.y -= st.d / (1 + st.λ)
        solver.alpha = solver.beta = inv(1 + st.λ)
        st.n_newton += 1
    elseif !_predictor_step!(solver)
        # c) predictor
        solver.status = MadNLP.ERROR_IN_STEP_COMPUTATION
        return
    end
    solver.iter += 1
    if !_refresh_slacks!(solver)
        solver.status = MadNLP.ERROR_IN_STEP_COMPUTATION
        return
    end
    solver.mu = inv(st.t)
    return
end

# Converged when the bound `ν / t` on the duality gap of the central path,
# relative to the objectives as for `PrimalDual`, is below the tolerance.
function update_termination!(solver::MadSDPSolver{T}, ::DualCentering) where {T}
    st = solver.state
    pobj = primal_objective(solver)
    dobj = dual_objective(solver)
    solver.obj_val = pobj
    update_residuals!(solver)
    _update_infeasibilities!(solver)
    solver.inf_compl = st.ν / max(st.t, eps(T)) / max(one(T), abs(pobj) + abs(dobj))
    _update_status!(solver, solver.inf_compl <= solver.opt.tol)
    return
end
