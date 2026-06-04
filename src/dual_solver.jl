# Dual Predictor-Corrector IPM for SDP, after Nesterov (2025)
# "Asymmetric Long-Step Primal-Dual Interior-Point Methods with Dual
# Centering" — Algorithm (7.13).
#
# Only matrix (PSD) blocks are supported in this v1; scalar/LP blocks raise
# an error.  The method carries only `y` and the dual slacks `S_i = C_i -
# A*_i(y)`.  Every outer step is either:
#   b) a damped Newton step on  ψ_t(y) = ζ(y) - t·⟨b,y⟩  with t fixed, or
#   c) a single predictor step that also advances t.
# ζ(y) = -Σ_i log det S_i is the dual log-det barrier; ν = Σ_i n_i.
#
# Reused machinery from the primal-dual NT solver:
#   * `SchurSystem` / `build_and_factorize!` — passing W = S^{-1} per block
#     gives precisely ∇²ζ(y).
#   * `LRO.unsafe_jtprod(model, y, MatrixIndex(i))` returns A*_i(y).
#   * `LRO.add_jprod!(model, M, h, MatrixIndex(i))` accumulates A_i(M) into h.

mutable struct DualSDPBlock{T}
    n::Int
    S::Matrix{T}           # dual slack  C_i - A*_i(y)
    Si::Matrix{T}           # S^{-1}
    Ls::LinearAlgebra.LowerTriangular{T,Matrix{T}}  # Cholesky factor of S
    Shat::Matrix{T}         # slack at ŷ = y + d (predictor)
    Dmat::Matrix{T}         # A*_i(Δy) used in proximity
    Xprim::Matrix{T}        # primal proxy X_{k+1} = (1/t) S^{-1}((1-α)Ŝ + αD)S^{-1}
    eigvals::Vector{T}      # eigenvalues of S_hat^{-1/2} D S_hat^{-1/2}
    buf_n::Matrix{T}
    buf_n2::Matrix{T}
end

function DualSDPBlock{T}(n::Integer) where {T}
    nint = Int(n)
    return DualSDPBlock{T}(
        nint,
        Matrix{T}(undef, nint, nint),
        Matrix{T}(undef, nint, nint),
        LinearAlgebra.LowerTriangular(Matrix{T}(undef, nint, nint)),
        Matrix{T}(undef, nint, nint),
        Matrix{T}(undef, nint, nint),
        zeros(T, nint, nint),
        Vector{T}(undef, nint),
        Matrix{T}(undef, nint, nint),
        Matrix{T}(undef, nint, nint),
    )
end

@kwdef mutable struct MadDualSDPOptions{T}
    tol::T = T(1e-7)
    max_iter::Int = 500
    max_wall_time::T = T(3600)
    # Algorithm parameters.  Nesterov's numerical experiments (§9) use
    #   β = 0.2,  A = 2  (and these violate (7.11) — the theoretical bound
    # is conservative).  We default to the same.  If `A` is not provided we
    # fall back to the formula  A = δ + ω*(2β/(1-β)²)  (7.11), which gives
    # the theoretical guarantee when κ̂ > 0.
    β::T = T(0.2)
    δ::T = T(1)
    A::Union{Nothing,T} = T(2)
    # Initial point: if `nothing`, try y₀ = -ρ·1 with ρ doubled until S ≻ 0.
    initial_y::Union{Nothing,Vector{T}} = nothing
    initial_shift::T = T(1)
    initial_shift_max::T = T(1e10)
    # Damped-Newton inner-loop tolerance is implicit (λ ≤ β).
    centering_max_iter::Int = 50
    regularize_schur::T = T(1e-12)
    max_schur_reg::Int = 8
    linear_solver::Type = MadNLP.LapackCPUSolver
end

mutable struct MadDualSDPSolver{T,M,SS<:SchurSystem{T}}
    model::M
    blocks::Vector{DualSDPBlock{T}}
    y::Vector{T}
    d::Vector{T}                 # B·g  damped-Newton direction
    g::Vector{T}                 # ∇ψ_t(y) = ∇ζ(y) - t·b
    grad_zeta::Vector{T}         # ∇ζ(y) = A(S^{-1})
    Δy::Vector{T}                # t·B·b  predictor direction
    dhat::Vector{T}              # holds d_k between centering and predictor
    work_m::Vector{T}
    sys::SS
    opt::MadDualSDPOptions{T}
    m::Int
    nblocks::Int
    ν::Int                       # barrier parameter Σ n_i
    A_param::T                   # A = δ + ω*(2β/(1-β)²)
    t::T
    λ::T
    α::T
    iter::Int
    n_predictor::Int
    n_newton::Int
    status::MadNLP.Status
    start_time::Float64
    inf_pr::T                    # ν / t   (proxy)
    obj_dual::T                  # ⟨b, y⟩
    obj_prim::T                  # ⟨C, X⟩  (optional, from primal proxy)
end

# ω(τ) = τ - log(1+τ),  ω_*(τ) = -τ - log(1-τ) for τ < 1.
_omega_star(τ) = -τ - log1p(-τ)

function MadDualSDPSolver(model::LRO.AbstractModel{T}; options...) where {T}
    opt = MadDualSDPOptions{T}(; options...)
    if !(model isa LRO.BufferedModelForSchur)
        model = LRO.BufferedModelForSchur(model, opt.regularize_schur)
    end
    if LRO.num_scalars(model) > 0
        error("MadDualSDPSolver: scalar/LP block not supported yet (use MadSDPSolver).")
    end
    nblocks = LRO.num_matrices(model)
    nblocks == 0 && error("MadDualSDPSolver: model has no matrix variables.")
    blocks =
        [DualSDPBlock{T}(LRO.side_dimension(model, LRO.MatrixIndex(i))) for i = 1:nblocks]
    m = model.meta.ncon
    sys = SchurSystem{T}(m; linear_solver = opt.linear_solver)
    ν = sum(b -> b.n, blocks)
    A_param = if opt.A === nothing
        opt.δ + _omega_star(2 * opt.β / (1 - opt.β)^2)
    else
        opt.A
    end
    return MadDualSDPSolver(
        model,
        blocks,
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        sys,
        opt,
        m,
        nblocks,
        ν,
        T(A_param),
        zero(T),
        zero(T),
        zero(T),
        0,
        0,
        0,
        MadNLP.INITIAL,
        time(),
        zero(T),
        zero(T),
        zero(T),
    )
end

# ----------------------------- helpers ------------------------------------

# Fill `out` with C_i - Σ_j y_j A_{i,j}.
function _build_slack!(out::Matrix{T}, model, y::Vector{T}, i::Int) where {T}
    _copy_to_dense!(out, LRO.grad(model, LRO.MatrixIndex(i)))
    AT_y = LRO.unsafe_jtprod(model, y, LRO.MatrixIndex(i))
    _axpy_sparse_dense!(out, -one(T), AT_y)
    # symmetrise
    @inbounds for q in axes(out, 2), p = 1:(q-1)
        v = (out[p, q] + out[q, p]) / 2
        out[p, q] = v
        out[q, p] = v
    end
    return out
end

# Factor S = L·L', set b.Ls and b.Si.  Returns false if S not PD.
function _factor_slack!(b::DualSDPBlock{T}) where {T}
    F = safe_cholesky!(b.buf_n, b.S)
    F === nothing && return false
    n = b.n
    # Copy L into b.Ls.data.
    Ldata = parent(b.Ls)
    fill!(Ldata, zero(T))
    @inbounds for j = 1:n, i = j:n
        Ldata[i, j] = F.L[i, j]
    end
    # S^{-1} via two triangular solves.
    II = Matrix{T}(LinearAlgebra.I, n, n)
    Lt = LinearAlgebra.UpperTriangular(transpose(parent(b.Ls)))
    Si = Lt \ (b.Ls \ II)
    copyto!(b.Si, Si)
    @inbounds for q = 1:n, p = 1:(q-1)
        v = (b.Si[p, q] + b.Si[q, p]) / 2
        b.Si[p, q] = v
        b.Si[q, p] = v
    end
    return true
end

# Refresh `b.S` and the factorization from current y.  Returns success.
function _refresh_slacks!(solver::MadDualSDPSolver{T}) where {T}
    for (i, b) in enumerate(solver.blocks)
        _build_slack!(b.S, solver.model, solver.y, i)
        _factor_slack!(b) || return false
    end
    return true
end

# ∇ζ(y)_j = Σ_i ⟨A_{i,j}, S_i^{-1}⟩ ;  ∇ψ_t = ∇ζ - t b.
function _compute_gradient!(solver::MadDualSDPSolver{T}) where {T}
    fill!(solver.grad_zeta, zero(T))
    for (i, b) in enumerate(solver.blocks)
        LRO.add_jprod!(solver.model, b.Si, solver.grad_zeta, LRO.MatrixIndex(i))
    end
    bvec = LRO.cons_constant(solver.model)
    @. solver.g = solver.grad_zeta - solver.t * bvec
    return
end

# Build & factorize ∇²ζ(y) via Schur with W = S^{-1}.
function _factorize_hessian!(solver::MadDualSDPSolver{T}) where {T}
    reg = solver.opt.regularize_schur
    W = LRO.ShapedSolution(zeros(T, 0), [b.Si for b in solver.blocks])
    factorised = false
    for _ = 1:(solver.opt.max_schur_reg)
        try
            build_and_factorize!(solver.sys, solver.model, W, reg)
            factorised = true
            break
        catch err
            (
                err isa LinearAlgebra.PosDefException ||
                err isa LinearAlgebra.SingularException
            ) || rethrow()
            reg = reg == zero(T) ? T(1e-12) : reg * 10
        end
    end
    return factorised
end

# Eigenvalues of  M = S_hat^{-1/2} · D · S_hat^{-1/2} ,
# computed as eigvals of the symmetric matrix  L_hat^{-1} · D · L_hat^{-T}.
# Stored into b.eigvals.
function _compute_proximity_eigs!(b::DualSDPBlock{T}) where {T}
    F = safe_cholesky!(b.buf_n, b.Shat)
    if F === nothing
        return false
    end
    # Form M = L \ D / L'   (symmetric).
    Lhat = F.L
    n = b.n
    # b.buf_n2 = L \ D
    copyto!(b.buf_n2, b.Dmat)
    LinearAlgebra.ldiv!(LinearAlgebra.LowerTriangular(Matrix(Lhat)), b.buf_n2)
    # b.buf_n2 = (L \ D) / L'   -> rdiv! with UpperTriangular
    LinearAlgebra.rdiv!(b.buf_n2, LinearAlgebra.UpperTriangular(transpose(Matrix(Lhat))))
    @inbounds for q = 1:n, p = 1:(q-1)
        v = (b.buf_n2[p, q] + b.buf_n2[q, p]) / 2
        b.buf_n2[p, q] = v
        b.buf_n2[q, p] = v
    end
    copyto!(b.eigvals, LinearAlgebra.eigvals(LinearAlgebra.Symmetric(b.buf_n2)))
    return true
end

# ξ(α) = Σ_i Σ_p [-log(1 - α μ_{i,p}) - log(1 + (α/(1-α)) μ_{i,p})]
function _xi_value(blocks::Vector{<:DualSDPBlock{T}}, α::T) where {T}
    s = zero(T)
    β2 = α / (1 - α)
    for b in blocks
        @inbounds for μ in b.eigvals
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
function _xi_alpha_max(blocks::Vector{<:DualSDPBlock{T}}) where {T}
    αmax = one(T)
    for b in blocks
        @inbounds for μ in b.eigvals
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
function _solve_alpha(blocks::Vector{<:DualSDPBlock{T}}, A::T) where {T}
    αmax = _xi_alpha_max(blocks)
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
        ξm = _xi_value(blocks, mid)
        if ξm < A
            lo = mid
        else
            hi = mid
        end
        (hi - lo) ≤ eps(T) * max(one(T), hi) && break
    end
    return (lo + hi) / 2
end

# ‖ŝ‖²_{s_k} = Σ_i tr(S_i^{-1} Ŝ_i S_i^{-1} Ŝ_i) = Σ_i ‖L_s^{-1} Ŝ L_s^{-T}‖_F².
# Uses the Cholesky `b.Ls` of S (at y_k, before predictor).
function _shat_norm2(blocks::Vector{<:DualSDPBlock{T}}) where {T}
    s = zero(T)
    for b in blocks
        copyto!(b.buf_n, b.Shat)
        LinearAlgebra.ldiv!(b.Ls, b.buf_n)
        LinearAlgebra.rdiv!(b.buf_n, transpose(b.Ls))
        @inbounds for q = 1:(b.n), p = 1:(b.n)
            s += b.buf_n[p, q]^2
        end
    end
    return s
end

# Primal proxy at the iterate.
# X_{k+1} = (1/t) S^{-1} ((1-α) Ŝ + α A*(Δy)) S^{-1}.
function _update_primal!(solver::MadDualSDPSolver{T}, α::T) where {T}
    t = solver.t
    for b in solver.blocks
        @. b.buf_n = (1 - α) * b.Shat + α * b.Dmat
        LinearAlgebra.mul!(b.buf_n2, b.Si, b.buf_n)
        LinearAlgebra.mul!(b.Xprim, b.buf_n2, b.Si)
        b.Xprim ./= t
        @inbounds for q = 1:(b.n), p = 1:(q-1)
            v = (b.Xprim[p, q] + b.Xprim[q, p]) / 2
            b.Xprim[p, q] = v
            b.Xprim[q, p] = v
        end
    end
    return
end

# ----------------------------- main loop ----------------------------------

function _find_initial_y!(solver::MadDualSDPSolver{T}) where {T}
    if solver.opt.initial_y !== nothing
        length(solver.opt.initial_y) == solver.m || error(
            "initial_y has length $(length(solver.opt.initial_y)), expected $(solver.m).",
        )
        copyto!(solver.y, solver.opt.initial_y)
        _refresh_slacks!(solver) ||
            error("MadDualSDPSolver: provided initial_y does not give S ≻ 0.")
        return
    end
    fill!(solver.y, zero(T))
    if _refresh_slacks!(solver)
        return
    end
    # Try y = -ρ·1 with ρ doubled until success.
    ρ = solver.opt.initial_shift
    while ρ ≤ solver.opt.initial_shift_max
        fill!(solver.y, -ρ)
        if _refresh_slacks!(solver)
            return
        end
        ρ *= 2
    end
    error(
        "MadDualSDPSolver: could not find interior y₀ by the y = -ρ·1 heuristic; " *
        "pass `initial_y = ...` explicitly.",
    )
end

function initialize!(solver::MadDualSDPSolver{T}) where {T}
    solver.iter = 0
    solver.n_predictor = 0
    solver.n_newton = 0
    solver.start_time = time()
    solver.status = MadNLP.REGULAR
    _find_initial_y!(solver)
    β = solver.opt.β
    # Pick t₀ as the least-squares best fit minimising ‖∇ζ(y₀) - t·b‖²_{y₀}.
    # For F_y unbounded this is essential — pure ζ-centering would diverge.
    # First, factor ∇²ζ at y₀ and compute the gradient.
    solver.t = zero(T)
    _compute_gradient!(solver)
    _factorize_hessian!(solver) || error("MadDualSDPSolver: failed to factorize ∇²ζ at y₀.")
    # d_ζ = B · ∇ζ.
    copyto!(solver.d, solver.grad_zeta)
    solve_inplace!(solver.sys, solver.d)
    # h_b = B · b.
    bvec = LRO.cons_constant(solver.model)
    copyto!(solver.dhat, bvec)
    solve_inplace!(solver.sys, solver.dhat)
    bBb = max(LinearAlgebra.dot(bvec, solver.dhat), eps(T))
    bBg = LinearAlgebra.dot(bvec, solver.d)
    t_fit = bBg / bBb
    bnorm = sqrt(bBb)
    solver.t = max(t_fit, β / max(bnorm, eps(T)))
    # Damped Newton on ψ_{t₀} until λ ≤ β/2.
    for _ = 1:(solver.opt.centering_max_iter)
        _refresh_slacks!(solver) ||
            error("MadDualSDPSolver: damped Newton left the interior during init.")
        _compute_gradient!(solver)
        _factorize_hessian!(solver) ||
            error("MadDualSDPSolver: failed to factorize ∇²ζ during init.")
        copyto!(solver.d, solver.g)
        solve_inplace!(solver.sys, solver.d)
        λ² = LinearAlgebra.dot(solver.g, solver.d)
        λ = sqrt(max(λ², zero(T)))
        solver.λ = λ
        if λ ≤ β / 2
            break
        end
        @. solver.y -= solver.d / (1 + λ)
        solver.iter += 1
        solver.n_newton += 1
    end
    return
end

# One predictor step.  Pre-condition: λ ≤ β at current y.
# d_k already in `solver.dhat`.
function predictor_step!(solver::MadDualSDPSolver{T}) where {T}
    # ŷ = y + d_k.
    @. solver.work_m = solver.y + solver.dhat
    # Build Ŝ and proximity data.
    for (i, b) in enumerate(solver.blocks)
        # Copy current S (pre-step) into a temp for X-proxy formula.
        # b.S is the slack at y_k.
        # Ŝ_i = C_i - A*_i(ŷ).
        _build_slack!(b.Shat, solver.model, solver.work_m, i)
    end
    # Δy = t · B · b   (re-use already-factorized Schur).
    b = LRO.cons_constant(solver.model)
    copyto!(solver.Δy, b)
    solve_inplace!(solver.sys, solver.Δy)
    solver.Δy .*= solver.t
    # D_i = A*_i(Δy).
    for (i, blk) in enumerate(solver.blocks)
        AT = LRO.unsafe_jtprod(solver.model, solver.Δy, LRO.MatrixIndex(i))
        fill!(blk.Dmat, zero(eltype(blk.Dmat)))
        _axpy_sparse_dense!(blk.Dmat, one(T), AT)
        @inbounds for q in axes(blk.Dmat, 2), p = 1:(q-1)
            v = (blk.Dmat[p, q] + blk.Dmat[q, p]) / 2
            blk.Dmat[p, q] = v
            blk.Dmat[q, p] = v
        end
        _compute_proximity_eigs!(blk) || (return false)
    end
    # ‖ŝ‖²_{s_k}  (uses the Cholesky of S at y_k — already stored in b.Ls).
    shat_n2 = _shat_norm2(solver.blocks)
    # 1-D root find on ξ(α) = A.
    α = _solve_alpha(solver.blocks, solver.A_param)
    solver.α = α
    # Primal proxy at the new iterate.
    _update_primal!(solver, α)
    # y_{k+1} = ŷ + α · Δy ;  t_{k+1} = ν · t_k / ((1-α) · ‖ŝ‖²_{s_k}).
    @. solver.y = solver.work_m + α * solver.Δy
    solver.t = solver.ν * solver.t / ((1 - α) * shat_n2)
    solver.iter += 1
    solver.n_predictor += 1
    return true
end

function step!(solver::MadDualSDPSolver{T}) where {T}
    opt = solver.opt
    β = opt.β
    # a) refresh S, compute g, factorize ∇²ζ, solve d = B·g, λ² = g'd.
    _refresh_slacks!(solver) || (solver.status = MadNLP.ERROR_IN_STEP_COMPUTATION; return)
    _compute_gradient!(solver)
    _factorize_hessian!(solver) ||
        (solver.status = MadNLP.ERROR_IN_STEP_COMPUTATION; return)
    copyto!(solver.d, solver.g)
    solve_inplace!(solver.sys, solver.d)
    λ² = LinearAlgebra.dot(solver.g, solver.d)
    λ = sqrt(max(λ², zero(T)))
    solver.λ = λ
    if λ > β
        # b) damped Newton, t unchanged.
        @. solver.y -= solver.d / (1 + λ)
        solver.iter += 1
        solver.n_newton += 1
    else
        # c) predictor — d_k is the centred direction we just computed; the
        # predictor needs ŷ = y + d_k (NOT y - d_k/(1+λ)).
        copyto!(solver.dhat, solver.d)
        predictor_step!(solver) || (solver.status = MadNLP.ERROR_IN_STEP_COMPUTATION)
    end
    return
end

function update_termination!(solver::MadDualSDPSolver{T}) where {T}
    opt = solver.opt
    solver.inf_pr = solver.ν / max(solver.t, eps(T))
    solver.obj_dual = LinearAlgebra.dot(LRO.cons_constant(solver.model), solver.y)
    # Primal objective only meaningful after the first predictor.
    if solver.n_predictor > 0
        s = zero(T)
        for (i, b) in enumerate(solver.blocks)
            s += LinearAlgebra.dot(LRO.grad(solver.model, LRO.MatrixIndex(i)), b.Xprim)
        end
        solver.obj_prim = s
    end
    if solver.inf_pr ≤ opt.tol
        solver.status = MadNLP.SOLVE_SUCCEEDED
    elseif solver.iter >= opt.max_iter
        solver.status = MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
    elseif time() - solver.start_time >= opt.max_wall_time
        solver.status = MadNLP.MAXIMUM_WALLTIME_EXCEEDED
    end
    return
end

function print_dual_header()
    @printf(
        "%4s %14s %14s %9s %9s %9s %9s\n",
        "iter",
        "p_obj",
        "d_obj",
        "ν/t",
        "λ",
        "α",
        "t"
    )
end

function print_dual_iter(solver::MadDualSDPSolver)
    if solver.iter % 10 == 0
        print_dual_header()
    end
    @printf(
        "%4d %14.6e %14.6e %9.2e %9.2e %9.2e %9.2e\n",
        solver.iter,
        solver.obj_prim,
        solver.obj_dual,
        solver.inf_pr,
        solver.λ,
        solver.α,
        solver.t,
    )
end

function solve!(solver::MadDualSDPSolver{T}; verbose::Bool = true) where {T}
    initialize!(solver)
    update_termination!(solver)
    verbose && print_dual_iter(solver)
    while solver.status == MadNLP.REGULAR
        step!(solver)
        solver.status == MadNLP.REGULAR || break
        update_termination!(solver)
        verbose && print_dual_iter(solver)
    end
    return solver
end

"""
    madsdp_dual(model::LRO.AbstractModel; kwargs...)

Solve a (matrix-block) SDP with Nesterov's dual-centered predictor-corrector
IPM (arXiv 2503.10155).  Returns the populated `MadDualSDPSolver`.
"""
madsdp_dual(model; kwargs...) = solve!(MadDualSDPSolver(model; kwargs...))
