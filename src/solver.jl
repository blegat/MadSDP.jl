# MadSDP primal-dual NT IPM driver, modelled on MadIPM's `MPCSolver`.
# We do NOT subtype `MadNLP.AbstractMadNLPSolver` for the MVP: the solver
# state is matrix-block-valued and doesn't naturally fit MadNLP's scalar
# `PrimalVector`.  We still consume MadNLP's `AbstractLinearSolver` for the
# m×m Schur factorisation (and through that, the GPU / cuDSS path).

mutable struct MadSDPSolver{T,M,SS<:SchurSystem{T}}
    model::M                            # LRO.BufferedModelForSchur (or LRO.Model)
    blocks::Vector{SDPBlock{T}}
    y::Vector{T}
    dely::Vector{T}
    Rp::Vector{T}
    # Scalar (LP) block — possibly empty (n_scalar == 0).
    n_scalar::Int
    x::Vector{T}                        # primal scalar variables x ≥ 0
    z::Vector{T}                        # dual scalar variables z ≥ 0
    delx::Vector{T}
    delz::Vector{T}
    Rd_lin::Vector{T}                   # scalar dual residual d_lin - z - C_lin' y
    W_lin::Vector{T}                    # x ./ z   (LP analog of NT scaling)
    Si_lin::Vector{T}                   # 1 ./ z
    RNT_lin::Vector{T}                  # Mehrotra second-order correction
    work_scalar::Vector{T}              # scratch buffer of length n_scalar
    work_m::Vector{T}                   # scratch buffer of length m
    sys::SS
    opt::MadSDPOptions{T}
    m::Int
    nblocks::Int
    # Iteration data
    mu::T
    sigma::T
    alpha::T
    beta::T
    inf_pr::T
    inf_du::T
    inf_compl::T
    obj_val::T
    iter::Int
    status::MadNLP.Status
    start_time::Float64
end

# Inner (unbuffered) `LRO.Model` — the source of truth for `C_lin` / `d_lin`.
# Both the `Model` and `BufferedModelForSchur` paths expose it as `.model` on
# the buffered wrapper.
_inner(m::LRO.BufferedModelForSchur) = m.model
_inner(m::LRO.Model) = m

function MadSDPSolver(model::LRO.AbstractModel{T}; options...) where {T}
    opt = MadSDPOptions{T}(; options...)
    if !(model isa LRO.BufferedModelForSchur)
        # Wrap with the Schur buffer so we can call `LRO.schur_complement!`.
        model = LRO.BufferedModelForSchur(model, opt.regularize_schur)
    end
    nblocks = LRO.num_matrices(model)
    if nblocks == 0
        error("MadSDP: model has no PSD blocks.")
    end
    blocks = [SDPBlock{T}(LRO.side_dimension(model, LRO.MatrixIndex(i))) for i = 1:nblocks]
    m = model.meta.ncon
    n_scalar = LRO.num_scalars(model)
    sys = SchurSystem{T}(m; linear_solver = opt.linear_solver)
    return MadSDPSolver(
        model,
        blocks,
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        n_scalar,
        zeros(T, n_scalar),
        zeros(T, n_scalar),
        zeros(T, n_scalar),
        zeros(T, n_scalar),
        zeros(T, n_scalar),
        zeros(T, n_scalar),
        zeros(T, n_scalar),
        zeros(T, n_scalar),
        zeros(T, n_scalar),
        zeros(T, m),
        sys,
        opt,
        m,
        nblocks,
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        0,
        MadNLP.INITIAL,
        time(),
    )
end

# Primal objective ⟨C, X⟩ summed over blocks + d_lin' x.
function primal_objective(solver::MadSDPSolver{T}) where {T}
    s = zero(T)
    for (i, b) in enumerate(solver.blocks)
        s += LinearAlgebra.dot(LRO.grad(solver.model, LRO.MatrixIndex(i)), b.X)
    end
    if solver.n_scalar > 0
        s += LinearAlgebra.dot(LRO.grad(solver.model, LRO.ScalarIndex), solver.x)
    end
    return s
end

function dual_objective(solver::MadSDPSolver{T}) where {T}
    return LinearAlgebra.dot(LRO.cons_constant(solver.model), solver.y)
end

# Initial point : X = S = β I per block,  y = 0,  x = z = β ones.
function initialize!(solver::MadSDPSolver{T}) where {T}
    model = solver.model
    inner = _inner(model)
    bscale = LinearAlgebra.norm(LRO.cons_constant(model), Inf)
    cscale = zero(T)
    for i = 1:(solver.nblocks)
        cscale = max(cscale, LinearAlgebra.norm(LRO.grad(model, LRO.MatrixIndex(i)), Inf))
    end
    if solver.n_scalar > 0
        cscale = max(cscale, LinearAlgebra.norm(inner.d_lin, Inf))
    end
    n_total_dim = total_dim(solver.blocks) + solver.n_scalar
    β = max(
        T(10),
        T(sqrt(max(bscale, one(T)) * max(cscale, one(T)))) * sqrt(T(n_total_dim)),
    )
    for b in solver.blocks
        fill!(b.X, zero(T))
        fill!(b.S, zero(T))
        @inbounds for k = 1:(b.n)
            b.X[k, k] = β
            b.S[k, k] = β
        end
        fill!(b.RNT, zero(T))
    end
    fill!(solver.y, zero(T))
    if solver.n_scalar > 0
        fill!(solver.x, β)
        fill!(solver.z, β)
        fill!(solver.RNT_lin, zero(T))
    end
    solver.iter = 0
    solver.status = MadNLP.REGULAR
    solver.start_time = time()
    update_residuals!(solver)
    solver.mu = total_complementarity(solver) / max(n_total_dim, 1)
    return
end

# Σ tr(X_i S_i) + dot(x, z).
function total_complementarity(solver::MadSDPSolver{T}) where {T}
    s = trace_XS(solver.blocks)
    if solver.n_scalar > 0
        s += LinearAlgebra.dot(solver.x, solver.z)
    end
    return s
end

function shaped_W(solver::MadSDPSolver{T}) where {T}
    if solver.n_scalar > 0
        return LRO.ShapedSolution(solver.W_lin, [b.W for b in solver.blocks])
    else
        return LRO.ShapedSolution(zeros(T, 0), [b.W for b in solver.blocks])
    end
end

# One outer Mehrotra predictor-corrector iteration.
function mpc_step!(solver::MadSDPSolver{T}) where {T}
    opt = solver.opt
    # 1. NT scaling (matrix + scalar).
    if !prepare_W!(solver.blocks)
        solver.status = MadNLP.ERROR_IN_STEP_COMPUTATION
        return
    end
    if solver.n_scalar > 0
        @inbounds for k = 1:(solver.n_scalar)
            zk = solver.z[k]
            solver.Si_lin[k] = one(T) / zk
            solver.W_lin[k] = solver.x[k] / zk
        end
    end

    # 2. Build & factorise the Schur matrix.
    reg = opt.regularize_schur
    factorised = false
    for _ = 1:(opt.max_schur_reg)
        try
            build_and_factorize!(solver.sys, solver.model, shaped_W(solver), reg)
            factorised = true
            break
        catch err
            err isa LinearAlgebra.PosDefException ||
                err isa LinearAlgebra.SingularException ||
                rethrow()
            reg = reg == zero(T) ? T(1e-12) : reg * 10
        end
    end
    if !factorised
        solver.status = MadNLP.ERROR_IN_STEP_COMPUTATION
        return
    end

    # 3. Residuals & complementarity.
    update_residuals!(solver)
    n_total_dim = total_dim(solver.blocks) + solver.n_scalar
    μ = total_complementarity(solver) / n_total_dim
    solver.mu = μ

    # 4. Predictor.
    predictor_rhs!(solver.dely, solver)
    solve_inplace!(solver.sys, solver.dely)
    compute_directions!(solver, solver.dely, :predict, zero(T))
    α_aff, β_aff = fraction_to_boundary(solver, opt.tau)
    μ_aff = affine_complementarity(solver, α_aff, β_aff)

    σ = clamp((μ_aff / max(μ, eps(T)))^opt.sigma_expon, T(1e-8), T(1))
    solver.sigma = σ
    σμ = σ * μ

    # 5. Mehrotra correction from predictor step.
    update_RNT!(solver)

    # 6. Corrector.
    corrector_rhs!(solver.dely, solver, σμ)
    solve_inplace!(solver.sys, solver.dely)
    compute_directions!(solver, solver.dely, :correct, σμ)
    α, β = fraction_to_boundary(solver, opt.tau)
    solver.alpha = α
    solver.beta = β

    # 7. Apply step.
    for b in solver.blocks
        @. b.X = b.X + α * b.delX
        @inbounds for q = 1:(b.n), p = 1:(q-1)
            v = (b.X[p, q] + b.X[q, p]) / 2
            b.X[p, q] = v
            b.X[q, p] = v
        end
        @. b.S = b.S + β * b.delS
        @inbounds for q = 1:(b.n), p = 1:(q-1)
            v = (b.S[p, q] + b.S[q, p]) / 2
            b.S[p, q] = v
            b.S[q, p] = v
        end
    end
    @. solver.y = solver.y + β * solver.dely
    if solver.n_scalar > 0
        @. solver.x = solver.x + α * solver.delx
        @. solver.z = solver.z + β * solver.delz
    end
    solver.iter += 1
    return
end

function update_termination!(solver::MadSDPSolver{T}) where {T}
    opt = solver.opt
    pobj = primal_objective(solver)
    dobj = dual_objective(solver)
    solver.obj_val = pobj

    solver.inf_pr =
        LinearAlgebra.norm(solver.Rp, Inf) /
        max(T(1), LinearAlgebra.norm(LRO.cons_constant(solver.model), Inf))

    Rd_norm = zero(T)
    for b in solver.blocks
        Rd_norm = max(Rd_norm, LinearAlgebra.norm(b.Rd, Inf))
    end
    if solver.n_scalar > 0
        Rd_norm = max(Rd_norm, LinearAlgebra.norm(solver.Rd_lin, Inf))
    end
    cnorm = zero(T)
    for i = 1:(solver.nblocks)
        cnorm =
            max(cnorm, LinearAlgebra.norm(LRO.grad(solver.model, LRO.MatrixIndex(i)), Inf))
    end
    if solver.n_scalar > 0
        cnorm = max(cnorm, LinearAlgebra.norm(_inner(solver.model).d_lin, Inf))
    end
    solver.inf_du = Rd_norm / max(T(1), cnorm)
    solver.inf_compl = abs(pobj - dobj) / max(T(1), abs(pobj) + abs(dobj))
    if max(solver.inf_pr, solver.inf_du, solver.inf_compl) <= opt.tol
        solver.status = MadNLP.SOLVE_SUCCEEDED
    elseif solver.iter >= opt.max_iter
        solver.status = MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
    elseif time() - solver.start_time >= opt.max_wall_time
        solver.status = MadNLP.MAXIMUM_WALLTIME_EXCEEDED
    elseif solver.mu < opt.mu_min
        solver.status = MadNLP.SOLVE_SUCCEEDED
    end
    return
end

function print_header()
    @printf(
        "%4s %14s %14s %9s %9s %9s %9s %9s\n",
        "iter",
        "p_obj",
        "d_obj",
        "inf_pr",
        "inf_du",
        "gap",
        "mu",
        "α/β"
    )
end

function print_iter(solver::MadSDPSolver)
    if solver.iter % 10 == 0
        print_header()
    end
    @printf(
        "%4d %14.6e %14.6e %9.2e %9.2e %9.2e %9.2e %4.2f/%4.2f\n",
        solver.iter,
        primal_objective(solver),
        dual_objective(solver),
        solver.inf_pr,
        solver.inf_du,
        solver.inf_compl,
        solver.mu,
        solver.alpha,
        solver.beta,
    )
end

function solve!(solver::MadSDPSolver{T}; verbose::Bool = true) where {T}
    initialize!(solver)
    update_termination!(solver)
    verbose && print_iter(solver)
    while solver.status == MadNLP.REGULAR
        mpc_step!(solver)
        solver.status == MadNLP.REGULAR || break
        update_termination!(solver)
        verbose && print_iter(solver)
    end
    return solver
end

"""
    madsdp(model::LRO.AbstractModel; kwargs...)

Solve the (possibly mixed LP + SDP) problem encoded as a LowRankOpt model using
the MadSDP primal-dual NT interior-point method.  Returns the populated
`MadSDPSolver`.
"""
madsdp(model; kwargs...) = solve!(MadSDPSolver(model; kwargs...))
