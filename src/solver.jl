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

function MadSDPSolver(
    model::LRO.AbstractModel{T};
    options...,
) where {T}
    opt = MadSDPOptions{T}(; options...)
    if !(model isa LRO.BufferedModelForSchur)
        # Wrap with the Schur buffer so we can call `LRO.schur_complement!`.
        model = LRO.BufferedModelForSchur(model, opt.regularize_schur)
    end
    if LRO.num_scalars(model) > 0
        error("MadSDP MVP: scalar primal variables are not yet supported (got $(LRO.num_scalars(model)) scalars). Please open an issue.")
    end
    nblocks = LRO.num_matrices(model)
    if nblocks == 0
        error("MadSDP: model has no PSD blocks.")
    end
    blocks = [SDPBlock{T}(LRO.side_dimension(model, LRO.MatrixIndex(i))) for i in 1:nblocks]
    m = model.meta.ncon
    sys = SchurSystem{T}(m; linear_solver = opt.linear_solver)
    return MadSDPSolver(
        model, blocks,
        zeros(T, m), zeros(T, m), zeros(T, m),
        sys, opt,
        m, nblocks,
        zero(T), zero(T), zero(T), zero(T),
        zero(T), zero(T), zero(T), zero(T),
        0, MadNLP.INITIAL, time(),
    )
end

# Primal objective ⟨C, X⟩ summed over blocks.
function primal_objective(solver::MadSDPSolver{T}) where {T}
    s = zero(T)
    for (i, b) in enumerate(solver.blocks)
        s += LinearAlgebra.dot(LRO.grad(solver.model, LRO.MatrixIndex(i)), b.X)
    end
    return s
end

function dual_objective(solver::MadSDPSolver{T}) where {T}
    return LinearAlgebra.dot(LRO.cons_constant(solver.model), solver.y)
end

# Trivial initial point : X = S = β I per block,  y = 0.
function initialize!(solver::MadSDPSolver{T}) where {T}
    model = solver.model
    # Scale heuristic, à la Loraine: pick β = max(10, sqrt(n_total) * scale).
    bscale = LinearAlgebra.norm(LRO.cons_constant(model), Inf)
    cscale = zero(T)
    for i in 1:solver.nblocks
        cscale = max(cscale, LinearAlgebra.norm(LRO.grad(model, LRO.MatrixIndex(i)), Inf))
    end
    n_total = total_dim(solver.blocks)
    β = max(T(10), T(sqrt(max(bscale, one(T)) * max(cscale, one(T)))) * sqrt(T(n_total)))
    for b in solver.blocks
        fill!(b.X, zero(T))
        fill!(b.S, zero(T))
        @inbounds for k in 1:b.n
            b.X[k, k] = β
            b.S[k, k] = β
        end
        fill!(b.RNT, zero(T))
    end
    fill!(solver.y, zero(T))
    solver.iter = 0
    solver.status = MadNLP.REGULAR
    solver.start_time = time()
    update_residuals!(solver)
    solver.mu = trace_XS(solver.blocks) / max(total_dim(solver.blocks), 1)
    return
end

# One outer Mehrotra predictor-corrector iteration.
function mpc_step!(solver::MadSDPSolver{T}) where {T}
    opt = solver.opt
    # 1. NT scaling
    if !prepare_W!(solver.blocks)
        solver.status = MadNLP.ERROR_IN_STEP_COMPUTATION
        return
    end

    # 2. Build & factorise the Schur matrix (with adaptive Tikhonov bump if
    # the factorisation is hit by curvature near the boundary).
    reg = opt.regularize_schur
    factorised = false
    for _ in 1:opt.max_schur_reg
        try
            build_and_factorize!(solver.sys, solver.model,
                LRO.ShapedSolution(zeros(T, 0), [b.W for b in solver.blocks]),
                reg)
            factorised = true
            break
        catch err
            err isa LinearAlgebra.PosDefException || err isa LinearAlgebra.SingularException || rethrow()
            reg = reg == zero(T) ? T(1e-12) : reg * 10
        end
    end
    if !factorised
        solver.status = MadNLP.ERROR_IN_STEP_COMPUTATION
        return
    end

    # 3. Update residuals & complementarity.
    update_residuals!(solver)
    n_total = total_dim(solver.blocks)
    μ = trace_XS(solver.blocks) / n_total
    solver.mu = μ

    # 4. Predictor.
    predictor_rhs!(solver.dely, solver)
    solve_inplace!(solver.sys, solver.dely)
    compute_directions!(solver, solver.dely, :predict, zero(T))
    α_aff, β_aff = fraction_to_boundary(solver, opt.tau)
    μ_aff = affine_complementarity(solver, α_aff, β_aff)

    # Mehrotra centering parameter.
    σ = clamp((μ_aff / max(μ, eps(T)))^opt.sigma_expon, T(1e-8), T(1))
    solver.sigma = σ
    σμ = σ * μ

    # 5. Build the Mehrotra correction matrix from the predictor step.
    update_RNT!(solver)

    # 6. Corrector.
    corrector_rhs!(solver.dely, solver, σμ)
    solve_inplace!(solver.sys, solver.dely)
    compute_directions!(solver, solver.dely, :correct, σμ)
    α, β = fraction_to_boundary(solver, opt.tau)
    solver.alpha = α
    solver.beta = β

    # 7. Take the step.  We use the same (α, β) across blocks (the minimum
    # found above), which is the standard NT update.
    for b in solver.blocks
        @. b.X = b.X + α * b.delX
        # symmetrise to keep X exactly symmetric
        @inbounds for q in 1:b.n, p in 1:(q-1)
            v = (b.X[p, q] + b.X[q, p]) / 2
            b.X[p, q] = v
            b.X[q, p] = v
        end
        @. b.S = b.S + β * b.delS
        @inbounds for q in 1:b.n, p in 1:(q-1)
            v = (b.S[p, q] + b.S[q, p]) / 2
            b.S[p, q] = v
            b.S[q, p] = v
        end
    end
    @. solver.y = solver.y + β * solver.dely
    solver.iter += 1
    return
end

function update_termination!(solver::MadSDPSolver{T}) where {T}
    opt = solver.opt
    pobj = primal_objective(solver)
    dobj = dual_objective(solver)
    solver.obj_val = pobj

    solver.inf_pr = LinearAlgebra.norm(solver.Rp, Inf) /
                    max(T(1), LinearAlgebra.norm(LRO.cons_constant(solver.model), Inf))
    Rd_norm = zero(T)
    for b in solver.blocks
        Rd_norm = max(Rd_norm, LinearAlgebra.norm(b.Rd, Inf))
    end
    cnorm = zero(T)
    for i in 1:solver.nblocks
        cnorm = max(cnorm, LinearAlgebra.norm(LRO.grad(solver.model, LRO.MatrixIndex(i)), Inf))
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
    @printf("%4s %14s %14s %9s %9s %9s %9s %9s\n",
        "iter", "p_obj", "d_obj", "inf_pr", "inf_du", "gap", "mu", "α/β")
end

function print_iter(solver::MadSDPSolver)
    if solver.iter % 10 == 0
        print_header()
    end
    @printf("%4d %14.6e %14.6e %9.2e %9.2e %9.2e %9.2e %4.2f/%4.2f\n",
        solver.iter,
        primal_objective(solver),
        dual_objective(solver),
        solver.inf_pr, solver.inf_du, solver.inf_compl,
        solver.mu, solver.alpha, solver.beta,
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

Solve the SDP encoded as a LowRankOpt model using the MadSDP primal-dual NT
interior-point method.  Returns the populated `MadSDPSolver`.
"""
madsdp(model; kwargs...) = solve!(MadSDPSolver(model; kwargs...))
