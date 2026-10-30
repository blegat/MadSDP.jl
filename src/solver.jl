# MadSDP primal-dual interior-point driver, modelled on MadIPM's `MPCSolver`.
# We do NOT subtype `MadNLP.AbstractMadNLPSolver`: the solver state is
# matrix-block-valued and doesn't naturally fit MadNLP's scalar
# `PrimalVector`. The algorithm is assembled from the components of
# `components.jl`.

mutable struct MadSDPSolver{T,M,B<:SDPBlock{T},D,SL,C,I,SW}
    model::M                            # LRO.BufferedModelForSchur
    direction::D
    step::SL
    centering::C
    initial_point::I
    schur::SW                           # workspace of the Schur solver
    blocks::Vector{B}
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
    zinv::Vector{T}                     # 1 ./ z
    corr_lin::Vector{T}                 # Mehrotra second-order correction
    work_scalar::Vector{T}              # scratch buffer of length n_scalar
    work_m::Vector{T}                   # scratch buffer of length m
    rd_nonzero::Vector{Bool}            # whether `Rd` is above roundoff
    cnorm::Vector{T}                    # ‖C_i‖_∞
    # Schur complement scalings `⟨Aⱼ, L Aₖ R⟩`; their scalar parts are `x`
    # and `1 ./ z` so that their product is `x ./ z`.
    L::LRO.ShapedSolution{T,Matrix{T}}
    R::LRO.ShapedSolution{T,Matrix{T}}
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
_inner(m::LRO.BufferedModelForSchur) = m.model
_inner(m::LRO.Model) = m

"""
    MadSDPSolver(
        model::LRO.AbstractModel;
        direction::AbstractDirection = NT(),
        step::AbstractStepLength = EigenStep(),
        centering::AbstractCentering = Mehrotra(),
        initial_point::AbstractInitialPoint = ScaledIdentity(),
        schur::AbstractSchurSolver = DenseCholesky(),
        options...,
    )

Primal-dual interior-point solver for the problem encoded in `model`,
assembled from the given components. The remaining keyword arguments are
the fields of [`MadSDPOptions`](@ref).

For instance, the algorithm of `mc_psd` [HRVW96] for the max-cut
relaxation is
```julia
MadSDPSolver(
    model;
    direction = HKM(),
    step = CholeskyBacktrack(),
    centering = AdaptiveMu(),
    initial_point = InitialPoint([I(n)], -(1 .+ vec(sum(abs, L, dims = 1)))),
)
```
"""
function MadSDPSolver(
    model::LRO.AbstractModel{T};
    direction::AbstractDirection = NT(),
    step::AbstractStepLength = EigenStep(),
    centering::AbstractCentering = Mehrotra(),
    initial_point::AbstractInitialPoint = ScaledIdentity(),
    schur::AbstractSchurSolver = DenseCholesky(),
    options...,
) where {T}
    opt = MadSDPOptions{T}(; options...)
    if !(model isa LRO.BufferedModelForSchur)
        model = LRO.BufferedModelForSchur(model, opt.datasparsity)
    end
    nblocks = LRO.num_matrices(model)
    n_scalar = LRO.num_scalars(model)
    if nblocks == 0 && n_scalar == 0
        error("MadSDP: model has no variables.")
    end
    B = typeof(SDPBlock{T}(direction, 0))
    blocks = B[
        SDPBlock{T}(direction, LRO.side_dimension(model, LRO.MatrixIndex(i))) for
        i = 1:nblocks
    ]
    m = model.meta.ncon
    vec_m() = zeros(T, m)
    vec_s() = zeros(T, n_scalar)
    x, zinv = vec_s(), vec_s()
    L = LRO.ShapedSolution(x, [schur_scaling(b.dir, b)[1] for b in blocks])
    R = LRO.ShapedSolution(zinv, [schur_scaling(b.dir, b)[2] for b in blocks])
    cnorm =
        T[LinearAlgebra.norm(LRO.grad(model, LRO.MatrixIndex(i)), Inf) for i = 1:nblocks]
    return MadSDPSolver(
        model,
        direction,
        step,
        centering,
        initial_point,
        schur_workspace(schur, T, m),
        blocks,
        vec_m(),
        vec_m(),
        vec_m(),
        n_scalar,
        x,
        vec_s(),
        vec_s(),
        vec_s(),
        vec_s(),
        zinv,
        vec_s(),
        vec_s(),
        vec_m(),
        trues(nblocks) |> Vector{Bool},
        cnorm,
        L,
        R,
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

# Barrier parameter ν = Σ n_i + n_scalar.
_barrier_dim(solver) = total_dim(solver.blocks) + solver.n_scalar

# Σ tr(X_i S_i) + dot(x, z).
function total_complementarity(solver::MadSDPSolver{T}) where {T}
    s = trace_XS(solver.blocks)
    if solver.n_scalar > 0
        s += LinearAlgebra.dot(solver.x, solver.z)
    end
    return s
end

# Initial point : X = S = β I per block,  y = 0,  x = z = β ones.
function initial_point!(solver::MadSDPSolver{T}, ::ScaledIdentity) where {T}
    model = solver.model
    inner = _inner(model)
    bscale = LinearAlgebra.norm(LRO.cons_constant(model), Inf)
    cscale = maximum(solver.cnorm; init = zero(T))
    if solver.n_scalar > 0
        cscale = max(cscale, LinearAlgebra.norm(inner.d_lin, Inf))
    end
    β = max(
        T(10),
        T(sqrt(max(bscale, one(T)) * max(cscale, one(T)))) * sqrt(T(_barrier_dim(solver))),
    )
    for b in solver.blocks
        copyto!(b.X, β * LinearAlgebra.I)
        copyto!(b.S, β * LinearAlgebra.I)
    end
    fill!(solver.y, zero(T))
    fill!(solver.x, β)
    fill!(solver.z, β)
    return
end

function initial_point!(solver::MadSDPSolver{T}, p::InitialPoint) where {T}
    model = solver.model
    copyto!(solver.y, p.y)
    for (i, b) in enumerate(solver.blocks)
        copyto!(b.X, p.X[i])
        # S = C - A*(y)
        _copy_to_dense!(b.S, LRO.grad(model, LRO.MatrixIndex(i)))
        AT_y = LRO.unsafe_jtprod(model, solver.y, LRO.MatrixIndex(i))
        _axpy_sparse_dense!(b.S, -one(T), AT_y)
    end
    if solver.n_scalar > 0
        inner = _inner(model)
        if isnothing(p.x)
            fill!(solver.x, one(T))
        else
            copyto!(solver.x, p.x)
        end
        LinearAlgebra.mul!(solver.z, inner.C_lin', solver.y)
        solver.z .= inner.d_lin .- solver.z
    end
    return
end

function initialize_centering!(solver, ::Mehrotra)
    solver.mu = total_complementarity(solver) / max(_barrier_dim(solver), 1)
    return
end

function initialize_centering!(solver, c::AdaptiveMu)
    solver.mu =
        c.initial_ratio * total_complementarity(solver) / max(_barrier_dim(solver), 1)
    return
end

function initialize!(solver::MadSDPSolver)
    initial_point!(solver, solver.initial_point)
    solver.iter = 0
    solver.status = MadNLP.REGULAR
    solver.start_time = time()
    update_residuals!(solver)
    initialize_centering!(solver, solver.centering)
    return
end

# Computes everything that depends only on the current iterate: the
# factorizations of the blocks and the scalings of the Schur complement.
function _prepare!(solver::MadSDPSolver{T}) where {T}
    for b in solver.blocks
        prepare_block!(b) || return false
        prepare_direction!(b.dir, b) || return false
        prepare_step!(solver.step, b) || return false
    end
    solver.zinv .= inv.(solver.z)
    return true
end

function _factorize_schur!(solver::MadSDPSolver{T}) where {T}
    opt = solver.opt
    H = solver.schur.H
    reg = opt.regularize_schur
    for _ = 1:(opt.max_schur_reg)
        LRO.schur_complement!(solver.model, solver.L, solver.R, H)
        if reg > zero(T)
            @inbounds for i in axes(H, 1)
                H[i, i] += reg
            end
        end
        factorize!(solver.schur) && return true
        reg = iszero(reg) ? T(1e-12) : 10 * reg
    end
    return false
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
        @. b.X += α * b.delX
        _symmetrize!(b.X)
        @. b.S += β * b.delS
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

function _start_iteration!(solver)
    if !_prepare!(solver) || !_factorize_schur!(solver)
        solver.status = MadNLP.ERROR_IN_STEP_COMPUTATION
        return false
    end
    return true
end

# One Mehrotra predictor-corrector iteration.
function iterate!(solver::MadSDPSolver{T}, c::Mehrotra) where {T}
    _start_iteration!(solver) || return
    μ = total_complementarity(solver) / max(_barrier_dim(solver), 1)
    solver.mu = μ
    # Predictor
    _solve_directions!(solver, zero(T), false)
    α_aff, β_aff = step_lengths(solver.step, solver)
    μ_aff = affine_complementarity(solver, α_aff, β_aff)
    σ = clamp((μ_aff / max(μ, eps(T)))^T(c.σ_exponent), T(1e-8), one(T))
    solver.sigma = σ
    correction!(solver)
    # Corrector
    _solve_directions!(solver, σ * μ, true)
    α, β = step_lengths(solver.step, solver)
    _take_step!(solver, α, β)
    return
end

# One iteration of `mc_psd` [HRVW96].
function iterate!(solver::MadSDPSolver{T}, c::AdaptiveMu) where {T}
    _start_iteration!(solver) || return
    _solve_directions!(solver, solver.mu, false)
    α, β = step_lengths(solver.step, solver)
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
    cnorm = maximum(solver.cnorm; init = zero(T))
    if solver.n_scalar > 0
        Rd_norm = max(Rd_norm, LinearAlgebra.norm(solver.Rd_lin, Inf))
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
        iterate!(solver, solver.centering)
        solver.status == MadNLP.REGULAR || break
        update_termination!(solver)
        verbose && print_iter(solver)
    end
    return solver
end

"""
    madsdp(model::LRO.AbstractModel; kwargs...)

Solve the (possibly mixed LP + SDP) problem encoded as a LowRankOpt model
with [`MadSDPSolver`](@ref). Returns the populated `MadSDPSolver`.
"""
madsdp(model; kwargs...) = solve!(MadSDPSolver(model; kwargs...))
