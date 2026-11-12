# MadSDP interior-point driver, modelled on MadIPM's `MPCSolver`.
# We do NOT subtype `MadNLP.AbstractMadNLPSolver`: the solver state is
# matrix-block-valued and doesn't naturally fit MadNLP's scalar
# `PrimalVector`. The algorithm is assembled from the components of
# `components.jl`; this file contains what all algorithms share.

mutable struct MadSDPSolver{T,M,B<:SDPBlock{T},A<:AbstractAlgorithm,AS,I,SW}
    model::M                            # LRO.BufferedDualModel
    algorithm::A
    state::AS                           # see `algorithm_state`
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
    L::LRO.ShapedConeVector{T,Matrix{T}}
    R::LRO.ShapedConeVector{T,Matrix{T}}
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

# Inner (unbuffered) `LRO.DualModel` — the source of truth for `C_lin` / `d_lin`.
_inner(m::LRO.BufferedDualModel) = m.model
_inner(m::LRO.DualModel) = m

"""
    MadSDPSolver(
        model::LRO.AbstractDualModel;
        algorithm::AbstractAlgorithm = PrimalDual(direction, step, centering),
        direction::AbstractDirection = NT(),
        step::AbstractStepLength = EigenStep(),
        centering::AbstractCentering = Mehrotra(),
        initial_point::AbstractInitialPoint = default_initial_point(algorithm),
        schur::AbstractSchurSolver = DenseCholesky(),
        options...,
    )

Interior-point solver for the problem encoded in `model`, assembled from
the given components. The keyword arguments `direction`, `step` and
`centering` configure the default [`PrimalDual`](@ref) algorithm and are
ignored if `algorithm` is given. The remaining keyword arguments are the
fields of [`MadSDPOptions`](@ref).

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
and the dual method of [Nes25] is `MadSDPSolver(model; algorithm =
DualCentering())`.
"""
function MadSDPSolver(
    model::LRO.AbstractDualModel{T};
    direction::AbstractDirection = NT(),
    step::AbstractStepLength = EigenStep(),
    centering::AbstractCentering = Mehrotra(),
    algorithm::AbstractAlgorithm = PrimalDual(direction, step, centering),
    initial_point::AbstractInitialPoint = default_initial_point(algorithm),
    schur::AbstractSchurSolver = DenseCholesky(),
    native_model = nothing,
    options...,
) where {T}
    opt = MadSDPOptions{T}(; options...)
    model =
        LRO.BufferedDualModel(model, opt.datasparsity; schur = needs_schur_buffer(schur))
    nblocks = LRO.num_matrices(model)
    n_scalar = LRO.num_scalars(model)
    if nblocks == 0 && n_scalar == 0 && !(schur isa Union{AugmentedKKT,RootKKT})
        error("MadSDP: model has no variables.")
    end
    check_model(algorithm, model)
    check_schur(schur, algorithm)
    B = typeof(SDPBlock{T}(algorithm, 1))
    blocks = B[
        SDPBlock{T}(algorithm, LRO.side_dimension(model, LRO.MatrixIndex(i))) for
        i = 1:nblocks
    ]
    m = model.meta.ncon
    vec_m() = zeros(T, m)
    vec_s() = zeros(T, n_scalar)
    x, zinv = vec_s(), vec_s()
    L = LRO.ShapedConeVector(x, [schur_scaling(algorithm, b)[1] for b in blocks])
    R = LRO.ShapedConeVector(zinv, [schur_scaling(algorithm, b)[2] for b in blocks])
    cnorm =
        T[LinearAlgebra.norm(LRO.grad(model, LRO.MatrixIndex(i)), Inf) for i = 1:nblocks]
    return MadSDPSolver(
        model,
        algorithm,
        algorithm_state(algorithm, T, m),
        initial_point,
        schur_workspace(schur, T, m, isnothing(native_model) ? model : native_model),
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

check_model(::AbstractAlgorithm, _) = nothing

# Primal objective ⟨C, X⟩ summed over blocks + d_lin' x.
function primal_objective(solver::MadSDPSolver{T}) where {T}
    s = zero(T)
    for (i, b) in enumerate(solver.blocks)
        s += LinearAlgebra.dot(LRO.grad(solver.model, LRO.MatrixIndex(i)), b.X)
    end
    if solver.n_scalar > 0
        s += LinearAlgebra.dot(LRO.grad(solver.model, LRO.ScalarIndex), solver.x)
    end
    return s + additional_objective(solver.schur)
end

function dual_objective(solver::MadSDPSolver{T}) where {T}
    return LinearAlgebra.dot(LRO.cons_constant(solver.model), solver.y)
end

# Barrier parameter ν = Σ n_i + n_scalar.
_barrier_dim(solver) =
    total_dim(solver.blocks) + solver.n_scalar + additional_barrier_degree(solver.schur)

# Σ tr(X_i S_i) + dot(x, z).
function total_complementarity(solver::MadSDPSolver{T}) where {T}
    s = trace_XS(solver.blocks)
    if solver.n_scalar > 0
        s += LinearAlgebra.dot(solver.x, solver.z)
    end
    return s + additional_complementarity(solver.schur)
end

# `out = C_i - A*_i(y)`
function _slack!(out::Matrix{T}, model, y::Vector{T}, i::Integer) where {T}
    _copy_to_dense!(out, LRO.grad(model, LRO.MatrixIndex(i)))
    AT_y = LRO.unsafe_jtprod(model, y, LRO.MatrixIndex(i))
    _axpy_sparse_dense!(out, -one(T), AT_y)
    return _symmetrize!(out)
end

# Sets the dual slacks `S = C - A*(y)` and `z = d - C_lin' y`.
function _set_dual_slacks!(solver::MadSDPSolver)
    for (i, b) in enumerate(solver.blocks)
        _slack!(b.S, solver.model, solver.y, i)
    end
    if solver.n_scalar > 0
        inner = _inner(solver.model)
        LinearAlgebra.mul!(solver.z, inner.C_lin', solver.y)
        solver.z .= inner.d_lin .- solver.z
    end
    return
end

# Initial point : X = S = β I per block,  y = 0,  x = z = β ones.
function initial_point!(solver::MadSDPSolver{T}, p::ScaledIdentity) where {T}
    model = solver.model
    inner = _inner(model)
    bscale = LinearAlgebra.norm(LRO.cons_constant(model), Inf)
    cscale = maximum(solver.cnorm; init = zero(T))
    if solver.n_scalar > 0
        cscale = max(cscale, LinearAlgebra.norm(inner.d_lin, Inf))
    end
    β = if isnothing(p.scale)
        max(
            T(10),
            sqrt(max(bscale, one(T)) * max(cscale, one(T))) * sqrt(T(_barrier_dim(solver))),
        )
    else
        T(p.scale)
    end
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
    copyto!(solver.y, p.y)
    for (i, b) in enumerate(solver.blocks)
        copyto!(b.X, p.X[i])
    end
    if isnothing(p.x)
        fill!(solver.x, one(T))
    else
        copyto!(solver.x, p.x)
    end
    _set_dual_slacks!(solver)
    return
end

# Whether the dual slacks are strictly feasible, also factorizing them.
function _dual_interior(solver::MadSDPSolver{T}) where {T}
    all(b -> _cholesky_factor!(b.Ls, b.S), solver.blocks) || return false
    return all(>(zero(T)), solver.z)
end

function initial_point!(solver::MadSDPSolver{T}, p::DualShift) where {T}
    for b in solver.blocks
        copyto!(b.X, LinearAlgebra.I)
    end
    fill!(solver.x, one(T))
    ρ = zero(T)
    while ρ <= p.max_shift
        fill!(solver.y, -ρ)
        _set_dual_slacks!(solver)
        if _dual_interior(solver)
            # The first `ρ > 0` for which the Cholesky factorization succeeds
            # may give a numerically singular `S`, so we double it once more.
            if !iszero(ρ)
                fill!(solver.y, -2ρ)
                _set_dual_slacks!(solver)
            end
            return
        end
        ρ = iszero(ρ) ? T(p.initial_shift) : 2ρ
    end
    error(
        "MadSDP: could not find a dual interior point `y = -ρ 1` with ρ ≤ $(p.max_shift); " *
        "use `InitialPoint` to give one.",
    )
end

function initialize!(solver::MadSDPSolver)
    initial_point!(solver, solver.initial_point)
    initialize_constraints!(solver.schur, solver)
    solver.iter = 0
    solver.status = MadNLP.REGULAR
    solver.start_time = time()
    initialize_algorithm!(solver, solver.algorithm)
    return
end

function _factorize_schur!(solver::MadSDPSolver)
    return factorize_schur!(solver.schur, solver)
end

function factorize_schur!(workspace, solver::MadSDPSolver{T}) where {T}
    opt = solver.opt
    H = workspace.H
    reg = opt.regularize_schur
    for _ = 1:(opt.max_schur_reg)
        LRO.schur_complement!(solver.model, solver.L, solver.R, H)
        if reg > zero(T)
            @inbounds for i in axes(H, 1)
                H[i, i] += reg
            end
        end
        factorize!(workspace) && return true
        reg = iszero(reg) ? T(1e-12) : 10 * reg
    end
    return false
end

iterate!(solver::MadSDPSolver) = iterate!(solver, solver.algorithm)

# Relative primal and dual infeasibilities from `Rp`, `Rd` and `Rd_lin`.
function _update_infeasibilities!(solver::MadSDPSolver{T}) where {T}
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
    solver.inf_du = max(Rd_norm / max(T(1), cnorm), additional_infeasibility(solver.schur))
    return
end

# Sets the status from `converged` and the iteration and time limits.
function _update_status!(solver::MadSDPSolver, converged::Bool)
    opt = solver.opt
    if converged
        solver.status = MadNLP.SOLVE_SUCCEEDED
    elseif solver.iter >= opt.max_iter
        solver.status = MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
    elseif time() - solver.start_time >= opt.max_wall_time
        solver.status = MadNLP.MAXIMUM_WALLTIME_EXCEEDED
    end
    return
end

function update_termination!(solver::MadSDPSolver)
    return update_termination!(solver, solver.algorithm)
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
        iterate!(solver)
        solver.status == MadNLP.REGULAR || break
        update_termination!(solver)
        verbose && print_iter(solver)
    end
    return solver
end

"""
    madsdp(model::LRO.AbstractDualModel; kwargs...)

Solve the (possibly mixed LP + SDP) problem encoded as a LowRankOpt model
with [`MadSDPSolver`](@ref). Returns the populated `MadSDPSolver`.
"""
madsdp(model; kwargs...) = solve!(MadSDPSolver(model; kwargs...))

function MadSDPSolver(model::LRO.Model; schur = nothing, options...)
    if isnothing(schur)
        direction = get(options, :direction, NT())
        algorithm = get(options, :algorithm, nothing)
        if algorithm isa PrimalDual
            direction = algorithm.direction
        end
        schur = if algorithm isa DualCentering
            DenseCholesky()
        else
            direction isa NT ? RootKKT() : AugmentedKKT()
        end
    end
    if schur isa AugmentedKKT && LRO.num_second_order_cones(model) > 0
        return MadSDPSolver(LRO.dual_model(model); schur, options...)
    end
    if schur isa Union{AugmentedKKT,RootKKT}
        data = model.data
        C, d, _, _ = scalar_block_bounds(model)
        dual =
            LRO.DualModel(data.C, data.A, data.b, SparseArrays.sparsevec(d), C, data.msizes)
        return MadSDPSolver(dual; schur, native_model = model, options...)
    end
    return MadSDPSolver(LRO.dual_model(model); schur, options...)
end
