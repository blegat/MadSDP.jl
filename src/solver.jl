# MadSDP interior-point driver, modelled on MadIPM's `MPCSolver`.
# We do NOT subtype `MadNLP.AbstractMadNLPSolver`: the solver state is
# matrix-block-valued and doesn't naturally fit MadNLP's scalar
# `PrimalVector`. The algorithm is assembled from the components of
# `components.jl`; this file contains what all algorithms share.

mutable struct MadSDPSolver{T,M,B<:SDPBlock{T},A<:AbstractAlgorithm,AS,I,KKT}
    model::M                            # LRO.BufferedModelForSchur
    algorithm::A
    state::AS                           # see `algorithm_state`
    initial_point::I
    kkt::KKT                            # see `SchurKKTSystem`
    blocks::Vector{B}
    y::Vector{T}
    Rp::Vector{T}
    # Search direction (ΔX, Δx, Δy, ΔS, Δz), right-hand side of the KKT
    # system and buffer of the iterative refinement, see `solve_system!`.
    d::SDPKKTVector{T}
    p::SDPKKTVector{T}
    _w::SDPKKTVector{T}
    # Scalar (LP) block — possibly empty (n_scalar == 0).
    n_scalar::Int
    x::Vector{T}                        # primal scalar variables x ≥ 0
    z::Vector{T}                        # dual scalar variables z ≥ 0
    Rd_lin::Vector{T}                   # scalar dual residual d_lin - z - C_lin' y
    zinv::Vector{T}                     # 1 ./ z
    corr_lin::Vector{T}                 # Mehrotra second-order correction
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
        algorithm::AbstractAlgorithm = PrimalDual(direction, step, centering),
        direction::AbstractDirection = NT(),
        step::AbstractStepLength = EigenStep(),
        centering::AbstractCentering = Mehrotra(),
        initial_point::AbstractInitialPoint = default_initial_point(algorithm),
        linear_solver::Type = MadNLP.LapackCPUSolver,
        opt_linear_solver = default_kkt_options(linear_solver),
        schur::Union{Nothing,AbstractSchurSolver} = nothing,
        options...,
    )

Interior-point solver for the problem encoded in `model`, assembled from
the given components. The keyword arguments `direction`, `step` and
`centering` configure the default [`PrimalDual`](@ref) algorithm and are
ignored if `algorithm` is given. The remaining keyword arguments are the
fields of [`MadSDPOptions`](@ref).

The Newton systems are solved by reduction to the Schur complement, see
`MadSDP.SchurKKTSystem`, which is factorized by the
`MadNLP.AbstractLinearSolver` `linear_solver` with the options
`opt_linear_solver`, by default LAPACK's Cholesky factorization. For
instance, `linear_solver = MadNLP.CHOLMODSolver` uses CHOLMOD.
The keyword `schur` is an alternative to `linear_solver` and
`opt_linear_solver`, see [`AbstractSchurSolver`](@ref).

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
    model::LRO.AbstractModel{T};
    direction::AbstractDirection = NT(),
    step::AbstractStepLength = EigenStep(),
    centering::AbstractCentering = Mehrotra(),
    algorithm::AbstractAlgorithm = PrimalDual(direction, step, centering),
    initial_point::AbstractInitialPoint = default_initial_point(algorithm),
    linear_solver::Union{Nothing,Type} = nothing,
    opt_linear_solver = nothing,
    schur::Union{Nothing,AbstractSchurSolver} = nothing,
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
    check_model(algorithm, model)
    if !isnothing(schur)
        if !isnothing(linear_solver) || !isnothing(opt_linear_solver)
            throw(
                ArgumentError(
                    "`schur` cannot be combined with `linear_solver` or `opt_linear_solver`.",
                ),
            )
        end
        check_schur(schur, algorithm)
    end
    B = typeof(SDPBlock{T}(algorithm, 1))
    blocks = B[
        SDPBlock{T}(algorithm, LRO.side_dimension(model, LRO.MatrixIndex(i))) for
        i = 1:nblocks
    ]
    m = model.meta.ncon
    vec_m() = zeros(T, m)
    vec_s() = zeros(T, n_scalar)
    x, zinv = vec_s(), vec_s()
    L = LRO.ShapedSolution(x, [schur_scaling(algorithm, b)[1] for b in blocks])
    R = LRO.ShapedSolution(zinv, [schur_scaling(algorithm, b)[2] for b in blocks])
    cnorm =
        T[LinearAlgebra.norm(LRO.grad(model, LRO.MatrixIndex(i)), Inf) for i = 1:nblocks]
    kkt = if schur isa QRSchur
        schur_workspace(schur, T, m)
    else
        LS, opt_LS = _linear_solver(schur, linear_solver, opt_linear_solver)
        MadNLP.create_kkt_system(
            SchurKKTSystem,
            model,
            blocks,
            L,
            R,
            LS;
            opt_linear_solver = opt_LS,
        )
    end
    kkt_vector() = SDPKKTVector{T}([b.n for b in blocks], n_scalar, m)
    return MadSDPSolver(
        model,
        algorithm,
        algorithm_state(algorithm, T, m),
        initial_point,
        kkt,
        blocks,
        vec_m(),
        vec_m(),
        kkt_vector(),
        kkt_vector(),
        kkt_vector(),
        n_scalar,
        x,
        vec_s(),
        vec_s(),
        zinv,
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

check_schur(::AbstractSchurSolver, ::AbstractAlgorithm) = nothing

# Linear solver and its options from the keywords of `MadSDPSolver`. The
# options are copied as the KKT system modifies them.
function _linear_solver(::Nothing, linear_solver, opt)
    LS = something(linear_solver, MadNLP.LapackCPUSolver)
    return LS, isnothing(opt) ? default_kkt_options(LS) : deepcopy(opt)
end
function _linear_solver(::DenseCholesky, _, _)
    return MadNLP.LapackCPUSolver, default_kkt_options(MadNLP.LapackCPUSolver)
end
function _linear_solver(s::MadNLPSchur, _, _)
    if isnothing(s.opt_linear_solver)
        # Bunch-Kaufman, the historical default of `MadNLPSchur` for LAPACK.
        return s.linear_solver, _lapack_options(s.linear_solver, MadNLP.BUNCHKAUFMAN)
    end
    return s.linear_solver, deepcopy(s.opt_linear_solver)
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

_initialize_kkt!(::Any) = nothing
_initialize_kkt!(kkt::SchurKKTSystem) = MadNLP.initialize!(kkt)

function initialize!(solver::MadSDPSolver)
    _initialize_kkt!(solver.kkt)
    initial_point!(solver, solver.initial_point)
    solver.iter = 0
    solver.status = MadNLP.REGULAR
    solver.start_time = time()
    initialize_algorithm!(solver, solver.algorithm)
    return
end

factorize_system!(solver::MadSDPSolver) = factorize_system!(solver.kkt, solver)

# Assembles and factorizes the KKT system, increasing the regularization
# from `opt.regularize_schur` until the factorization succeeds, at most
# `opt.max_schur_reg` times, as `MadIPM.factorize_regularized_system!`.
function factorize_system!(kkt::SchurKKTSystem{T}, solver::MadSDPSolver{T}) where {T}
    opt = solver.opt
    reg = opt.regularize_schur
    for _ = 1:(opt.max_schur_reg)
        MadNLP.regularize_diagonal!(kkt, reg - kkt.reg[], zero(T))
        MadNLP.build_kkt!(kkt)
        MadNLP.factorize_kkt!(kkt)
        is_factorized(kkt) && return true
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
    solver.inf_du = Rd_norm / max(T(1), cnorm)
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
    madsdp(model::LRO.AbstractModel; kwargs...)

Solve the (possibly mixed LP + SDP) problem encoded as a LowRankOpt model
with [`MadSDPSolver`](@ref). Returns the populated `MadSDPSolver`.
"""
madsdp(model; kwargs...) = solve!(MadSDPSolver(model; kwargs...))
