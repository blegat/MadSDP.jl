# Residuals, right-hand sides, search directions and step lengths, generic
# over the components.
#
# Convention for the dual slack:  S_i := C_i - Σ_j y_j A_{i,j}.  We do NOT
# enforce this exactly across iterations; instead we track the dual residual
#     R_d[i]    = C_i - S_i - Σ_j y_j A_{i,j}                       (matrix block)
#     R_d_lin   = d_lin - z - C_lin' y                              (scalar block)
# and a step of length 1 eliminates them.

# R_p[j] = b_j − Σ_i ⟨A_{i,j}, X_i⟩ − (C_lin x)_j ;
# R_d[i] = C_i − S_i − A^*(y)_i  ;   R_d_lin = d_lin − z − C_lin' y.
function update_residuals!(solver)
    model = solver.model
    inner = _inner(model)
    T = eltype(solver.Rp)
    # Accumulate A(X) + C_lin x into Rp, then negate and add b.
    fill!(solver.Rp, zero(T))
    for (i, b) in enumerate(solver.blocks)
        LRO.add_jprod!(model, b.X, solver.Rp, LRO.MatrixIndex(i))
    end
    if solver.n_scalar > 0
        LinearAlgebra.mul!(solver.work_m, inner.C_lin, solver.x)
        solver.Rp .+= solver.work_m
    end
    @inbounds for j in eachindex(solver.Rp)
        solver.Rp[j] = LRO.cons_constant(model)[j] - solver.Rp[j]
    end
    # Matrix dual residual.
    for (i, b) in enumerate(solver.blocks)
        AT_y = LRO.unsafe_jtprod(model, solver.y, LRO.MatrixIndex(i))
        _copy_to_dense!(b.Rd, LRO.grad(model, LRO.MatrixIndex(i)))
        b.Rd .-= b.S
        _axpy_sparse_dense!(b.Rd, -one(T), AT_y)
        # A dual residual at the roundoff level is dropped so that the
        # directions do not need to apply `E` to it.
        threshold = 100 * eps(T) * max(one(T), solver.cnorm[i])
        solver.rd_nonzero[i] = LinearAlgebra.norm(b.Rd, Inf) > threshold
    end
    # Scalar dual residual.
    if solver.n_scalar > 0
        LinearAlgebra.mul!(solver.Rd_lin, inner.C_lin', solver.y)
        @inbounds for k = 1:(solver.n_scalar)
            solver.Rd_lin[k] = inner.d_lin[k] - solver.z[k] - solver.Rd_lin[k]
        end
    end
    return
end

# `out = E(M)` where `M` may be the `FillArrays.Zeros` returned by
# `LRO.unsafe_jtprod` if no constraint involves the block.
function _apply_E!(out, ws, b, M)
    if M isa LRO.FillArrays.Zeros
        fill!(out, zero(eltype(out)))
        return out
    end
    return apply_E!(out, ws, b, M)
end

# Right-hand side `p` of the linearized KKT system (see `SchurKKTSystem`)
# targeting `X S = μ I`:
#   r_c = μ S⁻¹ - X + corr,        r_cl = μ ./ z - x + corr_lin
#   r_p = R_p,   r_d = R_d,        r_dl = R_d_lin
# where the corrections are only included if `corrected`. The dual residuals
# at the roundoff level are not given to `E`, see `update_residuals!`.
function set_rhs!(p::SDPKKTVector{T}, solver, μ::T, corrected::Bool) where {T}
    copyto!(p.y, solver.Rp)
    for (i, b) in enumerate(solver.blocks)
        @. p.X[i] = μ * b.Si - b.X
        if corrected
            p.X[i] .+= b.work.corr
        end
        copyto!(p.S[i], b.Rd)
    end
    @inbounds for k = 1:(solver.n_scalar)
        p.x[k] = μ * solver.zinv[k] - solver.x[k]
        if corrected
            p.x[k] += solver.corr_lin[k]
        end
    end
    copyto!(p.z, solver.Rd_lin)
    solver.kkt.rd_negligible .= .!solver.rd_nonzero
    return p
end

# Solves `K d = p` and improves the solution with at most
# `opt.refine_max_iter` iterations of iterative refinement, until the
# residual is below `opt.refine_tol * ‖p‖∞`, as `MadIPM.solve_system!`.
function solve_system!(
    d::SDPKKTVector{T},
    solver,
    p::SDPKKTVector{T},
) where {T}
    kkt = solver.kkt
    copyto!(MadNLP.full(d), MadNLP.full(p))
    MadNLP.solve_kkt!(kkt, d)
    max_iter = solver.opt.refine_max_iter
    max_iter > 0 || return d
    # The residuals are not at the roundoff level of the dual residuals.
    fill!(kkt.rd_negligible, false)
    w = solver._w
    tol = solver.opt.refine_tol * max(one(T), MadNLP.norm(p, Inf))
    for _ = 1:max_iter
        copyto!(MadNLP.full(w), MadNLP.full(p))
        LinearAlgebra.mul!(w, kkt, d, -one(T), one(T))
        MadNLP.norm(w, Inf) <= tol && break
        MadNLP.solve_kkt!(kkt, w)
        MadNLP.axpy!(one(T), w, d)
    end
    return d
end

# Second-order corrections from the affine-scaling directions currently
# stored in `solver.d`.
function correction!(solver)
    d = solver.d
    for (i, b) in enumerate(solver.blocks)
        correction!(b.work.corr, b.work.dir, b, d.X[i], d.S[i])
    end
    @inbounds for k = 1:(solver.n_scalar)
        solver.corr_lin[k] = -d.x[k] * d.z[k] * solver.zinv[k]
    end
    return
end

# Largest `α ≤ 1` such that `x + α Δx ≥ 0`.
function _max_scalar_step(x::Vector{T}, dx::Vector{T}) where {T}
    α = typemax(T)
    @inbounds for k in eachindex(x)
        if dx[k] < zero(T)
            α = min(α, -x[k] / dx[k])
        end
    end
    return α
end

# `L⁻¹ Δ L⁻ᵀ` given by its products so that the step length does not form it
# with triangular solves of matrices.
struct _CongruenceInverse{T}
    L::Matrix{T}
    Δ::Matrix{T}
    tmp::Vector{T}
end

function LinearAlgebra.mul!(w::AbstractVector, A::_CongruenceInverse, v::AbstractVector)
    L = LinearAlgebra.LowerTriangular(A.L)
    copyto!(A.tmp, v)
    LinearAlgebra.ldiv!(L', A.tmp)
    LinearAlgebra.mul!(w, LinearAlgebra.Symmetric(A.Δ, :U), A.tmp)
    LinearAlgebra.ldiv!(L, w)
    return w
end

struct _EigenStepWorkspace{T}
    eigen::LRO.EigenWorkspace{T}
    tmp::Vector{T}
end

# The Lanczos iterations reuse their workspace so that the step lengths do not
# allocate.
function step_workspace(::EigenStep, ::Type{T}, n) where {T}
    return _EigenStepWorkspace(LRO.EigenWorkspace(zeros(T, n, n)), zeros(T, n))
end

prepare_step!(::EigenStep, b::SDPBlock) = _cholesky_factor!(b.work.Lx, b.X)

# Fraction `τ` of the largest step keeping `L L' + α Δ` positive definite.
function _eigen_step(workspace::_EigenStepWorkspace{T}, L, Δ, τ) where {T}
    A = _CongruenceInverse(L, Δ, workspace.tmp)
    λ = LRO.lanczos_eigmin!(workspace.eigen, A)
    return λ > -sqrt(eps(T)) ? one(T) : min(one(T), -T(τ) / λ)
end

function step_lengths(s::EigenStep, solver)
    T = eltype(solver.y)
    α = β = one(T)
    d = solver.d
    for (i, b) in enumerate(solver.blocks)
        α = min(α, _eigen_step(b.work.step, b.work.Lx, d.X[i], s.τ))
        β = min(β, _eigen_step(b.work.step, b.Ls, d.S[i], s.τ))
    end
    if solver.n_scalar > 0
        α = min(α, T(s.τ) * _max_scalar_step(solver.x, d.x))
        β = min(β, T(s.τ) * _max_scalar_step(solver.z, d.z))
    end
    return α, β
end

step_workspace(::CholeskyBacktrack, ::Type, _) = nothing

prepare_step!(::CholeskyBacktrack, ::SDPBlock) = true

# Backtracks on `X(i) + α ΔX(i)` for each block `i`.
function _backtrack(s::CholeskyBacktrack, blocks, X, ΔX, x, dx)
    T = eltype(x)
    α = one(T)
    for (i, b) in enumerate(blocks)
        while !_is_add_mul_posdef(b.buf, X(i), α, ΔX(i))
            α *= T(s.shrink)
            if α < eps(T)
                return zero(T)
            end
        end
    end
    if !isone(α)
        α *= T(s.boundary)
    end
    if !isempty(x)
        α = min(α, T(s.boundary) * _max_scalar_step(x, dx))
    end
    return α
end

function step_lengths(s::CholeskyBacktrack, solver)
    blocks, d = solver.blocks, solver.d
    α = _backtrack(s, blocks, i -> blocks[i].X, i -> d.X[i], solver.x, d.x)
    β = _backtrack(s, blocks, i -> blocks[i].S, i -> d.S[i], solver.z, d.z)
    return α, β
end

# Complementarity at the trial point `(X + α ΔX, S + β ΔS)`.
function affine_complementarity(solver, α::T, β::T) where {T}
    s = zero(T)
    d = solver.d
    for (i, b) in enumerate(solver.blocks)
        ΔX, ΔS = d.X[i], d.S[i]
        s += LinearAlgebra.dot(b.X, b.S)
        s += β * LinearAlgebra.dot(b.X, ΔS)
        s += α * LinearAlgebra.dot(ΔX, b.S)
        s += α * β * LinearAlgebra.dot(ΔX, ΔS)
    end
    @inbounds for k = 1:(solver.n_scalar)
        s += (solver.x[k] + α * d.x[k]) * (solver.z[k] + β * d.z[k])
    end
    return s / max(_barrier_dim(solver), 1)
end
