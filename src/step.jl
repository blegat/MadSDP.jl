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

# Right-hand side `h` of the Schur system `H Δy = h` targeting `μ S⁻¹`:
#   h = R_p + Σ_i A_i(X − μ S⁻¹ + E(R_d) − corr)
#           + C_lin (x − μ ./ z + (x ./ z) .* R_d_lin − corr_lin)
# where the corrections are only included if `corrected`.
function rhs!(h::Vector{T}, solver, μ::T, corrected::Bool) where {T}
    model = solver.model
    copyto!(h, solver.Rp)
    for (i, b) in enumerate(solver.blocks)
        @. b.buf = b.X - μ * b.Si
        if solver.rd_nonzero[i]
            _apply_E!(b.buf2, b.dir, b, b.Rd)
            b.buf .+= b.buf2
        end
        if corrected
            b.buf .-= b.corr
        end
        LRO.add_jprod!(model, b.buf, h, LRO.MatrixIndex(i))
    end
    if solver.n_scalar > 0
        @inbounds for k = 1:(solver.n_scalar)
            x, zinv = solver.x[k], solver.zinv[k]
            v = x - μ * zinv + x * zinv * solver.Rd_lin[k]
            if corrected
                v -= solver.corr_lin[k]
            end
            solver.work_scalar[k] = v
        end
        LinearAlgebra.mul!(h, _inner(model).C_lin, solver.work_scalar, true, true)
    end
    return h
end

# Directions from the solution `Δy` of the Schur system:
#   ΔS = R_d − A*(Δy),   ΔX = μ S⁻¹ − X − E(R_d) + E(A*(Δy)) + corr
#   Δz = R_d_lin − C_lin' Δy,   Δx = μ ./ z − x − (x ./ z) .* Δz + corr_lin
function directions!(solver, dely::Vector{T}, μ::T, corrected::Bool) where {T}
    model = solver.model
    for (i, b) in enumerate(solver.blocks)
        AT_dy = LRO.unsafe_jtprod(model, dely, LRO.MatrixIndex(i))
        copyto!(b.delS, b.Rd)
        _axpy_sparse_dense!(b.delS, -one(T), AT_dy)
        _symmetrize!(b.delS)
        _apply_E!(b.buf2, b.dir, b, AT_dy)
        @. b.delX = μ * b.Si - b.X + b.buf2
        if solver.rd_nonzero[i]
            _apply_E!(b.buf2, b.dir, b, b.Rd)
            b.delX .-= b.buf2
        end
        if corrected
            b.delX .+= b.corr
        end
        _symmetrize!(b.delX)
    end
    if solver.n_scalar > 0
        LinearAlgebra.mul!(solver.delz, _inner(model).C_lin', dely)
        @inbounds for k = 1:(solver.n_scalar)
            solver.delz[k] = solver.Rd_lin[k] - solver.delz[k]
            x, zinv = solver.x[k], solver.zinv[k]
            solver.delx[k] = μ * zinv - x - x * zinv * solver.delz[k]
            if corrected
                solver.delx[k] += solver.corr_lin[k]
            end
        end
    end
    return
end

# Second-order corrections from the affine-scaling directions currently
# stored in `delX`, `delS`, `delx` and `delz`.
function correction!(solver)
    for b in solver.blocks
        correction!(b.corr, b.dir, b)
    end
    @inbounds for k = 1:(solver.n_scalar)
        solver.corr_lin[k] = -solver.delx[k] * solver.delz[k] * solver.zinv[k]
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

prepare_step!(::EigenStep, b::SDPBlock) = _cholesky_factor!(b.Lx, b.X)

# Fraction `τ` of the largest step keeping `L L' + α Δ` positive definite.
function _eigen_step(buf, L, Δ, τ)
    T = eltype(buf)
    copyto!(buf, Δ)
    LinearAlgebra.ldiv!(LinearAlgebra.LowerTriangular(L), buf)
    LinearAlgebra.rdiv!(buf, LinearAlgebra.LowerTriangular(L)')
    _symmetrize!(buf)
    λ = first(LinearAlgebra.eigvals!(LinearAlgebra.Symmetric(buf, :L), 1:1))
    return λ > -sqrt(eps(T)) ? one(T) : min(one(T), -T(τ) / λ)
end

function step_lengths(s::EigenStep, solver)
    T = eltype(solver.y)
    α = β = one(T)
    for b in solver.blocks
        α = min(α, _eigen_step(b.buf, b.Lx, b.delX, s.τ))
        β = min(β, _eigen_step(b.buf, b.Ls, b.delS, s.τ))
    end
    if solver.n_scalar > 0
        α = min(α, T(s.τ) * _max_scalar_step(solver.x, solver.delx))
        β = min(β, T(s.τ) * _max_scalar_step(solver.z, solver.delz))
    end
    return α, β
end

prepare_step!(::CholeskyBacktrack, ::SDPBlock) = true

function _backtrack(s::CholeskyBacktrack, blocks, X, ΔX, x, dx)
    T = eltype(x)
    α = one(T)
    for b in blocks
        while !_is_add_mul_posdef(b.buf, X(b), α, ΔX(b))
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
    α = _backtrack(s, solver.blocks, b -> b.X, b -> b.delX, solver.x, solver.delx)
    β = _backtrack(s, solver.blocks, b -> b.S, b -> b.delS, solver.z, solver.delz)
    return α, β
end

# Complementarity at the trial point `(X + α ΔX, S + β ΔS)`.
function affine_complementarity(solver, α::T, β::T) where {T}
    s = zero(T)
    for b in solver.blocks
        s += LinearAlgebra.dot(b.X, b.S)
        s += β * LinearAlgebra.dot(b.X, b.delS)
        s += α * LinearAlgebra.dot(b.delX, b.S)
        s += α * β * LinearAlgebra.dot(b.delX, b.delS)
    end
    @inbounds for k = 1:(solver.n_scalar)
        s += (solver.x[k] + α * solver.delx[k]) * (solver.z[k] + β * solver.delz[k])
    end
    return s / max(_barrier_dim(solver), 1)
end
