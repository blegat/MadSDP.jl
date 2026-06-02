# Predictor / corrector RHS and step computation for primal-dual NT.
#
# Convention for the dual slack:  S_i := C_i - Σ_j y_j A_{i,j}.  We do NOT
# enforce this exactly across iterations; instead we track the dual residual
#     R_d[i]    = C_i - S_i - Σ_j y_j A_{i,j}                       (matrix block)
#     R_d_lin   = d_lin - z - C_lin' y                              (scalar block)
# and a corrector step at full β eliminates them.  Standard infeasible-IPM
# mechanism — same as Loraine.

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

function _copy_to_dense!(Y::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T}
    if X isa SparseArrays.SparseMatrixCSC
        fill!(Y, zero(T))
        rows = SparseArrays.rowvals(X)
        vals = SparseArrays.nonzeros(X)
        @inbounds for j in axes(X, 2)
            for k in SparseArrays.nzrange(X, j)
                Y[rows[k], j] = vals[k]
            end
        end
    else
        copyto!(Y, X)
    end
    return Y
end

# In-place  Y .+= α * X  where X may be sparse and Y is dense.
function _axpy_sparse_dense!(
    Y::AbstractMatrix{T},
    α::T,
    X::SparseArrays.SparseMatrixCSC{T},
) where {T}
    rows = SparseArrays.rowvals(X)
    vals = SparseArrays.nonzeros(X)
    @inbounds for j in axes(X, 2)
        for k in SparseArrays.nzrange(X, j)
            Y[rows[k], j] += α * vals[k]
        end
    end
    return Y
end
_axpy_sparse_dense!(Y::AbstractMatrix, α, X::AbstractMatrix) = (Y .+= α .* X; Y)

# Predictor RHS  h = R_p + matrix-block contribution + scalar-block contribution.
function predictor_rhs!(h::Vector{T}, solver) where {T}
    model = solver.model
    inner = _inner(model)
    copyto!(h, solver.Rp)
    # Matrix blocks : h += A_i(W (R_d + S) W).
    for (i, b) in enumerate(solver.blocks)
        @. b.buf_n = b.Rd + b.S
        LinearAlgebra.mul!(b.buf_n2, b.W, b.buf_n)
        LinearAlgebra.mul!(b.buf_n, b.buf_n2, b.W)
        @inbounds for q = 1:(b.n), p = 1:(q-1)
            v = (b.buf_n[p, q] + b.buf_n[q, p]) / 2
            b.buf_n[p, q] = v
            b.buf_n[q, p] = v
        end
        LRO.add_jprod!(model, b.buf_n, h, LRO.MatrixIndex(i))
    end
    # Scalar block : h += C_lin * (W_lin .* (R_d_lin + z))  =
    #                     C_lin * (W_lin .* R_d_lin) + C_lin * x.
    if solver.n_scalar > 0
        @. solver.work_scalar = solver.W_lin * solver.Rd_lin + solver.x
        LinearAlgebra.mul!(solver.work_m, inner.C_lin, solver.work_scalar)
        h .+= solver.work_m
    end
    return h
end

# Corrector RHS.
function corrector_rhs!(h::Vector{T}, solver, σμ::T) where {T}
    model = solver.model
    inner = _inner(model)
    copyto!(h, solver.Rp)
    # Matrix blocks.
    for (i, b) in enumerate(solver.blocks)
        LinearAlgebra.mul!(b.buf_n, b.G', b.Rd)
        LinearAlgebra.mul!(b.buf_n2, b.buf_n, b.G)
        @inbounds for k = 1:(b.n)
            b.buf_n2[k, k] += b.D[k] - σμ / b.D[k]
        end
        b.buf_n2 .-= b.RNT
        LinearAlgebra.mul!(b.buf_n, b.G, b.buf_n2)
        LinearAlgebra.mul!(b.buf_n2, b.buf_n, b.G')
        @inbounds for q = 1:(b.n), p = 1:(q-1)
            v = (b.buf_n2[p, q] + b.buf_n2[q, p]) / 2
            b.buf_n2[p, q] = v
            b.buf_n2[q, p] = v
        end
        LRO.add_jprod!(model, b.buf_n2, h, LRO.MatrixIndex(i))
    end
    # Scalar block:
    #   h += C_lin * (W_lin .* R_d_lin + x + (δx .* δz − σμ) ./ z).
    if solver.n_scalar > 0
        @inbounds for k = 1:(solver.n_scalar)
            solver.work_scalar[k] =
                solver.W_lin[k] * solver.Rd_lin[k] +
                solver.x[k] +
                (solver.delx[k] * solver.delz[k] - σμ) * solver.Si_lin[k]
        end
        LinearAlgebra.mul!(solver.work_m, inner.C_lin, solver.work_scalar)
        h .+= solver.work_m
    end
    return h
end

# δz = R_d_lin − C_lin' δy ;  δx_predictor = −x − W_lin .* δz ;
# δx_corrector = σμ./z − x − W_lin .* δz + RNT_lin.
function compute_directions!(solver, dely::Vector{T}, kind::Symbol, σμ::T) where {T}
    model = solver.model
    inner = _inner(model)
    # Matrix block directions.
    for (i, b) in enumerate(solver.blocks)
        AT_dy = LRO.unsafe_jtprod(model, dely, LRO.MatrixIndex(i))
        copyto!(b.delS, b.Rd)
        _axpy_sparse_dense!(b.delS, -one(T), AT_dy)
        @inbounds for q = 1:(b.n), p = 1:(q-1)
            v = (b.delS[p, q] + b.delS[q, p]) / 2
            b.delS[p, q] = v
            b.delS[q, p] = v
        end
        LinearAlgebra.mul!(b.buf_n, b.W, b.delS)
        LinearAlgebra.mul!(b.buf_n2, b.buf_n, b.W)
        if kind === :predict
            @. b.delX = -b.X - b.buf_n2
        elseif kind === :correct
            LinearAlgebra.mul!(b.buf_n, b.G, b.RNT)
            LinearAlgebra.mul!(b.delX, b.buf_n, b.G')
            @. b.delX = σμ * b.Si - b.X - b.buf_n2 + b.delX
        else
            error("unknown kind $kind")
        end
        @inbounds for q = 1:(b.n), p = 1:(q-1)
            v = (b.delX[p, q] + b.delX[q, p]) / 2
            b.delX[p, q] = v
            b.delX[q, p] = v
        end
    end
    # Scalar block directions.
    if solver.n_scalar > 0
        # δz = R_d_lin − C_lin' δy.
        LinearAlgebra.mul!(solver.delz, inner.C_lin', dely)
        @inbounds for k = 1:(solver.n_scalar)
            solver.delz[k] = solver.Rd_lin[k] - solver.delz[k]
        end
        if kind === :predict
            @inbounds for k = 1:(solver.n_scalar)
                solver.delx[k] = -solver.x[k] - solver.W_lin[k] * solver.delz[k]
            end
        elseif kind === :correct
            @inbounds for k = 1:(solver.n_scalar)
                solver.delx[k] =
                    σμ * solver.Si_lin[k] - solver.x[k] - solver.W_lin[k] * solver.delz[k] +
                    solver.RNT_lin[k]
            end
        end
    end
    return
end

# Eigenvalue-based fraction-to-boundary on matrix blocks combined with scalar
# fraction-to-boundary  α = -τ / min(δx ./ x)  on the LP block.
function fraction_to_boundary(solver, τ::T) where {T}
    αmin = one(T)
    βmin = one(T)
    for b in solver.blocks
        LinearAlgebra.mul!(b.buf_n, b.Gi, b.delX)
        LinearAlgebra.mul!(b.buf_n2, b.buf_n, b.Gi')
        @inbounds for q = 1:(b.n), p = 1:(b.n)
            b.buf_n2[p, q] *= b.DDsi[p] * b.DDsi[q]
        end
        @inbounds for q = 1:(b.n), p = 1:(q-1)
            v = (b.buf_n2[p, q] + b.buf_n2[q, p]) / 2
            b.buf_n2[p, q] = v
            b.buf_n2[q, p] = v
        end
        λX = LinearAlgebra.eigmin(LinearAlgebra.Symmetric(b.buf_n2))
        αi = λX > -eps(T)^T(1//2) ? one(T) : min(one(T), -τ / λX)
        αmin = min(αmin, αi)

        LinearAlgebra.mul!(b.buf_n, b.G', b.delS)
        LinearAlgebra.mul!(b.buf_n2, b.buf_n, b.G)
        @inbounds for q = 1:(b.n), p = 1:(b.n)
            b.buf_n2[p, q] *= b.DDsi[p] * b.DDsi[q]
        end
        @inbounds for q = 1:(b.n), p = 1:(q-1)
            v = (b.buf_n2[p, q] + b.buf_n2[q, p]) / 2
            b.buf_n2[p, q] = v
            b.buf_n2[q, p] = v
        end
        λS = LinearAlgebra.eigmin(LinearAlgebra.Symmetric(b.buf_n2))
        βi = λS > -eps(T)^T(1//2) ? one(T) : min(one(T), -τ / λS)
        βmin = min(βmin, βi)
    end
    if solver.n_scalar > 0
        @inbounds for k = 1:(solver.n_scalar)
            dx, x = solver.delx[k], solver.x[k]
            if dx < zero(T) && x > zero(T)
                αmin = min(αmin, -τ * x / dx)
            end
            dz, z = solver.delz[k], solver.z[k]
            if dz < zero(T) && z > zero(T)
                βmin = min(βmin, -τ * z / dz)
            end
        end
    end
    return αmin, βmin
end

# Mehrotra second-order correction (matrix + scalar).
function update_RNT!(solver)
    for b in solver.blocks
        LinearAlgebra.mul!(b.buf_n, b.Gi, b.delX)
        LinearAlgebra.mul!(b.RNT, b.buf_n, b.delS)
        LinearAlgebra.mul!(b.buf_n, b.RNT, b.G)

        LinearAlgebra.mul!(b.RNT, b.G', b.delS)
        LinearAlgebra.mul!(b.buf_n2, b.RNT, b.delX)
        LinearAlgebra.mul!(b.RNT, b.buf_n2, b.Gi')

        @inbounds for q = 1:(b.n), p = 1:(b.n)
            num = -(b.buf_n[p, q] + b.RNT[p, q])
            den = b.D[p] + b.D[q]
            b.RNT[p, q] = num / den
        end
    end
    if solver.n_scalar > 0
        @inbounds for k = 1:(solver.n_scalar)
            solver.RNT_lin[k] = -solver.delx[k] * solver.delz[k] * solver.Si_lin[k]
        end
    end
    return
end

# Trial complementarity at affine point.
function affine_complementarity(solver, α::T, β::T) where {T}
    s = zero(T)
    n = 0
    for b in solver.blocks
        s += LinearAlgebra.dot(b.X, b.S)
        s += β * LinearAlgebra.dot(b.X, b.delS)
        s += α * LinearAlgebra.dot(b.delX, b.S)
        s += α * β * LinearAlgebra.dot(b.delX, b.delS)
        n += b.n
    end
    if solver.n_scalar > 0
        @inbounds for k = 1:(solver.n_scalar)
            xk = solver.x[k] + α * solver.delx[k]
            zk = solver.z[k] + β * solver.delz[k]
            s += xk * zk
        end
        n += solver.n_scalar
    end
    return s / max(n, 1)
end
