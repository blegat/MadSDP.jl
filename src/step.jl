# Predictor / corrector RHS and step computation for primal-dual NT.
#
# Convention for the dual slack:  S_i := C_i - Σ_j y_j A_{i,j}.  We do NOT
# enforce this exactly across iterations; instead we track the dual residual
#     R_d[i] = C_i - S_i - Σ_j y_j A_{i,j},
# and a corrector step at full β eliminates it.  This is the standard
# infeasible-IPM mechanism — same as Loraine.

# Builds R_p[j] = b_j − Σ_i ⟨A_{i,j}, X_i⟩ and R_d[i] = C_i − S_i − A^*(y)[i].
function update_residuals!(solver)
    model = solver.model
    T = eltype(solver.Rp)
    # R_p = -A(X) accumulated, then add b.
    fill!(solver.Rp, zero(T))
    for (i, b) in enumerate(solver.blocks)
        LRO.add_jprod!(model, b.X, solver.Rp, LRO.MatrixIndex(i))
    end
    @inbounds for j in eachindex(solver.Rp)
        solver.Rp[j] = LRO.cons_constant(model)[j] - solver.Rp[j]
    end
    # Dual residual per block: R_d[i] = C_i - S_i - A^*(y)[i].
    for (i, b) in enumerate(solver.blocks)
        AT_y = LRO.unsafe_jtprod(model, solver.y, LRO.MatrixIndex(i))
        # b.Rd := C_i (densify if needed, since LRO.grad(model, LRO.MatrixIndex(i)) may be sparse).
        _copy_to_dense!(b.Rd, LRO.grad(model, LRO.MatrixIndex(i)))
        b.Rd .-= b.S
        _axpy_sparse_dense!(b.Rd, -one(T), AT_y)
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
function _axpy_sparse_dense!(Y::AbstractMatrix{T}, α::T, X::SparseArrays.SparseMatrixCSC{T}) where {T}
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

# Predictor RHS:
#   h[j] = R_p[j] + Σ_i ⟨A_{i,j}, W_i (R_d[i] + S_i) W_i⟩.
function predictor_rhs!(h::Vector{T}, solver) where {T}
    model = solver.model
    copyto!(h, solver.Rp)
    for (i, b) in enumerate(solver.blocks)
        # buf = R_d + S
        @. b.buf_n = b.Rd + b.S
        # buf_n2 = W * buf
        LinearAlgebra.mul!(b.buf_n2, b.W, b.buf_n)
        # buf = buf_n2 * W
        LinearAlgebra.mul!(b.buf_n, b.buf_n2, b.W)
        # symmetrise
        @inbounds for q in 1:b.n, p in 1:(q-1)
            v = (b.buf_n[p, q] + b.buf_n[q, p]) / 2
            b.buf_n[p, q] = v
            b.buf_n[q, p] = v
        end
        LRO.add_jprod!(model, b.buf_n, h, LRO.MatrixIndex(i))
    end
    return h
end

# Corrector RHS:
#   h[j] = R_p[j] + Σ_i ⟨A_{i,j}, G_i (G_i' R_d[i] G_i + diag(D_i)
#                                       − σμ ./ D_i − RNT_i) G_i'⟩.
function corrector_rhs!(h::Vector{T}, solver, σμ::T) where {T}
    model = solver.model
    copyto!(h, solver.Rp)
    for (i, b) in enumerate(solver.blocks)
        # M = G' R_d G
        LinearAlgebra.mul!(b.buf_n, b.G', b.Rd)
        LinearAlgebra.mul!(b.buf_n2, b.buf_n, b.G)
        # Add diag(D) − σμ./D − RNT
        @inbounds for k in 1:b.n
            b.buf_n2[k, k] += b.D[k] - σμ / b.D[k]
        end
        b.buf_n2 .-= b.RNT
        # Compute G * buf_n2 * G'
        LinearAlgebra.mul!(b.buf_n, b.G, b.buf_n2)
        LinearAlgebra.mul!(b.buf_n2, b.buf_n, b.G')
        @inbounds for q in 1:b.n, p in 1:(q-1)
            v = (b.buf_n2[p, q] + b.buf_n2[q, p]) / 2
            b.buf_n2[p, q] = v
            b.buf_n2[q, p] = v
        end
        LRO.add_jprod!(model, b.buf_n2, h, LRO.MatrixIndex(i))
    end
    return h
end

# Given δy (already solved), compute δX, δS per block.  `kind` selects the
# variant of δX:
#   :predict  → δX = -X − W δS W
#   :correct  → δX = σμ S^{-1} − X − W δS W + G RNT G'
function compute_directions!(solver, dely::Vector{T}, kind::Symbol, σμ::T) where {T}
    model = solver.model
    for (i, b) in enumerate(solver.blocks)
        AT_dy = LRO.unsafe_jtprod(model, dely, LRO.MatrixIndex(i))
        # δS = R_d − A^*(δy)
        copyto!(b.delS, b.Rd)
        _axpy_sparse_dense!(b.delS, -one(T), AT_dy)
        @inbounds for q in 1:b.n, p in 1:(q-1)
            v = (b.delS[p, q] + b.delS[q, p]) / 2
            b.delS[p, q] = v
            b.delS[q, p] = v
        end

        # Ξ = W δS W
        LinearAlgebra.mul!(b.buf_n, b.W, b.delS)
        LinearAlgebra.mul!(b.buf_n2, b.buf_n, b.W)
        if kind === :predict
            @. b.delX = -b.X - b.buf_n2
        elseif kind === :correct
            # G RNT G'
            LinearAlgebra.mul!(b.buf_n, b.G, b.RNT)
            LinearAlgebra.mul!(b.delX, b.buf_n, b.G')           # GRNTGt
            @. b.delX = σμ * b.Si - b.X - b.buf_n2 + b.delX
        else
            error("unknown kind $kind")
        end
        @inbounds for q in 1:b.n, p in 1:(q-1)
            v = (b.delX[p, q] + b.delX[q, p]) / 2
            b.delX[p, q] = v
            b.delX[q, p] = v
        end
    end
    return
end

# Eigenvalue-based fraction-to-boundary: returns (α_primal, β_dual).  Per block
# we compute λ_min of the symmetric scaled directions and take the conservative
# minimum across blocks.
function fraction_to_boundary(solver, τ::T) where {T}
    αmin = one(T)
    βmin = one(T)
    for b in solver.blocks
        # δXb = Gi δX Gi'
        LinearAlgebra.mul!(b.buf_n, b.Gi, b.delX)
        LinearAlgebra.mul!(b.buf_n2, b.buf_n, b.Gi')
        # XXX_X = Diag(DDsi) δXb Diag(DDsi)
        @inbounds for q in 1:b.n, p in 1:b.n
            b.buf_n2[p, q] *= b.DDsi[p] * b.DDsi[q]
        end
        @inbounds for q in 1:b.n, p in 1:(q-1)
            v = (b.buf_n2[p, q] + b.buf_n2[q, p]) / 2
            b.buf_n2[p, q] = v
            b.buf_n2[q, p] = v
        end
        λX = LinearAlgebra.eigmin(LinearAlgebra.Symmetric(b.buf_n2))
        αi = λX > -eps(T)^T(1//2) ? one(T) : min(one(T), -τ / λX)
        αmin = min(αmin, αi)

        # δSb = G' δS G
        LinearAlgebra.mul!(b.buf_n, b.G', b.delS)
        LinearAlgebra.mul!(b.buf_n2, b.buf_n, b.G)
        @inbounds for q in 1:b.n, p in 1:b.n
            b.buf_n2[p, q] *= b.DDsi[p] * b.DDsi[q]
        end
        @inbounds for q in 1:b.n, p in 1:(q-1)
            v = (b.buf_n2[p, q] + b.buf_n2[q, p]) / 2
            b.buf_n2[p, q] = v
            b.buf_n2[q, p] = v
        end
        λS = LinearAlgebra.eigmin(LinearAlgebra.Symmetric(b.buf_n2))
        βi = λS > -eps(T)^T(1//2) ? one(T) : min(one(T), -τ / λS)
        βmin = min(βmin, βi)
    end
    return αmin, βmin
end

# Computes the Mehrotra second-order correction matrix
#   RNT_i = -(Gi δX δS G + G' δS δX Gi') ./ (D ⊕ D),
# where  (D ⊕ D)_{p,q} = D_p + D_q,  ./ is elementwise division.
function update_RNT!(solver)
    for b in solver.blocks
        LinearAlgebra.mul!(b.buf_n, b.Gi, b.delX)        # Gi δX
        LinearAlgebra.mul!(b.RNT, b.buf_n, b.delS)        # Gi δX δS
        LinearAlgebra.mul!(b.buf_n, b.RNT, b.G)           # Gi δX δS G

        LinearAlgebra.mul!(b.RNT, b.G', b.delS)           # G' δS
        LinearAlgebra.mul!(b.buf_n2, b.RNT, b.delX)       # G' δS δX
        LinearAlgebra.mul!(b.RNT, b.buf_n2, b.Gi')        # G' δS δX Gi'

        @inbounds for q in 1:b.n, p in 1:b.n
            num = -(b.buf_n[p, q] + b.RNT[p, q])
            den = b.D[p] + b.D[q]
            b.RNT[p, q] = num / den
        end
    end
    return
end

# Trial complementarity μ_aff = ⟨X + α δX, S + β δS⟩ / total_dim.
function affine_complementarity(solver, α::T, β::T) where {T}
    s = zero(T)
    n = 0
    for b in solver.blocks
        # ⟨X + α δX, S + β δS⟩ = ⟨X,S⟩ + β⟨X,δS⟩ + α⟨δX,S⟩ + αβ⟨δX,δS⟩
        s += LinearAlgebra.dot(b.X, b.S)
        s += β * LinearAlgebra.dot(b.X, b.delS)
        s += α * LinearAlgebra.dot(b.delX, b.S)
        s += α * β * LinearAlgebra.dot(b.delX, b.delS)
        n += b.n
    end
    return s / max(n, 1)
end
