# Per-PSD-block state of the primal-dual interior-point method.
#
# For block `i` of side dimension `n`:
#   X, S         primal and dual PSD matrices, central path: X S = μ I
#   delX, delS   search directions
#   Rd           dual residual  C - S - A*(y)
#   Ls           Cholesky factor of S (lower triangle)
#   Si           S⁻¹
#   Lx           Cholesky factor of X, only computed if the step length
#                strategy needs it
#   corr         second-order correction added to ΔX
#   buf, buf2    scratch matrices
#   dir          workspace of the search direction
struct SDPBlock{T,DW}
    n::Int
    X::Matrix{T}
    S::Matrix{T}
    delX::Matrix{T}
    delS::Matrix{T}
    Rd::Matrix{T}
    Ls::Matrix{T}
    Si::Matrix{T}
    Lx::Matrix{T}
    corr::Matrix{T}
    buf::Matrix{T}
    buf2::Matrix{T}
    dir::DW
end

function SDPBlock{T}(direction::AbstractDirection, n::Integer) where {T}
    mat() = zeros(T, n, n)
    return SDPBlock(
        Int(n),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        direction_workspace(direction, T, n),
    )
end

# Trace of primal/dual product over all blocks.
function trace_XS(blocks::Vector{<:SDPBlock{T}}) where {T}
    s = zero(T)
    @inbounds for b in blocks
        s += LinearAlgebra.dot(b.X, b.S)
    end
    return s
end

total_dim(blocks::Vector{<:SDPBlock}) = sum(b -> b.n, blocks; init = 0)

# `A = (A + A') / 2`
function _symmetrize!(A::AbstractMatrix)
    @inbounds for j in axes(A, 2), i = 1:(j-1)
        v = (A[i, j] + A[j, i]) / 2
        A[i, j] = v
        A[j, i] = v
    end
    return A
end

function safe_cholesky!(work::AbstractMatrix{T}, M::AbstractMatrix{T}) where {T}
    copyto!(work, M)
    _symmetrize!(work)
    F = LinearAlgebra.cholesky!(LinearAlgebra.Hermitian(work, :L); check = false)
    if !LinearAlgebra.issuccess(F)
        return nothing
    end
    return F
end

# Cholesky factor of `M` in the lower triangle of `L`, the upper triangle
# being zero. Returns `false` if `M` is not positive definite.
function _cholesky_factor!(L::Matrix{T}, M::Matrix{T}) where {T}
    F = safe_cholesky!(L, M)
    isnothing(F) && return false
    @inbounds for j in axes(L, 2), i = 1:(j-1)
        L[i, j] = zero(T)
    end
    return true
end

# `Si = S⁻¹` from the Cholesky factor `Ls` of `S`.
function _inverse_from_cholesky!(Si::Matrix{T}, Ls::Matrix{T}) where {T}
    copyto!(Si, Ls)
    LinearAlgebra.LAPACK.potri!('L', Si)
    @inbounds for j in axes(Si, 2), i = 1:(j-1)
        Si[i, j] = Si[j, i]
    end
    return Si
end

# Factorizes `S` and computes `S⁻¹`; the quantities every direction needs.
function prepare_block!(b::SDPBlock)
    _cholesky_factor!(b.Ls, b.S) || return false
    _inverse_from_cholesky!(b.Si, b.Ls)
    return true
end

# Whether `X + α ΔX` has a Cholesky factorization, using `buf` as workspace.
function _is_add_mul_posdef(buf, X, α, ΔX)
    @. buf = X + α * ΔX
    F = LinearAlgebra.cholesky!(LinearAlgebra.Hermitian(buf, :L); check = false)
    return LinearAlgebra.issuccess(F)
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

function _copy_to_dense!(Y::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T}
    if X isa SparseArrays.SparseMatrixCSC
        fill!(Y, zero(T))
        _axpy_sparse_dense!(Y, one(T), X)
    else
        copyto!(Y, X)
    end
    return Y
end
