# Nesterov–Todd scaling.
#
# Given primal X ≻ 0 and dual S ≻ 0, the NT scaling W is the unique symmetric
# positive definite matrix satisfying  W S W = X.  Equivalently, with Cholesky
# factors X = Lx Lx' and S = Ls Ls', if  Ls' Lx = U Σ V'  is the SVD, then
#     G = Lx V Σ^{-1/2},     W = G G'.
# The diagonal Σ collects the singular values; we also build  Gi = G^{-1}  and
# DDsi[k] = 1/sqrt((G' S G)_{kk}) for use in the step-length test.
#
# Reference : Loraine.jl/src/prepare_W.jl.

function safe_cholesky!(work::AbstractMatrix{T}, M::AbstractMatrix{T}) where {T}
    copyto!(work, M)
    # Symmetrise to avoid stray asymmetry from rounding.
    @inbounds for j in axes(work, 2), i in 1:(j-1)
        work[i, j] = (work[i, j] + work[j, i]) / 2
        work[j, i] = work[i, j]
    end
    F = LinearAlgebra.cholesky!(LinearAlgebra.Hermitian(work, :L); check = false)
    if !LinearAlgebra.issuccess(F)
        return nothing
    end
    return F
end

# Computes the NT factors for one block. Returns `true` on success, `false` if
# either X or S is too close to singular for Cholesky to succeed.
function prepare_W_block!(b::SDPBlock{T}) where {T}
    Fx = safe_cholesky!(b.buf_n, b.X)
    Fx === nothing && return false
    Lx = Matrix(Fx.L)               # n×n lower-triangular

    Fs = safe_cholesky!(b.buf_n2, b.S)
    Fs === nothing && return false
    Ls = Matrix(Fs.L)

    LsT_Lx = Ls' * Lx                # n×n
    svd_result = LinearAlgebra.svd(LsT_Lx)
    U, D, V = svd_result.U, svd_result.S, svd_result.V

    @. b.D = D
    Dinv_sqrt = LinearAlgebra.Diagonal(one(T) ./ sqrt.(D))

    LinearAlgebra.mul!(b.G, Lx * V, Dinv_sqrt)
    # Gi = D^{1/2} V' Lx^{-1} ;  use Lx \ I then multiply.
    LxInv = Lx \ Matrix{T}(LinearAlgebra.I, b.n, b.n)
    Dsqrt = LinearAlgebra.Diagonal(sqrt.(D))
    LinearAlgebra.mul!(b.Gi, Dsqrt * V', LxInv)
    LinearAlgebra.mul!(b.W, b.G, b.G')
    # Symmetrise.
    @inbounds for j in 1:b.n, i in 1:(j-1)
        b.W[i, j] = (b.W[i, j] + b.W[j, i]) / 2
        b.W[j, i] = b.W[i, j]
    end

    # Si = S^{-1} via Cholesky.
    Ls_full = LinearAlgebra.LowerTriangular(Ls)
    Si_buf = Ls_full' \ (Ls_full \ Matrix{T}(LinearAlgebra.I, b.n, b.n))
    copyto!(b.Si, Si_buf)

    # DDsi[k] = 1 / sqrt((G' S G)_{kk}). G' S G has diagonal entries that are
    # exactly D[k] when (X,S) lie on the central path; off-path this gives a
    # well-defined symmetric scaling for the step-length test.
    GtSG = b.G' * b.S * b.G
    @inbounds for k in 1:b.n
        d = GtSG[k, k]
        b.DDsi[k] = d > zero(T) ? one(T) / sqrt(d) : one(T)
    end
    return true
end

function prepare_W!(blocks::Vector{<:SDPBlock})
    for b in blocks
        prepare_W_block!(b) || return false
    end
    return true
end

# my_kron(A, B, C) = B * C * A'.  Loraine names it `my_kron` because
# (A ⊗ B) vec(C) = vec(B C A')  in column-major vec.
function my_kron!(out::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T}
    BC = B * C
    LinearAlgebra.mul!(out, BC, A')
    return out
end
