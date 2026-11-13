# Search directions, see `AbstractDirection`.

# Nesterov–Todd scaling.
#
# Given primal X ≻ 0 and dual S ≻ 0, the NT scaling W is the unique symmetric
# positive definite matrix satisfying  W S W = X.  Equivalently, with Cholesky
# factors X = Lx Lx' and S = Ls Ls', if  Ls' Lx = U Σ V'  is the SVD, then
#     G = Lx V Σ^{-1/2},     W = G G',
# so that G⁻¹ X G⁻ᵀ = Gᵀ S G = Σ.
# Compute the SVD directly: forming M'M would square its condition number
# and can lose positive singular values near the cone boundary.
#
# Reference: Loraine.jl/src/prepare_W.jl.
struct NTWorkspace{T,E}
    W::Matrix{T}
    G::Matrix{T}
    Gi::Matrix{T}
    Lx::Matrix{T}
    M::Matrix{T}
    RNT::Matrix{T}
    tmp::Matrix{T}
    D::Vector{T}
    svd::E
end

function direction_workspace(::NT, ::Type{T}, n) where {T}
    mat() = zeros(T, n, n)
    return NTWorkspace(
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        zeros(T, n),
        FastLapackInterface.SVDsvdWs(mat(); jobu = 'N', jobvt = 'A'),
    )
end

prepare_direction!(ws, b, normal_equations::Bool) = prepare_direction!(ws, b)
function prepare_direction!(
    ws::NTWorkspace{T},
    b::SDPBlock,
    normal_equations::Bool = true,
) where {T}
    _cholesky_factor!(ws.Lx, b.X) || return false
    Lx = LinearAlgebra.LowerTriangular(ws.Lx)
    # M = Ls' * Lx
    copyto!(ws.M, ws.Lx)
    LinearAlgebra.lmul!(LinearAlgebra.LowerTriangular(b.Ls)', ws.M)
    _, singular_values, Vt = LinearAlgebra.LAPACK.gesvd!(ws.svd, 'N', 'A', ws.M)
    copyto!(ws.D, singular_values)
    if any(<=(zero(T)), ws.D)
        return false
    end
    # G = Lx V Σ⁻¹ᐟ² and Gi = Σ¹ᐟ² V' Lx⁻¹.
    LinearAlgebra.transpose!(ws.G, Vt)
    LinearAlgebra.lmul!(Lx, ws.G)
    copyto!(ws.Gi, Vt)
    @inbounds for k in axes(ws.G, 2)
        scale = sqrt(ws.D[k])
        for i in axes(ws.G, 1)
            ws.G[i, k] /= scale
            ws.Gi[k, i] *= scale
        end
    end
    LinearAlgebra.rdiv!(ws.Gi, Lx)
    if normal_equations
        LinearAlgebra.mul!(ws.W, ws.G, ws.G')
        _symmetrize!(ws.W)
    end
    return true
end

schur_scaling(ws::NTWorkspace, ::SDPBlock) = (ws.W, ws.W)

# `out = W M W`
function apply_E!(out, ws::NTWorkspace, ::SDPBlock, M)
    LinearAlgebra.mul!(ws.tmp, ws.W, M)
    LinearAlgebra.mul!(out, ws.tmp, ws.W)
    return _symmetrize!(out)
end

# Mehrotra correction in the scaled space where `G⁻¹ X G⁻ᵀ = Gᵀ S G = Σ`:
# `RNT[p, q] = -(G⁻¹ ΔX ΔS G + Gᵀ ΔS ΔX G⁻ᵀ)[p, q] / (Σ[p] + Σ[q])` and
# `corr = G RNT Gᵀ`. As `ΔX` and `ΔS` are symmetric, the second term is the
# transpose of the first one.
correction!(corr, ws, b, normal_equations::Bool) = correction!(corr, ws, b)
function correction!(corr, ws::NTWorkspace, b::SDPBlock, normal_equations::Bool = true)
    if !normal_equations
        # RootKKT retains both directions in NT coordinates. Their Jordan
        # product avoids undoing and reapplying the ill-conditioned scaling.
        LinearAlgebra.mul!(ws.RNT, ws.M, ws.tmp)
        for q in axes(ws.RNT, 2), p = 1:q
            value = -(ws.RNT[p, q]+ws.RNT[q, p])/(ws.D[p]+ws.D[q])
            ws.RNT[p, q] = ws.RNT[q, p] = value
        end
        return corr
    end
    LinearAlgebra.mul!(ws.tmp, ws.Gi, b.work.delX)
    LinearAlgebra.mul!(ws.RNT, ws.tmp, b.work.delS)
    LinearAlgebra.mul!(ws.M, ws.RNT, ws.G)
    @inbounds for q in axes(ws.RNT, 2), p in axes(ws.RNT, 1)
        ws.RNT[p, q] = -(ws.M[p, q] + ws.M[q, p]) / (ws.D[p] + ws.D[q])
    end
    LinearAlgebra.mul!(ws.tmp, ws.G, ws.RNT)
    LinearAlgebra.mul!(corr, ws.tmp, ws.G')
    return _symmetrize!(corr)
end

# HKM: linearizing `(X + ΔX) (S + ΔS) = μ I` and solving for ΔX gives
# `ΔX = μ S⁻¹ - X - X ΔS S⁻¹ - ΔX ΔS S⁻¹`, which is then symmetrized.
struct HKMWorkspace{T}
    tmp::Matrix{T}
end

direction_workspace(::HKM, ::Type{T}, n) where {T} = HKMWorkspace(zeros(T, n, n))

prepare_direction!(::HKMWorkspace, ::SDPBlock) = true

schur_scaling(::HKMWorkspace, b::SDPBlock) = (b.Si, b.X)

# `out = (X M S⁻¹ + S⁻¹ M X) / 2`. If `M` is sparse, e.g., the diagonal
# `A*(Δy)` of max-cut, `X M` is cheap and only `(X M) S⁻¹` is a dense product.
function apply_E!(out, ws::HKMWorkspace, b::SDPBlock, M)
    LinearAlgebra.mul!(ws.tmp, b.X, M)
    LinearAlgebra.mul!(out, ws.tmp, b.Si)
    return _symmetrize!(out)
end

# `corr = -(ΔX ΔS S⁻¹ + S⁻¹ ΔS ΔX) / 2`
function correction!(corr, ws::HKMWorkspace, b::SDPBlock)
    LinearAlgebra.mul!(ws.tmp, b.work.delX, b.work.delS)
    LinearAlgebra.mul!(corr, ws.tmp, b.Si, -1, false)
    return _symmetrize!(corr)
end
