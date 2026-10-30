# Search directions, see `AbstractDirection`.

# Nesterov–Todd scaling.
#
# Given primal X ≻ 0 and dual S ≻ 0, the NT scaling W is the unique symmetric
# positive definite matrix satisfying  W S W = X.  Equivalently, with Cholesky
# factors X = Lx Lx' and S = Ls Ls', if  Ls' Lx = U Σ V'  is the SVD, then
#     G = Lx V Σ^{-1/2},     W = G G',
# so that G⁻¹ X G⁻ᵀ = Gᵀ S G = Σ.
# We obtain `V` and `Σ` from the eigendecomposition
# `(Ls' Lx)' (Ls' Lx) = Lx' S Lx = V Σ² V'` which can be computed with a
# reusable LAPACK workspace. On the central path, `Lx' S Lx = μ I` so
# squaring does not deteriorate the conditioning.
#
# Reference : Loraine.jl/src/prepare_W.jl.
struct NTWorkspace{T,E}
    W::Matrix{T}
    G::Matrix{T}
    Gi::Matrix{T}
    Lx::Matrix{T}
    M::Matrix{T}
    N::Matrix{T}
    RNT::Matrix{T}
    tmp::Matrix{T}
    D::Vector{T}
    eigen::E
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
        mat(),
        zeros(T, n),
        LRO.EigenWorkspace(mat()),
    )
end

function prepare_direction!(ws::NTWorkspace{T}, b::SDPBlock) where {T}
    _cholesky_factor!(ws.Lx, b.X) || return false
    Lx = LinearAlgebra.LowerTriangular(ws.Lx)
    # M = Ls' * Lx
    copyto!(ws.M, ws.Lx)
    LinearAlgebra.lmul!(LinearAlgebra.LowerTriangular(b.Ls)', ws.M)
    # N = M' * M = V Σ² V'
    LinearAlgebra.mul!(ws.N, ws.M', ws.M)
    _symmetrize!(ws.N)
    λ, V = LRO.eigen!(ws.eigen, ws.N; vectors = true)
    @inbounds for k in eachindex(ws.D)
        λ[k] > zero(T) || return false
        ws.D[k] = sqrt(λ[k])
    end
    # G = Lx V Σ^{-1/2}
    copyto!(ws.G, V)
    LinearAlgebra.lmul!(Lx, ws.G)
    # Gi = Σ^{1/2} V' Lx⁻¹
    LinearAlgebra.transpose!(ws.Gi, V)
    @inbounds for k in axes(ws.G, 2)
        scale = sqrt(ws.D[k])
        for i in axes(ws.G, 1)
            ws.G[i, k] /= scale
            ws.Gi[k, i] *= scale
        end
    end
    LinearAlgebra.rdiv!(ws.Gi, Lx)
    LinearAlgebra.mul!(ws.W, ws.G, ws.G')
    _symmetrize!(ws.W)
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
# `corr = G RNT Gᵀ`.
function correction!(corr, ws::NTWorkspace, b::SDPBlock)
    LinearAlgebra.mul!(ws.tmp, ws.Gi, b.delX)
    LinearAlgebra.mul!(ws.RNT, ws.tmp, b.delS)
    LinearAlgebra.mul!(ws.M, ws.RNT, ws.G)
    LinearAlgebra.mul!(ws.tmp, ws.G', b.delS)
    LinearAlgebra.mul!(ws.RNT, ws.tmp, b.delX)
    LinearAlgebra.mul!(ws.tmp, ws.RNT, ws.Gi')
    @inbounds for q in axes(ws.RNT, 2), p in axes(ws.RNT, 1)
        ws.RNT[p, q] = -(ws.M[p, q] + ws.tmp[p, q]) / (ws.D[p] + ws.D[q])
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
    LinearAlgebra.mul!(ws.tmp, b.delX, b.delS)
    LinearAlgebra.mul!(corr, ws.tmp, b.Si, -1, false)
    return _symmetrize!(corr)
end
