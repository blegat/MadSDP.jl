# Per-PSD-block state for the primal-dual NT IPM.
#
# Maintained variables (for block `i` of side dimension `n_i`):
#   X, S         primal/dual PSD matrices ; central path: X*S = μ I
#   delX, delS   search directions
#   W, G, Gi     NT scaling : W = G*G' = X^{1/2}(X^{1/2} S X^{1/2})^{-1/2} X^{1/2}
#   Si           cached S^{-1}
#   D            singular values from prepare_W
#   DDsi         1 ./ sqrt.(diag(G' S G)) — used to symmetrize the step length test
#   RNT          Mehrotra second-order correction matrix
#   Rd           dual residual  C - S - A^*(y)  (kept at 0 by maintaining S = C - A^*(y))
#   Rc           centrality residual buffer
#   buf_n        scratch matrix
struct SDPBlock{T}
    n::Int
    X::Matrix{T}
    S::Matrix{T}
    delX::Matrix{T}
    delS::Matrix{T}
    W::Matrix{T}
    G::Matrix{T}
    Gi::Matrix{T}
    Si::Matrix{T}
    D::Vector{T}
    DDsi::Vector{T}
    RNT::Matrix{T}
    Rd::Matrix{T}
    buf_n::Matrix{T}
    buf_n2::Matrix{T}
end

function SDPBlock{T}(n::Integer) where {T}
    return SDPBlock{T}(
        Int(n),
        Matrix{T}(undef, n, n),  # X
        Matrix{T}(undef, n, n),  # S
        Matrix{T}(undef, n, n),  # delX
        Matrix{T}(undef, n, n),  # delS
        Matrix{T}(undef, n, n),  # W
        Matrix{T}(undef, n, n),  # G
        Matrix{T}(undef, n, n),  # Gi
        Matrix{T}(undef, n, n),  # Si
        Vector{T}(undef, n),     # D
        Vector{T}(undef, n),     # DDsi
        zeros(T, n, n),          # RNT (initialised to 0)
        Matrix{T}(undef, n, n),  # Rd
        Matrix{T}(undef, n, n),  # buf_n
        Matrix{T}(undef, n, n),  # buf_n2
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
