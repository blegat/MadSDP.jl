# Tests for the dual-centered predictor-corrector solver (Nesterov, 2025).

using Test
using LinearAlgebra
using SparseArrays
import LowRankOpt as LRO
import MadNLP
using MadSDP

dual_primal_obj(s) = s.obj_prim
dual_dual_obj(s) = s.obj_dual

function build_sdp_dual(
    C::AbstractMatrix{T},
    As::Vector{<:AbstractMatrix{T}},
    b::Vector{T},
) where {T}
    n = LinearAlgebra.checksquare(C)
    m = length(As)
    @assert m == length(b)
    Amat = Matrix{SparseMatrixCSC{T,Int64}}(undef, 1, m)
    for j = 1:m
        Amat[1, j] = sparse(As[j])
    end
    d_lin = SparseVector{T,Int64}(0, Int64[], T[])
    C_lin = SparseMatrixCSC{T,Int64}(m, 0, Int64[1], Int64[], T[])
    return LRO.Model([sparse(C)], Amat, b, d_lin, C_lin, [n])
end

@testset "dual: trivial 2×2 SDP" begin
    # min ⟨C, X⟩ s.t. X[1,1] = 1, X ⪰ 0, C = diag(2, 1) ≻ 0 → optimum 2.
    C = Matrix{Float64}([2.0 0.0; 0.0 1.0])
    A1 = Matrix{Float64}([1.0 0.0; 0.0 0.0])
    model = build_sdp_dual(C, [A1], [1.0])
    s = MadDualSDPSolver(model; tol = 1e-7, max_iter = 200)
    solve!(s; verbose = false)
    @test s.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(dual_dual_obj(s), 2.0; atol = 1e-4)
    @test isapprox(dual_primal_obj(s), 2.0; atol = 1e-4)
end

@testset "dual: 2×2 SDP with C ≻ 0 (trace=1)" begin
    # min trace(C X) s.t. tr(X) = 1, X ⪰ 0,  C = diag(3, 2) → optimum λ_min = 2.
    C = Matrix{Float64}([3.0 0.0; 0.0 2.0])
    A1 = Matrix{Float64}([1.0 0.0; 0.0 1.0])
    model = build_sdp_dual(C, [A1], [1.0])
    s = MadDualSDPSolver(model; tol = 1e-7, max_iter = 200)
    solve!(s; verbose = false)
    @test s.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(dual_dual_obj(s), 2.0; atol = 1e-4)
end

@testset "dual: maxcut 4-cycle (C ⊁ 0 → needs shift heuristic)" begin
    W = Float64[
        0 1 0 1
        1 0 1 0
        0 1 0 1
        1 0 1 0
    ]
    D = Diagonal(vec(sum(W, dims = 2)))
    L = Matrix(D - W)
    C = -L / 4
    n = 4
    As = [
        Matrix{Float64}([k1 == k && k2 == k ? 1.0 : 0.0 for k1 = 1:n, k2 = 1:n]) for k = 1:n
    ]
    b = ones(n)
    model = build_sdp_dual(C, As, b)
    s = MadDualSDPSolver(model; tol = 1e-6, max_iter = 300)
    solve!(s; verbose = false)
    @test s.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(dual_dual_obj(s), -4.0; atol = 1e-3)
end

@testset "dual: maxcut LRO example (val ≈ -18)" begin
    W = Float64[
        0 5 7 6
        5 0 0 1
        7 0 0 1
        6 1 1 0
    ]
    D = Diagonal(vec(sum(W, dims = 2)))
    L = Matrix(D - W)
    C = -L / 4
    n = 4
    As = [
        Matrix{Float64}([k1 == k && k2 == k ? 1.0 : 0.0 for k1 = 1:n, k2 = 1:n]) for k = 1:n
    ]
    b = ones(n)
    model = build_sdp_dual(C, As, b)
    s = MadDualSDPSolver(model; tol = 1e-6, max_iter = 300)
    solve!(s; verbose = false)
    @test s.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(dual_dual_obj(s), -18.0; atol = 1e-3)
end

@testset "dual: scalar block rejected" begin
    # Build a model with one scalar variable to verify we error out.
    Cmat = reshape([1.0], 1, 1)
    A1 = reshape([1.0], 1, 1)
    d_lin_vec = [2.0]
    nzd = findall(!iszero, d_lin_vec)
    d_lin = SparseVector{Float64,Int64}(
        length(d_lin_vec),
        Int64.(nzd),
        Float64[d_lin_vec[i] for i in nzd],
    )
    C_lin = SparseMatrixCSC{Float64,Int64}(sparse(Float64[1 0;][:, 1:1]))
    model = LRO.Model([sparse(Cmat)], reshape([sparse(A1)], 1, 1), [1.0], d_lin, C_lin, [1])
    @test_throws ErrorException MadDualSDPSolver(model)
end
