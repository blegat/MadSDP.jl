# Tests for the dual-centered predictor-corrector algorithm (Nesterov, 2025).

using Test
using LinearAlgebra
using SparseArrays
import LowRankOpt as LRO
import MadNLP
using MadSDP

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

function solve_dual(model; kws...)
    solver = MadSDPSolver(model; algorithm = DualCentering(), kws...)
    solve!(solver; verbose = false)
    return solver
end

@testset "dual: trivial 2×2 SDP" begin
    # min ⟨C, X⟩ s.t. X[1,1] = 1, X ⪰ 0, C = diag(2, 1) ≻ 0 → optimum 2.
    C = Matrix{Float64}([2.0 0.0; 0.0 1.0])
    A1 = Matrix{Float64}([1.0 0.0; 0.0 0.0])
    model = build_sdp_dual(C, [A1], [1.0])
    s = solve_dual(model; tol = 1e-7, max_iter = 200)
    @test s.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(MadSDP.dual_objective(s), 2.0; atol = 1e-4)
    @test isapprox(MadSDP.primal_objective(s), 2.0; atol = 1e-4)
end

@testset "dual: 2×2 SDP with C ≻ 0 (trace=1)" begin
    # min trace(C X) s.t. tr(X) = 1, X ⪰ 0,  C = diag(3, 2) → optimum λ_min = 2.
    C = Matrix{Float64}([3.0 0.0; 0.0 2.0])
    A1 = Matrix{Float64}([1.0 0.0; 0.0 1.0])
    model = build_sdp_dual(C, [A1], [1.0])
    s = solve_dual(model; tol = 1e-7, max_iter = 200)
    @test s.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(MadSDP.dual_objective(s), 2.0; atol = 1e-4)
end

function maxcut_model(W)
    L = Matrix(Diagonal(vec(sum(W, dims = 2))) - W)
    n = size(W, 1)
    As = [Matrix{Float64}([k1 == k == k2 for k1 = 1:n, k2 = 1:n]) for k = 1:n]
    return build_sdp_dual(-L / 4, As, ones(n)), L
end

@testset "dual: maxcut 4-cycle (C ⊁ 0 → needs DualShift)" begin
    W = Float64[
        0 1 0 1
        1 0 1 0
        0 1 0 1
        1 0 1 0
    ]
    model, _ = maxcut_model(W)
    s = solve_dual(model; tol = 1e-6, max_iter = 300)
    @test s.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(MadSDP.dual_objective(s), -4.0; atol = 1e-3)
    # The dual iterates stay feasible.
    @test s.inf_du < 1e-10
end

@testset "dual: maxcut LRO example (val ≈ -18)" begin
    W = Float64[
        0 5 7 6
        5 0 0 1
        7 0 0 1
        6 1 1 0
    ]
    model, L = maxcut_model(W)
    s = solve_dual(model; tol = 1e-6, max_iter = 300)
    @test s.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(MadSDP.dual_objective(s), -18.0; atol = 1e-3)
    # Same with the dual feasible start of `mc_psd`, the theoretical `A` of
    # (7.11) and the `MadNLPSchur` solver.
    y0 = -(1 .+ vec(sum(abs, L / 4, dims = 1)))
    s = MadSDPSolver(
        model;
        algorithm = DualCentering(A = nothing),
        initial_point = InitialPoint([Matrix(1.0I, 4, 4)], y0),
        schur = MadNLPSchur(),
        tol = 1e-6,
        max_iter = 300,
    )
    solve!(s; verbose = false)
    @test s.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(MadSDP.dual_objective(s), -18.0; atol = 1e-3)
end

@testset "dual: infeasible initial point" begin
    model, _ = maxcut_model(ones(2, 2) - I)
    @test_throws ErrorException solve!(
        MadSDPSolver(
            model;
            algorithm = DualCentering(),
            initial_point = InitialPoint([Matrix(1.0I, 2, 2)], [1.0, 1.0]),
        );
        verbose = false,
    )
    @test_throws ErrorException solve!(
        MadSDPSolver(
            model;
            algorithm = DualCentering(),
            initial_point = DualShift(1.0, 0.1),
        );
        verbose = false,
    )
end

@testset "dual: scalar block rejected" begin
    # Build a model with one scalar variable to verify we error out.
    Cmat = reshape([1.0], 1, 1)
    A1 = reshape([1.0], 1, 1)
    d_lin = SparseVector{Float64,Int64}(1, Int64[1], Float64[2.0])
    C_lin = SparseMatrixCSC{Float64,Int64}(sparse(Float64[1 0;][:, 1:1]))
    model = LRO.Model([sparse(Cmat)], reshape([sparse(A1)], 1, 1), [1.0], d_lin, C_lin, [1])
    @test_throws ErrorException MadSDPSolver(model; algorithm = DualCentering())
end

# `DualShift` is also a dual feasible start for `PrimalDual`.
@testset "PrimalDual from DualShift" begin
    model, _ = maxcut_model(ones(4, 4) - I)
    s = MadSDPSolver(model; direction = HKM(), initial_point = DualShift())
    solve!(s; verbose = false)
    @test s.status == MadNLP.SOLVE_SUCCEEDED
    # The SDP value of max-cut on K₄ is 4, with X = (4I - J) / 3.
    @test isapprox(MadSDP.primal_objective(s), -4.0; atol = 1e-4)
end
