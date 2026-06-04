using Test
using LinearAlgebra
using SparseArrays
import LowRankOpt as LRO
import NLPModels
import MadNLP
using MadSDP

primal_obj(solver) = MadSDP.primal_objective(solver)

# Build an LRO.Model from a single dense PSD block, given block cost `C` and a
# list of constraint matrices `Aj` with RHS `b_j`.
function build_sdp(
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

# Mixed model with one PSD block + an LP block of size n_scalar.
function build_mixed(
    C::AbstractMatrix{T},
    As::Vector{<:AbstractMatrix{T}},
    d_lin_vec::Vector{T},
    C_lin_mat::AbstractMatrix{T},
    b::Vector{T},
) where {T}
    n = LinearAlgebra.checksquare(C)
    m = length(As)
    @assert m == length(b) == size(C_lin_mat, 1)
    @assert size(C_lin_mat, 2) == length(d_lin_vec)
    Amat = Matrix{SparseMatrixCSC{T,Int64}}(undef, 1, m)
    for j = 1:m
        Amat[1, j] = sparse(As[j])
    end
    nz = findall(!iszero, d_lin_vec)
    d_lin =
        SparseVector{T,Int64}(length(d_lin_vec), Int64.(nz), T[d_lin_vec[i] for i in nz])
    C_lin = SparseMatrixCSC{T,Int64}(sparse(C_lin_mat))
    return LRO.Model([sparse(C)], Amat, b, d_lin, C_lin, [n])
end

# Tiny SDP : min ⟨C, X⟩ s.t. X[1,1] = 1, X ⪰ 0, with C = diag(2, 1).
# Optimal X = e_1 e_1^T, value = 2.
@testset "trivial 2×2 SDP" begin
    C = Matrix{Float64}([2.0 0.0; 0.0 1.0])
    A1 = Matrix{Float64}([1.0 0.0; 0.0 0.0])
    model = build_sdp(C, [A1], [1.0])

    solver = MadSDPSolver(model; tol = 1e-8, max_iter = 60)
    solve!(solver; verbose = false)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(primal_obj(solver), 2.0; atol = 1e-5)
end

# Asymmetric 4-node weighted maxcut from the LRO README (known SDP value ≈ 18).
@testset "maxcut LRO example" begin
    W = Float64[
        0 5 7 6;
        5 0 0 1;
        7 0 0 1;
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
    model = build_sdp(C, As, b)

    solver = MadSDPSolver(model; tol = 1e-7, max_iter = 80)
    solve!(solver; verbose = false)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(primal_obj(solver), -18.0; atol = 1e-4)
end

# Same problem, but pre-wrap in BufferedModelForSchur to exercise that path.
@testset "BufferedModelForSchur input" begin
    C = Matrix{Float64}([3.0 0.0; 0.0 2.0])
    A1 = Matrix{Float64}([1.0 0.0; 0.0 1.0])
    model = build_sdp(C, [A1], [1.0])
    buffered = LRO.BufferedModelForSchur(model, 1e-12)

    solver = MadSDPSolver(buffered; tol = 1e-8, max_iter = 60)
    solve!(solver; verbose = false)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
    # min trace(C*X) s.t. trace(X) = 1, X ⪰ 0 ; optimum = λ_min(C) = 2.
    @test isapprox(primal_obj(solver), 2.0; atol = 1e-5)
end

# Mixed LP : min t + 2 x₁ + x₂  s.t.  t + x₁ = 1,  t + x₂ = 1,  t ≥ 0, x ≥ 0,
# encoded as an SDP block of size 1 (variable t) plus two LP variables.  At
# the optimum t = 1, x₁ = x₂ = 0, value = 1.  (Two constraints, two scalars
# exercises the C_lin path nontrivially.)
@testset "mixed LP + 1×1 SDP" begin
    Cmat = reshape([1.0], 1, 1)                  # cost on t : ⟨[1], [t]⟩ = t
    A1 = reshape([1.0], 1, 1)                  # constraint 1 picks up t
    A2 = reshape([1.0], 1, 1)                  # constraint 2 picks up t
    d_lin = [2.0, 1.0]                           # scalar costs
    C_lin = Float64[1 0; 0 1]                    # constraint 1 ↔ x₁, constraint 2 ↔ x₂
    b = [1.0, 1.0]
    model = build_mixed(Cmat, [A1, A2], d_lin, C_lin, b)

    solver = MadSDPSolver(model; tol = 1e-7, max_iter = 60)
    solve!(solver; verbose = false)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(primal_obj(solver), 1.0; atol = 1e-5)
    # Optimum is at the vertex t = 1, x = 0.
    @test isapprox(solver.blocks[1].X[1, 1], 1.0; atol = 1e-3)
    @test all(<(1e-3), solver.x)
end

# 4-cycle maxcut SDP : known SDP value = 4.  We minimise ⟨-L/4, X⟩, so the
# optimal objective is -4.
@testset "maxcut 4-cycle" begin
    W = Float64[
        0 1 0 1;
        1 0 1 0;
        0 1 0 1;
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
    model = build_sdp(C, As, b)

    solver = MadSDPSolver(model; tol = 1e-7, max_iter = 80)
    solve!(solver; verbose = false)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
    @test isapprox(primal_obj(solver), -4.0; atol = 1e-4)
end

@testset "dual_solver" begin
    include("dual_runtests.jl")
end

@testset "MOI_wrapper" begin
    include("MOI_wrapper.jl")
end
