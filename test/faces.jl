module TestFaces
using Test, LinearAlgebra
import MadSDP
include("../experiments/face_restriction.jl")

struct FailingSchur <: MadSDP.AbstractSchurSolver end
function MadSDP.schur_workspace(::FailingSchur, ::Type{T}, m) where {T}
    throw(SingularException(1))
end

@testset "Face coordinate maps and dependent equations" begin
    e = face_example()
    V = Matrix(1.0I, 5, 5)[:, 1:2]
    Y = [2.0 0.3; 0.3 1.0]
    X = MadSDP.face_lift(V, Y)
    @test MadSDP.face_project(V, X) ≈ Y
    @test dot(e.C, X) ≈ dot(MadSDP.face_project(V, e.C), Y)
    for a in e.A
        @test dot(a, X) ≈ dot(MadSDP.face_project(V, a), Y)
    end
    r = MadSDP.restrict_face(e.C, e.A, e.b, e.support)
    @test size(r.U) == (2, 1)
    @test norm(e.b - r.U * (r.U' * e.b)) < 1e-15
    @test isnothing(MadSDP.restrict_face(e.C, e.A, [1.0, 1.0], e.support))
    @test isnothing(MadSDP.restrict_face(e.C, [zeros(5, 5)], [0.0], e.support))
    rotated = face_example(; rotated = true)
    Vdense = Matrix(qr(hcat(rotated.support, [0.0, 1, 0, 0, 0])).Q)[:, 1:2]
    @test MadSDP.face_project(Vdense, MadSDP.face_lift(Vdense, Y)) ≈ Y
    @test_throws ArgumentError MadSDP.restrict_face(e.C, e.A, e.b, 2V)
    mixed = LRO.Model(
        [sparse(e.C)],
        reshape(sparse.(e.A), 1, 2),
        e.b,
        sparse([1.0]),
        spzeros(2, 1),
        [5],
    )
    @test_throws ArgumentError MadSDP.face_data(mixed)
    @test_throws ArgumentError MadSDP.adaptive_face(
        MadSDP.face_model(e.C, e.A, e.b);
        face = MadSDP.FaceOptions(; history = 0),
    )
end

@testset "Original-space verification detects hidden dual errors" begin
    e = face_example()
    # Reduced slack zero; full slack indefinite due to its cross block.
    y = e.y + [0.0, 0.1]
    S = e.C - sum(y[i] * e.A[i] for i = 1:2)
    @test abs(only(e.support' * S * e.support)) < 1e-15
    r = MadSDP.face_residuals(e.C, e.A, e.b, e.X, y, S)
    @test r.dual_psd > 1e-4
    @test r.complementarity > 1e-3
    @test r.gap < 1e-15
    @test MadSDP.face_eigmin(BigFloat.([1.0 0.5; 0.5 -1.0])) ≈ -sqrt(big"1.25")
    bad = MadSDP.face_residuals(e.C, e.A, e.b, e.X, e.y, e.S + I)
    @test bad.dual > 0.1
    @test MadSDP.face_merit(evaluate(e, e.X, e.y, e.S)) < 1e-16
    dense = face_example(; rotated = true)
    r = MadSDP.refine_rank_one(dense.C, dense.A, dense.b, dense.X, dense.y)
    @test MadSDP.face_merit(evaluate(dense, r.X, r.y, r.S)) < 1e-14
    ref = stored_reference(dense)
    @test MadSDP.face_merit(ref.residuals) < big"1e-70"
    @test norm(ref.X - BigFloat.(dense.X)) < big"1e-14"
    @test MadSDP.face_eigmin(BigFloat.(dense.S)) ≈ BigFloat(eigmin(Symmetric(dense.S))) atol =
        1e-15
end

@testset "Adaptive trials and rollback" begin
    e = face_example()
    model = MadSDP.face_model(e.C, e.A, e.b)
    conservative = MadSDP.adaptive_face(model; tol = 1e-14)
    @test !isempty(conservative.events)
    @test size(first(conservative.events).candidate.V, 2) == 2
    @test MadSDP.face_merit(evaluate(e, conservative.X, conservative.y, conservative.S)) <
          1e-13
    aggressive =
        MadSDP.adaptive_face(model; tol = 1e-14, face = MadSDP.FaceOptions(; discard = 0.9))
    @test !first(aggressive.events).candidate.accepted
    @test first(aggressive.events).candidate.reason == :inconsistent
    rollback = MadSDP.adaptive_face(
        model;
        tol = 1e-14,
        face = MadSDP.FaceOptions(; max_restarts = 0),
    )
    @test aggressive.X == rollback.X
    @test aggressive.y == rollback.y
    @test aggressive.S == rollback.S
    failed = MadSDP.try_face_candidate(
        e.C,
        e.A,
        e.b,
        e.support,
        Matrix(1.0I, 5, 5),
        zeros(2);
        schur = FailingSchur(),
    )
    @test failed.reason == :numerical_failure
    @test !failed.accepted
    oracle = MadSDP.face_candidate(
        e.C,
        e.A,
        e.b,
        e.support,
        Matrix(1.0I, 5, 5),
        zeros(2);
        tol = 1e-14,
    )
    @test oracle.accepted
    @test MadSDP.face_merit(evaluate(e, oracle.X, oracle.y, oracle.S)) < 1e-14
    # Trial construction must preserve original data.
    bounded = MadSDP.adaptive_face(model; tol = 1e-14, bounds = true)
    @test !isnothing(bounded.dual_bound)
    @test bounded.dual_bound == MadSDP.face_bound(e.C, e.A, e.b, bounded.bound_y)
    @test bounded.dual_bound <= Rational{BigInt}(e.y[1])
    @test isnothing(MadSDP.face_bound(e.C, e.A, e.b, e.y + [0.0, 0.1]))
    @test isnothing(MadSDP.face_bound(e.C, e.A, e.b, e.y + [1.0, 0.0]))
    @test MadSDP.face_bound(e.C, e.A, e.b, e.y) == Rational{BigInt}(e.y[1])
    rotating = MadSDP.adaptive_face(
        MadSDP.face_model(
            face_example(; rotated = true).C,
            face_example(; rotated = true).A,
            e.b,
        );
        tol = 1e-16,
        rotate = true,
    )
    @test any(
        ev -> hasproperty(ev.candidate, :rotated) && ev.candidate.rotated,
        rotating.events,
    )
    @test MadSDP.face_merit(rotating.residuals) <= 1e-16
    direct = MadSDP.adaptive_face(
        model;
        tol = 1e-14,
        refine = false,
        face = MadSDP.FaceOptions(; projector_svd = false),
    )
    @test !isempty(direct.events)
    @test size(first(direct.events).candidate.V, 2) == 2
    rounded = MadSDP.face_round(e.A, e.b, Matrix(1.0I, 5, 5), 2e.X)
    @test norm([dot(a, rounded) for a in e.A] - e.b) < 1e-14
    cut = maxcut_example()
    @test MadSDP.face_bound(cut.C, cut.A, cut.b, cut.y) == -18
    @test count(>(1e-8), eigvals(Symmetric(cut.S))) == 3
    @test cut.S * cut.support ≈ zeros(4, 1) atol = 1e-15
    @test MadSDP.face_data(model) == (e.C, e.A, e.b)
end
end
