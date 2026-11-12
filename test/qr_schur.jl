module TestQRSchur

using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO
import MadNLP
import MadSDP

function example(; scalars = 0, T = Float64)
    C = sparse(T[3 0.1 0; 0.1 2 0.2; 0 0.2 1])
    A = [sparse(T[i == j == k for i = 1:3, j = 1:3]) for _ = 1:1, k = 1:3]
    d = sparse(ones(T, scalars))
    L = sparse(reshape(T.(1:3scalars), 3, scalars))
    return LRO.DualModel([C], A, ones(T, 3), d, L, [3])
end

@testset "QR Newton directions" begin
    for scalars in (0, 2), diagonal in (true, false)
        model = example(; scalars)
        solvers = [
            MadSDP.MadSDPSolver(model; schur, initial_point = MadSDP.ScaledIdentity())
            for schur in (MadSDP.DenseCholesky(), MadSDP.QRSchur())
        ]
        for s in solvers
            MadSDP.initialize!(s)
            s.blocks[1].X .=
                diagonal ? Matrix(Diagonal([2.0, 3.0, 4.0])) :
                [2.0 0.3 0.1; 0.3 3.0 0.4; 0.1 0.4 4.0]
            s.blocks[1].S .=
                diagonal ? Matrix(Diagonal([4.0, 2.0, 3.0])) :
                [4.0 -0.2 0.1; -0.2 2.0 0.3; 0.1 0.3 3.0]
            MadSDP.update_residuals!(s)
            @test MadSDP._prepare!(s, s.algorithm)
            @test MadSDP._factorize_schur!(s)
        end
        for corrected in (false, true)
            μ = corrected ? 0.3 : 0.0
            for s in solvers
                if corrected
                    MadSDP.correction!(s)
                end
                MadSDP._solve_directions!(s, μ, corrected)
                b = only(s.blocks)
                lhs =
                    [dot(LRO.jac(model, j, LRO.MatrixIndex(1)), b.work.delX) for j = 1:3] + model.C_lin * s.delx
                @test lhs ≈ s.Rp atol = 1e-10
                dual = sum(LRO.jac(model, j, LRO.MatrixIndex(1)) * s.dely[j] for j = 1:3)
                @test b.work.delS + dual ≈ b.Rd atol = 1e-10
                W = b.work.dir.W
                center = μ * inv(b.S) - b.X
                if corrected
                    center += b.work.corr
                end
                @test b.work.delX + W * b.work.delS * W ≈ center atol = 1e-10
            end
            a, b = solvers
            @test a.dely ≈ b.dely atol = 1e-10
            @test a.delx ≈ b.delx atol = 1e-10
            @test a.delz ≈ b.delz atol = 1e-10
            @test a.blocks[1].work.delX ≈ b.blocks[1].work.delX atol = 1e-10
        end
        if diagonal && scalars == 0
            F = solvers[2].schur.factor
            @test F.prow != collect(1:size(F.Q, 1))
        end
    end
end

@testset "QR solves PSD and LP problems" begin
    for model in (
        example(),
        example(; scalars = 2),
        LRO.DualModel(
            SparseMatrixCSC{Float64,Int}[],
            Matrix{SparseMatrixCSC{Float64,Int}}(undef, 0, 1),
            [1.0],
            sparse([1.0, 2.0]),
            sparse([1.0 1.0]),
            Int[],
        ),
    )
        s = MadSDP.MadSDPSolver(
            model;
            schur = MadSDP.QRSchur(),
            initial_point = MadSDP.ScaledIdentity(),
            step = MadSDP.CholeskyBacktrack(),
        )
        MadSDP.solve!(s; verbose = false)
        @test s.status == MadNLP.SOLVE_SUCCEEDED
        @test max(s.inf_pr, s.inf_du, s.inf_compl) <= s.opt.tol
    end
end

@testset "Unsupported QR configurations" begin
    @test_throws ArgumentError MadSDP.MadSDPSolver(
        example();
        schur = MadSDP.QRSchur(),
        direction = MadSDP.HKM(),
    )
    @test_throws ArgumentError MadSDP.MadSDPSolver(
        example();
        schur = MadSDP.QRSchur(),
        algorithm = MadSDP.DualCentering(),
    )
    @test_throws ArgumentError MadSDP.MadSDPSolver(
        example(; T = Float32);
        schur = MadSDP.QRSchur(),
    )
    # An unconstrained multiplier gives a zero column of the weighted Jacobian.
    model = LRO.DualModel(
        [sparse(ones(1, 1))],
        reshape([spzeros(1, 1)], 1, 1),
        [0.0],
        spzeros(0),
        spzeros(1, 0),
        [1],
    )
    s = MadSDP.MadSDPSolver(model; schur = MadSDP.QRSchur())
    MadSDP.solve!(s; verbose = false)
    @test s.status == MadNLP.ERROR_IN_STEP_COMPUTATION
end

@testset "QR rank-deficient systems" begin
    for (d, matrices) in (
        (1, [ones(1, 1), ones(1, 1)]),
        (2, [Matrix(Diagonal([1.0, 0.0])), Matrix(Diagonal([1.0, 0.0]))]),
    )
        model = LRO.DualModel(
            [sparse(Matrix(1.0I, d, d))],
            reshape(sparse.(matrices), 1, 2),
            ones(2),
            spzeros(0),
            spzeros(2, 0),
            [d],
        )
        s = MadSDP.MadSDPSolver(model; schur = MadSDP.QRSchur())
        MadSDP.solve!(s; verbose = false)
        @test s.status == MadNLP.ERROR_IN_STEP_COMPUTATION
    end
end

function iteration_allocations(solver)
    return @allocated begin
        MadSDP.iterate!(solver)
        MadSDP.update_termination!(solver)
    end
end

@testset "Existing Schur iteration allocations" begin
    for schur in (MadSDP.DenseCholesky(), MadSDP.MadNLPSchur())
        solver = MadSDP.MadSDPSolver(example(; scalars = 2); schur)
        MadSDP.initialize!(solver)
        iteration_allocations(solver)
        @test solver.status == MadNLP.REGULAR
        @test iteration_allocations(solver) == 0
    end
end

@testset "Schur workspace policy" begin
    model = example(; scalars = 2)
    full = LRO.BufferedDualModel(model, 8)
    lean = LRO.BufferedDualModel(model, 8; schur = false)
    for input in (model, full, lean),
        schur in (MadSDP.QRSchur(), MadSDP.DenseCholesky(), MadSDP.MadNLPSchur())

        solver = MadSDP.MadSDPSolver(input; schur, step = MadSDP.CholeskyBacktrack())
        @test solver.model.model === model
        @test isnothing(solver.model.schur_buffer) == (schur isa MadSDP.QRSchur)
        if input isa LRO.BufferedDualModel
            @test solver.model.jprod_buffer === input.jprod_buffer
            @test solver.model.jtprod_buffer === input.jtprod_buffer
        end
        MadSDP.solve!(solver; verbose = false)
        @test solver.status == MadNLP.SOLVE_SUCCEEDED
        @test max(solver.inf_pr, solver.inf_du, solver.inf_compl) <= solver.opt.tol
    end
end

end
