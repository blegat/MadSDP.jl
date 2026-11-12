module TestTermination

using Test
using SparseArrays
import LowRankOpt as LRO
import MadNLP
import MadSDP

function scalar_solver()
    one = sparse(ones(1, 1))
    model = LRO.Model([one], reshape([one], 1, 1), [1.0], spzeros(0), spzeros(1, 0), [1])
    return MadSDP.MadSDPSolver(model)
end

@testset "Small barrier parameter does not certify optimality" begin
    for (x, s, y) in ((2.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0))
        solver = scalar_solver()
        MadSDP.initialize!(solver)
        solver.blocks[1].X[1, 1] = x
        solver.blocks[1].S[1, 1] = s
        solver.y[1] = y
        solver.mu = 0.0
        MadSDP.update_residuals!(solver)
        MadSDP.update_termination!(solver)
        @test solver.status == MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL
        @test max(solver.inf_pr, solver.inf_du, solver.inf_compl) > solver.opt.tol
    end
    solver = scalar_solver()
    MadSDP.initialize!(solver)
    solver.blocks[1].X[1, 1] = 1.0
    solver.blocks[1].S[1, 1] = 0.0
    solver.y[1] = 1.0
    solver.mu = 0.0
    MadSDP.update_residuals!(solver)
    MadSDP.update_termination!(solver)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
    @test max(solver.inf_pr, solver.inf_du, solver.inf_compl) == 0
    @test MadSDP._solvercore_status(MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL) ==
          :small_step
end

end
