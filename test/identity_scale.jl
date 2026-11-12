module TestIdentityScale

using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO
import MadSDP

function example(; scalars = 0, T = Float64)
    C = sparse(T[3 0.1 0; 0.1 2 0.2; 0 0.2 1])
    A = [sparse(T[i == j == k for i = 1:3, j = 1:3]) for _ = 1:1, k = 1:3]
    d = sparse(ones(T, scalars))
    L = sparse(reshape(T.(1:3scalars), 3, scalars))
    return LRO.Model([C], A, ones(T, 3), d, L, [3])
end

@testset "Explicit identity scale" begin
    for scale in (1, 0.2, 10.0)
        s = MadSDP.MadSDPSolver(
            example(; scalars = 2);
            initial_point = MadSDP.ScaledIdentity(scale),
        )
        MadSDP.initialize!(s)
        @test s.blocks[1].X == scale * I(3)
        @test s.blocks[1].S == scale * I(3)
        @test s.x == s.z == fill(scale, 2)
        @test iszero(s.y)
    end
    for scale in (0, -1.0, Inf, NaN)
        @test_throws ArgumentError MadSDP.ScaledIdentity(scale)
    end
end

@testset "Automatic identity scale is preserved" begin
    model = example(; scalars = 2)
    model.b .*= 1000
    solver = MadSDP.MadSDPSolver(model)
    MadSDP.initialize!(solver)
    expected = sqrt(15000.0)
    @test solver.blocks[1].X ≈ expected * I(3)
    @test solver.blocks[1].S ≈ expected * I(3)
    @test solver.x ≈ fill(expected, 2)
    @test solver.z ≈ fill(expected, 2)
end

end
