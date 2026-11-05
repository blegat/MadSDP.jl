using Test
using LinearAlgebra
import MadSDP

function cholesky_allocations(L, M)
    return @allocated MadSDP._cholesky_factor!(L, M)
end

@testset "Cholesky factor" begin
    M = [4.0 1.0; 3.0 2.0]
    L = similar(M)
    @test MadSDP._cholesky_factor!(L, M)
    @test L * L' ≈ (M + M') / 2
    @test iszero(L[1, 2])
    @test M == [4.0 1.0; 3.0 2.0]
    cholesky_allocations(L, M)
    @test cholesky_allocations(L, M) == 0

    for N in ([1.0 2.0; 2.0 1.0], zeros(2, 2))
        @test !MadSDP._cholesky_factor!(L, N)
        @test cholesky_allocations(L, N) == 0
    end
end
