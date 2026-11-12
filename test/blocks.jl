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

    # Entries far below the rounding of the factor are set to zero so that
    # they cannot become subnormal in the products of the scaling.
    for T in (Float32, Float64)
        F = T[1 0 0; floatmin(T) 1 0; eps(T) 1 1]
        M = F * F'
        L = similar(M)
        @test MadSDP._cholesky_factor!(L, M)
        @test iszero(L[2, 1])
        @test L[3, 1] == eps(T)
        @test L * L' ≈ M
    end
end
