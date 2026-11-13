module TestNTScaling
using Test, LinearAlgebra
import MadSDP

@testset "NT scaling near the cone boundary" begin
    algorithm = MadSDP.PrimalDual(MadSDP.NT(), MadSDP.EigenStep(), MadSDP.Mehrotra())
    for T in (Float32, Float64)
        b = MadSDP.SDPBlock{T}(algorithm, 2)
        tiny = T == Float32 ? T(1e-3) : T(1e-8)
        Q = T[cos(0.2) -sin(0.2); sin(0.2) cos(0.2)]
        b.X .= Diagonal(T[tiny, 1])
        b.S .= Q*Diagonal(T[tiny, 1])*Q'
        @test MadSDP.prepare_block!(b)
        @test MadSDP.prepare_direction!(b.work.dir, b)
        w = b.work.dir
        @test all(>(0), w.D)
        tolerance = T == Float32 ? T(1e-4) : T(1e-7)
        @test w.Gi*b.X*w.Gi' ≈ Diagonal(w.D) rtol=tolerance
        @test w.G'*b.S*w.G ≈ Diagonal(w.D) rtol=tolerance
        @test w.W*b.S*w.W ≈ b.X rtol=tolerance
    end
end
end
