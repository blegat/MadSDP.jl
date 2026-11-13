module TestSOC
using Test, LinearAlgebra, SparseArrays
import MadSDP
import LowRankOpt as LRO
import NLPModels
import JuMP
import MathOptInterface as MOI

@testset "Lorentz Jordan algebra and NT scaling" begin
    x, y = [3.0, 1, 0.5], [4.0, -1, 1]
    out, tmp = zeros(3), zeros(3)
    MadSDP._soc_product!(tmp, x, y)
    MadSDP._soc_ldiv!(out, x, tmp)
    @test out ≈ y
    MadSDP._soc_inverse!(out, x)
    MadSDP._soc_product!(tmp, x, out)
    @test tmp ≈ [1, 0, 0]
    for inverse in (false, true)
        MadSDP._soc_power!(out, x, inverse)
        MadSDP._soc_product!(tmp, out, out)
        expected = inverse ? MadSDP._soc_inverse!(zeros(3), x) : x
        @test tmp ≈ expected
    end
    model = LRO.Model(
        LRO.DualModel(Matrix{Float64}[], (), [-1.0], spzeros(0), spzeros(1, 0), Int[]),
    )
    LRO.append_second_order_constraints!(
        model,
        sparse(reshape([1.0, 0, 0], 3, 1)),
        zeros(3),
        [3],
    )
    b = MadSDP.SOCBlock(model, LRO.SOCIndex(1))
    b.X .= x
    b.S .= y
    @test MadSDP.prepare_soc!(b)
    @test b.G*b.G*b.S ≈ b.X
    @test b.Gi*b.X ≈ b.G*b.S
    @test b.G*b.Gi ≈ I(3)
    @test dot(b.lambda, b.lambda) ≈ dot(b.X, b.S)
    for (state, direction) in (
        (x, [-4.0, 1, 0]),
        (x, [-1.0, 0, 0]),
        (x, [1.0, 0, 0]),
        (x, [0.0, 1, 0]),
        (x, [-1.0, 1, 0]),
        (x, [1.0, 1, 0]),
    )
        bound = MadSDP._soc_max_step(state, direction)
        @test bound > 0
        if isfinite(bound)
            @test MadSDP._soc_interior(state+0.999bound*direction)
            @test !MadSDP._soc_interior(state+1.001bound*direction)
        end
    end
end

@testset "Native SOC and rotated SOC solves" begin
    for rotated in (false, true), step in (MadSDP.EigenStep(), MadSDP.CholeskyBacktrack())
        m = JuMP.Model(MadSDP.Optimizer)
        JuMP.set_silent(m)
        JuMP.set_attribute(m, "step", step)
        JuMP.set_attribute(m, "tol", 1e-10)
        JuMP.@variable(m, t)
        constraint = if rotated
            JuMP.@constraint(m, [t, 0.5, 3, 4] in MOI.RotatedSecondOrderCone(4))
        else
            JuMP.@constraint(m, [t, 3, 4] in MOI.SecondOrderCone(3))
        end
        JuMP.@objective(m, Min, t+2)
        JuMP.optimize!(m)
        opt = JuMP.unsafe_backend(m)
        @test JuMP.termination_status(m) == MOI.OPTIMAL
        @test JuMP.value(t) ≈ (rotated ? 25 : 5) atol=1e-5
        @test JuMP.dual(constraint) ≈ (rotated ? [1, 50, -6, -8] : [1, -0.6, -0.8]) atol=1e-4
        @test LRO.num_second_order_cones(opt.model) == 1
        @test LRO.num_matrices(opt.model) == 0
        @test isempty(opt.solver.inner.blocks)
        @test opt.model.meta.nvar == 1
        @test norm(
            NLPModels.grad(opt.model, opt.solver.stats.solution)+NLPModels.jtprod(
                opt.model,
                opt.solver.stats.solution,
                opt.solver.stats.multipliers,
            ),
            Inf,
        ) < 1e-6
        @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ JuMP.objective_value(m) atol=1e-5
    end
end

@testset "Equalities, interval, SOC and PSD together" begin
    m = JuMP.Model(MadSDP.Optimizer)
    JuMP.set_silent(m)
    JuMP.@variable(m, x)
    JuMP.@variable(m, t)
    JuMP.@variable(m, u)
    JuMP.@constraint(m, x == 1)
    JuMP.@constraint(m, 0 <= x <= 3)
    JuMP.@constraint(m, [t, x, 2] in MOI.SecondOrderCone(3))
    JuMP.@constraint(m, [u 1; 1 u] in JuMP.PSDCone())
    JuMP.@objective(m, Min, t+u)
    JuMP.optimize!(m)
    o = JuMP.unsafe_backend(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.objective_value(m) ≈ sqrt(5)+1 atol=1e-5
    @test o.model.meta.nvar == 3
    @test o.model.meta.ncon == 9
    @test o.model.dim.num_scalars == 2
    @test o.model.dim.second_order_dimensions == [3]
    @test o.model.dim.side_dimensions == [2]
    @test norm(
        NLPModels.grad(o.model, o.solver.stats.solution)+NLPModels.jtprod(
            o.model,
            o.solver.stats.solution,
            o.solver.stats.multipliers,
        ),
        Inf,
    ) < 1e-6
end
@testset "Constant SOC feasibility" begin
    for feasible in (false, true)
        m = JuMP.Model(MadSDP.Optimizer)
        JuMP.set_silent(m)
        vector = feasible ? [2.0, 1, 0] : [1.0, 2, 0]
        JuMP.@constraint(m, vector in MOI.SecondOrderCone(3))
        JuMP.@objective(m, Min, 7)
        JuMP.optimize!(m)
        o = JuMP.unsafe_backend(m)
        @test JuMP.termination_status(m) == (feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
        @test o.model.meta.nvar == 0
        @test o.model.meta.ncon == 3
        if feasible
            @test JuMP.objective_value(m) == 7
        end
    end
end

end
