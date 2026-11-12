module TestBoundedConstraints
using Test, LinearAlgebra
import JuMP
import MadSDP
import LowRankOpt as LRO
import NLPModels
import MathOptInterface as MOI

@testset "Native bounded scalar rows" begin
    for upper in (false, true), interval in (false, true)
        m = JuMP.Model(MadSDP.Optimizer)
        JuMP.set_silent(m)
        JuMP.@variable(m, x)
        c = if interval
            JuMP.@constraint(m, 1 <= 2x+3 <= 7)
        elseif upper
            JuMP.@constraint(m, 2x+3 <= 7)
        else
            JuMP.@constraint(m, 2x+3 >= 1)
        end
        if upper
            JuMP.@objective(m, Max, 3x+2)
        else
            JuMP.@objective(m, Min, 3x+2)
        end
        JuMP.optimize!(m)
        opt = JuMP.unsafe_backend(m)
        @test JuMP.termination_status(m) == MOI.OPTIMAL
        @test JuMP.value(x) ≈ (upper ? 2 : -1) atol=1e-5
        @test JuMP.dual(c) ≈ (upper ? -1.5 : 1.5) atol=1e-5
        @test opt.model.meta.nvar == 1
        @test opt.model.meta.ncon == 1
        @test opt.solver.inner.schur isa MadSDP.RootKKTWorkspace
        @test norm(
            NLPModels.grad(opt.model, opt.solver.stats.solution) + NLPModels.jtprod(
                opt.model,
                opt.solver.stats.solution,
                opt.solver.stats.multipliers,
            ),
            Inf,
        ) < 1e-5
        @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ JuMP.objective_value(m) atol=1e-5
    end
end

@testset "Equality with and without PSD blocks" begin
    for psd in (false, true)
        m = JuMP.Model(MadSDP.Optimizer)
        JuMP.set_silent(m)
        JuMP.@variable(m, x)
        c = JuMP.@constraint(m, 2x+3 == 7)
        if psd
            JuMP.@variable(m, t)
            JuMP.@constraint(m, [t x; x t] in JuMP.PSDCone())
            JuMP.@objective(m, Min, t+1)
        else
            JuMP.@objective(m, Min, x+1)
        end
        JuMP.optimize!(m)
        opt = JuMP.unsafe_backend(m)
        @test JuMP.termination_status(m) == MOI.OPTIMAL
        @test JuMP.value(x) ≈ 2 atol=1e-5
        @test JuMP.objective_value(m) ≈ 3 atol=1e-5
        @test JuMP.dual(c) ≈ 0.5 atol=1e-5
        @test length(MadSDP.equality_block(opt.solver.inner.schur).eta) == 1
        @test opt.solver.inner.n_scalar == 0
        @test norm(
            NLPModels.grad(opt.model, opt.solver.stats.solution) + NLPModels.jtprod(
                opt.model,
                opt.solver.stats.solution,
                opt.solver.stats.multipliers,
            ),
            Inf,
        ) < 1e-5
        @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ 3 atol=1e-5
    end
end
@testset "Equality-implied bounds stay outside the barrier" begin
    m = JuMP.Model(MadSDP.Optimizer)
    JuMP.set_silent(m)
    JuMP.@variable(m, 0 <= x <= 1)
    JuMP.@variable(m, y)
    JuMP.@constraint(m, y == 0)
    JuMP.@constraint(m, x+y == 1)
    JuMP.@objective(m, Min, x+2y)
    JuMP.optimize!(m)
    o = JuMP.unsafe_backend(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.value(x) ≈ 1 atol=1e-7
    @test JuMP.value(y) ≈ 0 atol=1e-7
    @test o.solver.inner.n_scalar == 0
    @test o.model.dim.num_scalars == 4
    @test norm(
        NLPModels.grad(o.model, o.solver.stats.solution) +
        NLPModels.jtprod(o.model, o.solver.stats.solution, o.solver.stats.multipliers),
        Inf,
    ) < 1e-7
end

@testset "Redundant equalities use KKT regularization" begin
    m = JuMP.Model(MadSDP.Optimizer)
    JuMP.set_silent(m)
    JuMP.set_attribute(m, "schur", MadSDP.RootKKT())
    JuMP.@variable(m, x)
    JuMP.@constraint(m, x == 1)
    JuMP.@constraint(m, 2x == 2)
    JuMP.@objective(m, Min, x)
    JuMP.optimize!(m)
    o = JuMP.unsafe_backend(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.value(x) ≈ 1 atol=1e-7
    @test o.solver.inner.schur.regularization > 0
    @test norm(
        NLPModels.grad(o.model, o.solver.stats.solution) +
        NLPModels.jtprod(o.model, o.solver.stats.solution, o.solver.stats.multipliers),
        Inf,
    ) < 1e-7
end

end
