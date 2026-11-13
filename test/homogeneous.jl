module TestHomogeneous
using Test, LinearAlgebra
import JuMP, MadSDP
import MathOptInterface as MOI
function factory()
    return MOI.OptimizerWithAttributes(
        MadSDP.Optimizer,
        "algorithm"=>MadSDP.HomogeneousPrimalDual(),
        "schur"=>MadSDP.SparseKKT(),
    )
end
@testset "Homogeneous scalar constraints" begin
    for inequality in (false, true)
        model=JuMP.Model(factory())
        JuMP.set_silent(model)
        JuMP.@variable(model, x)
        if inequality
            JuMP.@constraint(model, x>=1)
        else
            JuMP.@constraint(model, x==1)
        end
        JuMP.@objective(model, Min, x)
        JuMP.optimize!(model)
        @test JuMP.termination_status(model)==MOI.OPTIMAL
        @test JuMP.value(x) ≈ 1 atol=1e-6
    end
end
@testset "Homogeneous SOC and PSD" begin
    model=JuMP.Model(factory())
    JuMP.set_silent(model)
    JuMP.@variable(model, t)
    JuMP.@variable(model, x)
    JuMP.@variable(model, -2 <= y <= 3)
    JuMP.@constraint(model, x==1)
    JuMP.@constraint(model, [t, x, y] in JuMP.SecondOrderCone())
    JuMP.@constraint(model, [t x; x 2] in JuMP.PSDCone())
    JuMP.@objective(model, Min, t+0.1y)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model)==MOI.OPTIMAL
    @test JuMP.objective_value(model) ≈ sqrt(0.99) atol=1e-6
    optimizer=JuMP.unsafe_backend(model)
    inner=MadSDP.MadSDPSolver(
        optimizer.model;
        algorithm = MadSDP.HomogeneousPrimalDual(),
        schur = MadSDP.SparseKKT(),
    )
    MadSDP.initialize!(inner)
    initial_primal=copy(inner.Rp)
    initial_equalities=copy(inner.schur.equality.residual)
    initial_psd=[copy(b.Rd) for b in inner.blocks]
    initial_soc=[copy(b.Rd) for b in inner.schur.soc]
    initial_scalar=copy(inner.Rd_lin)
    initial_gap=MadSDP.primal_objective(inner)-MadSDP.dual_objective(inner)+inner.state.kappa
    total=MadSDP.total_complementarity(inner)+inner.state.tau*inner.state.kappa
    function normalization()
        value=-dot(initial_primal, inner.y)+dot(
            initial_equalities,
            inner.schur.equality.eta,
        )+dot(initial_scalar, inner.x)-initial_gap*inner.state.tau
        for (r, b) in zip(initial_psd, inner.blocks)
            value+=dot(r, b.X)
        end
        for (r, b) in zip(initial_soc, inner.schur.soc)
            value+=dot(r, b.X)
        end
        return value
    end
    @test normalization() ≈ -total
    for _ = 1:3
        MadSDP.iterate!(inner)
        @test normalization() ≈ -total rtol=1e-7
    end
end
@testset "Homogeneous infeasibility and recession certificates" begin
    infeasible=JuMP.Model(factory())
    JuMP.set_silent(infeasible)
    JuMP.@variable(infeasible, x)
    JuMP.@constraint(infeasible, x>=1)
    JuMP.@constraint(infeasible, x<=0)
    JuMP.optimize!(infeasible)
    @test JuMP.termination_status(infeasible)==MOI.INFEASIBLE
    @test JuMP.dual_status(infeasible)==MOI.INFEASIBILITY_CERTIFICATE
    unbounded=JuMP.Model(factory())
    JuMP.set_silent(unbounded)
    JuMP.@variable(unbounded, y)
    JuMP.@objective(unbounded, Min, y)
    JuMP.optimize!(unbounded)
    @test JuMP.termination_status(unbounded)==MOI.DUAL_INFEASIBLE
    @test JuMP.primal_status(unbounded)==MOI.INFEASIBILITY_CERTIFICATE
    soc=JuMP.Model(factory())
    JuMP.set_silent(soc)
    JuMP.@variable(soc, t)
    JuMP.@constraint(soc, t==0)
    JuMP.@constraint(soc, [t, 1.0] in JuMP.SecondOrderCone())
    JuMP.optimize!(soc)
    @test JuMP.termination_status(soc)==MOI.INFEASIBLE
    @test JuMP.dual_status(soc)==MOI.INFEASIBILITY_CERTIFICATE
end
@testset "Homogeneous objective sense and constant" begin
    model=JuMP.Model(factory())
    JuMP.set_silent(model)
    JuMP.@variable(model, x)
    c=JuMP.@constraint(model, x<=1)
    JuMP.@objective(model, Max, 3x+2)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model)==MOI.OPTIMAL
    @test JuMP.objective_value(model) ≈ 5 atol=1e-6
    @test JuMP.dual_objective_value(model) ≈ 5 atol=1e-6
    @test JuMP.dual(c) ≈ -3 atol=1e-6
end
@testset "Inconsistent equality targets remain detectable" begin
    model=JuMP.Model(factory())
    JuMP.set_silent(model)
    JuMP.@variable(model, x)
    JuMP.@constraint(model, x==1)
    JuMP.@constraint(model, 2x==3)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model)==MOI.INFEASIBLE
    @test JuMP.dual_status(model)==MOI.INFEASIBILITY_CERTIFICATE
end
end
