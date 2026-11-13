module TestRootKKT
using Test, LinearAlgebra, SparseArrays
import JuMP
import MadSDP
import MadNLP
import NLPModels
import MathOptInterface as MOI

@testset "Root KKT linear solver modules" begin
    for schur in (
        MadSDP.RootKKT(MadNLP.UmfpackSolver),
        MadSDP.RootKKT(MadNLP.MumpsSolver),
        MadSDP.RootKKT(MadNLP.LDLSolver),
        MadSDP.RootKKT(MadNLP.QDLDLSolver),
        MadSDP.SparseKKT(),
        MadSDP.SparseKKT(
            linear_solver = MadNLP.QDLDLSolver,
            qdldl_static_regularization = 1e-6,
            equilibrate = false,
        ),
        MadSDP.SparseKKT(linear_solver = MadNLP.UmfpackSolver),
    )
        model = JuMP.Model(MadSDP.Optimizer)
        JuMP.set_silent(model)
        JuMP.set_attribute(model, "schur", schur)
        JuMP.@variable(model, t)
        JuMP.@variable(model, x)
        JuMP.@variable(model, -2 <= y <= 3)
        JuMP.@constraint(model, x == 1)
        JuMP.@constraint(model, [t, x, y] in JuMP.SecondOrderCone())
        JuMP.@constraint(model, [t x; x 2] in JuMP.PSDCone())
        JuMP.@objective(model, Min, t + 0.1y)
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == MOI.OPTIMAL
        @test JuMP.objective_value(model) ≈ sqrt(0.99) atol=1e-6
        @test JuMP.value(x) ≈ 1 atol=1e-7
        optimizer=JuMP.unsafe_backend(model)
        stats=optimizer.solver.stats
        @test norm(
            NLPModels.grad(optimizer.model, stats.solution) +
            NLPModels.jtprod(optimizer.model, stats.solution, stats.multipliers),
            Inf,
        ) < 1e-6
        @test optimizer.solver.inner.schur.linear_solver isa schur.linear_solver
        if haskey(schur.options, :equilibrate) && !schur.options.equilibrate
            @test all(isone, optimizer.solver.inner.schur.scale)
        end
    end
end
@testset "Cone Hessian triangular coordinates" begin
    for T in (Float32, Float64), n in (3, 4)
        H=T[1/(i+j-1) for i = 1:n, j = 1:n]+Matrix{T}(I, n, n)
        k=n*(n+1)÷2
        matrix=sparse(tril(ones(T, k, k)))
        indices=zeros(Int, k, k)
        for j = 1:k, p in nzrange(matrix, j)
            indices[matrix.rowval[p], j]=p
        end
        MadSDP._set_psd_hessian!(matrix, indices, H)
        vector=T.(sin.(1:k))
        X=zeros(T, n, n)
        row=0
        for q = 1:n, p = 1:q
            row+=1
            X[p, q]=X[q, p]=vector[row]/(p==q ? one(T) : sqrt(T(2)))
        end
        product=H*X*H
        expected=T[]
        for q = 1:n, p = 1:q
            push!(expected, product[p, q]*(p==q ? one(T) : sqrt(T(2))))
        end
        @test -Symmetric(matrix, :L)*vector ≈ expected
    end
end
@testset "Small and operator eigenvalue steps" begin
    for n in (3, 4, 12)
        L=Matrix(Diagonal(collect(1.0:n)))
        delta=L*Diagonal(vcat(-3.0, ones(n-1)))*L'
        workspace=MadSDP.step_workspace(MadSDP.EigenStep(), Float64, n)
        alpha=MadSDP._eigen_step(workspace, L, delta, 0.95)
        @test alpha ≈ 0.95/3
        @test isposdef(Symmetric(L*L'+alpha*delta))
        @test eigmin(Symmetric(L*L'+(1.01/3)*delta)) < 0
    end
end
end
