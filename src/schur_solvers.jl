# Factorizations of the Schur complement, see `AbstractSchurSolver`.

mutable struct DenseCholeskyWorkspace{T}
    H::Matrix{T}
    F::LinearAlgebra.Cholesky{T,Matrix{T}}
end

function schur_workspace(::DenseCholesky, ::Type{T}, m) where {T}
    H = zeros(T, m, m)
    return DenseCholeskyWorkspace(H, LinearAlgebra.Cholesky(H, 'L', 0))
end

function factorize!(ws::DenseCholeskyWorkspace)
    H = LinearAlgebra.Hermitian(ws.H, :L)
    ws.F = LinearAlgebra.cholesky!(H; check = false)
    return LinearAlgebra.issuccess(ws.F)
end

schur_solve!(ws::DenseCholeskyWorkspace, rhs) = LinearAlgebra.ldiv!(ws.F, rhs)

function schur_workspace(s::MadNLPSchur, ::Type{T}, m) where {T}
    return SchurSystem{T}(m; linear_solver = s.linear_solver)
end

function factorize!(sys::SchurSystem)
    try
        MadNLP.factorize!(sys.linear_solver)
    catch err
        err isa LinearAlgebra.PosDefException ||
            err isa LinearAlgebra.SingularException ||
            rethrow()
        return false
    end
    return true
end

schur_solve!(sys::SchurSystem, rhs) = solve_inplace!(sys, rhs)
