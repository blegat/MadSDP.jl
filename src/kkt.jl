# Workspace of `MadNLPSchur`: the Schur complement `H` factorized by a
# `MadNLP.AbstractLinearSolver`. `H` is symmetric positive definite as long
# as the iterates lie in the interior; we use LAPACK Bunch-Kaufman (robust to
# mild ill-conditioning at the central path's end) by default and the GPU
# path (MadNLPGPU.CUDSSSolver, ...) is one option away.

mutable struct SchurSystem{T,LS<:MadNLP.AbstractLinearSolver{T}}
    m::Int
    H::Matrix{T}            # dense m×m Schur matrix (lower-triangular meaningful)
    rhs::Vector{T}          # buffer for the RHS / solution vector
    linear_solver::LS
    reg::T                  # current regularisation level on the diagonal
end

function SchurSystem{T}(m::Integer; linear_solver::Type = MadNLP.LapackCPUSolver) where {T}
    H = zeros(T, m, m)
    rhs = zeros(T, m)
    opt = MadNLP.default_options(linear_solver)
    # Force a symmetric factorisation; Bunch-Kaufman handles slight indefiniteness.
    if hasproperty(opt, :lapack_algorithm)
        opt.lapack_algorithm = MadNLP.BUNCHKAUFMAN
    end
    ls = linear_solver(H; opt = opt)
    return SchurSystem{T,typeof(ls)}(Int(m), H, rhs, ls, zero(T))
end

function solve_inplace!(sys::SchurSystem{T}, rhs::AbstractVector{T}) where {T}
    copyto!(sys.rhs, rhs)
    MadNLP.solve!(sys.linear_solver, sys.rhs)
    copyto!(rhs, sys.rhs)
    return rhs
end
