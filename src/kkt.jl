# Schur-complement KKT for primal-dual SDP IPM.
#
# After NT scaling, the reduced Newton equation in the dual `y` reads
#     M Δy = h,    M_{j,k} = Σ_i ⟨A_{i,j}, W_i A_{i,k} W_i⟩,
# which is *precisely* what `LowRankOpt.schur_complement!` computes.  `M` is
# symmetric positive definite as long as the iterates lie in the interior, so
# we factor it with LAPACK Bunch-Kaufman (robust to mild ill-conditioning at
# the central path's end).  Wrapping factorisation behind MadNLP's
# `AbstractLinearSolver` keeps the GPU path (MadNLPGPU.CUDSSSolver, ...) one
# line away.

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

# Builds  H = Σ_i ⟨A_{i,·}, W_i A_{i,·} W_i⟩  +  reg * I  and factorises it.
# `W` is a `LRO.ShapedSolution` whose matrix views are the NT scalings `W_i`.
function build_and_factorize!(sys::SchurSystem{T}, model, W, reg::T) where {T}
    sys.reg = reg
    LRO.schur_complement!(model, W, sys.H)
    if reg > zero(T)
        @inbounds for i in 1:sys.m
            sys.H[i, i] += reg
        end
    end
    MadNLP.factorize!(sys.linear_solver)
    return sys
end

function solve_inplace!(sys::SchurSystem{T}, rhs::AbstractVector{T}) where {T}
    copyto!(sys.rhs, rhs)
    MadNLP.solve!(sys.linear_solver, sys.rhs)
    copyto!(rhs, sys.rhs)
    return rhs
end
