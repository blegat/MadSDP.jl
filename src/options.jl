# Numerical options of `MadSDPSolver`. The algorithm itself is selected by
# the components of `components.jl`.
@kwdef mutable struct MadSDPOptions{T}
    tol::T = T(1e-7)
    max_iter::Int = 100
    max_wall_time::T = T(3600)
    print_level::MadNLP.LogLevels = MadNLP.INFO
    output_file::String = ""
    mu_min::T = T(1e-14)
    regularize_schur::T = T(1e-12)
    max_schur_reg::Int = 8
    # Threshold `κ` on the number of nonzeros of the sparse constraint
    # matrices above which `LRO.BufferedModelForSchur` uses dense products.
    datasparsity::Int = 8
end
