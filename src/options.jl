@kwdef mutable struct MadSDPOptions{T}
    tol::T = T(1e-7)
    feas_tol::T = T(1e-7)
    max_iter::Int = 100
    max_wall_time::T = T(3600)
    print_level::MadNLP.LogLevels = MadNLP.INFO
    output_file::String = ""
    tau::T = T(0.95)
    mu_min::T = T(1e-14)
    sigma_expon::T = T(3)
    initial_alpha::T = T(0.95)
    regularize_schur::T = T(1e-12)
    max_schur_reg::Int = 8
    linear_solver::Type = MadNLP.LapackCPUSolver
end
