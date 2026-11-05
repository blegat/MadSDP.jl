# Compares MadSDP configured like `mc_psd` with MCPSD, which hardcodes
# `mc_psd`, on random max-cut instances.
#
#     julia --project=benchmark benchmark/maxcut.jl 500 1000 2000
#
# The configuration of MadSDP is the target to beat when designing new
# components: it should take about as many iterations as MCPSD and
# each iteration should not be slower.

using LinearAlgebra, SparseArrays, Random, Printf
import LowRankOpt as LRO
import MathOptInterface as MOI
import MCPSD
using MadSDP

BLAS.set_num_threads(1)

# Laplacian / 4 of a random graph with average degree about 10.
function instance(n; seed = 1)
    rng = MersenneTwister(seed)
    G = sprand(rng, n, n, 10 / n)
    G = triu(G + G', 1)
    G = G + G'
    fill!(nonzeros(G), 1.0)
    return Matrix(Diagonal(vec(sum(G, dims = 2))) - G) / 4
end

# min ⟨-L, X⟩ s.t. Xⱼⱼ = 1, X ⪰ 0
function lro_model(L)
    n = size(L, 1)
    A = [sparse([j], [j], [1.0], n, n) for _ = 1:1, j = 1:n]
    return LRO.Model(
        [sparse(-L)],
        A,
        ones(n),
        sparsevec(Int[], Float64[], 0),
        spzeros(Float64, Int, n, 0),
        [n],
    )
end

# The algorithm of `mc_psd`, starting from its dual feasible point.
function mc_psd_like(L)
    X0 = [Matrix(1.0I, size(L)...)]
    y0 = -(1 .+ vec(sum(abs, L, dims = 1)))
    return (
        direction = HKM(),
        step = CholeskyBacktrack(),
        centering = AdaptiveMu(),
        initial_point = InitialPoint(X0, y0),
    )
end

const CONFIGS = [
    "mc_psd-like" => mc_psd_like,
    "HKM + Mehrotra" => L -> (direction = HKM(),),
    "NT + Mehrotra" => L -> NamedTuple(),
]

# `mc_psd` stops when the gap is below `10^-5.5 |obj|` while MadSDP divides
# the gap by `|pobj| + |dobj| ≈ 2 |obj|`.
const TOL = 2 * 10^-5.5

function madsdp(L, config)
    solver = MadSDPSolver(lro_model(L); tol = TOL, config...)
    time = @elapsed solve!(solver; verbose = false)
    return time, solver.iter, -MadSDP.primal_objective(solver), solver.status
end

function mcpsd(L)
    time = @elapsed (_, y, res) =
        MCPSD.mc_psd(Symmetric(L); silent = true, time_limit_sec = 10_000)
    it = res[MCPSD.NumberOfIterations()]
    return time, it, sum(y), res[MOI.TerminationStatus()]
end

function report(n, name, (time, it, obj, status))
    @printf(
        "n=%-5d %-16s %8.3f s %3d it %8.4f s/it obj %.8f %s\n",
        n,
        name,
        time,
        it,
        time / it,
        obj,
        status,
    )
    return
end

function main(sizes)
    # Compile every code path on a small instance first.
    L = instance(50)
    mcpsd(L)
    for (_, config) in CONFIGS
        madsdp(L, config(L))
    end
    for n in sizes
        L = instance(n)
        report(n, "MCPSD", mcpsd(L))
        for (name, config) in CONFIGS
            report(n, name, madsdp(L, config(L)))
        end
    end
    return
end

main(isempty(ARGS) ? [200, 500, 1000] : parse.(Int, ARGS))
