module Koala
using LinearAlgebra, IterativeSolvers, LinearMaps, SparseArrays, COSMO, JuMP, Cbc

LP_solver = Cbc.Optimizer
SDP_solver = (eps_abs) -> optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => eps_abs, "eps_rel" => eps_abs)

include("graphutils.jl")
include("SDP.jl")
include("utils.jl")

include("Problems/problems.jl")
include("Strategies/strategies.jl")
include("Bounds/bounds.jl")

end # module Koala
