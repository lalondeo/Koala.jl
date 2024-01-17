module Koala
using LinearAlgebra, IterativeSolvers, LinearMaps, SparseArrays, COSMO, JuMP

#LP_solver = () -> (cplex_available ? CPLEX.Optimizer : HiGHS.Optimizer)

include("graphutils.jl")
include("SDP.jl")

include("Problems/problems.jl")
#include("Strategies/strategies.jl")
include("Bounds/bounds.jl")

end # module Koala
