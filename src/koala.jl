module Koala

using COSMO
using JuMP
using Pkg
SDP_solver = (eps_abs) -> optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => eps_abs)


cplex_available = true

try 
	global cplex_available
	using CPLEX
	cplex_available = true
catch
	using HiGHS
	global cplex_available
	cplex_available = false
end

LP_solver = () -> (cplex_available ? CPLEX.Optimizer : HiGHS.Optimizer)
	

include("Problems/problems.jl")
include("Strategies/strategies.jl")
#include("Limitations/limitations.jl")

end
	