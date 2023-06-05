module Koala

using COSMO
using JuMP
SDP_solver = (eps_abs) -> optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => eps_abs)


cplex_available = true

try 
	using CPLEX
	cplex_available = true
catch
	using HiGHS
	cplex_available = false
end

LP_solver = () -> (cplex_available ? CPLEX.Optimizer : HiGHS.Optimizer)
	

include("Problems/problems.jl")
include("Strategies/strategies.jl")
#include("Limitations/limitations.jl")

end
	