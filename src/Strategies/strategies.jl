
module Strategies

include("../utils.jl")
import ..Problems
import ..LP_solver
import ..SDP_solver

include("blackbox.jl")
include("classical_value.jl")
include("yao.jl")
include("entangled_value.jl")
include("nonsignalling.jl")

end